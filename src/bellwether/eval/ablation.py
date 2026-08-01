"""Weather ablation: does temperature explain what a foundation model gets wrong.

Three arms, scored on an identical origin set:

1. **base**, the uncorrected foundation model.
2. **base + calendar**, corrected by a residual quantile regression on calendar features
   alone. The control. Rebuilding a predictive distribution from residual quantiles is a
   recalibration, and recalibration alone can move coverage without any weather in it. An
   arm that beats `base` here has demonstrated nothing about temperature.
3. **base + weather**, the same corrector with temperature columns added. Only the gap
   between this and the calendar arm is attributable to weather.

Two properties this file exists to guarantee:

* **No leakage.** The corrector at origin *i* is fitted only on residuals from origins
  strictly before *i*. It is refitted at every origin rather than fitted once, which is
  slower and is the only version of the experiment that means anything.
* **Identical windows.** Every arm is scored on the same origins, which excludes the
  warmup origins the corrector needs before it can produce anything. Comparing a corrected
  arm on 300 windows against a base arm on 360 would compare window sets, not models.

The base model's forecasts are computed once and cached across arms. Refitting the
corrector is cheap; re-running a foundation model over hundreds of origins is not, and
running it three times would change nothing about the result except the wall clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bellwether.eval.backtest import BacktestResult, WindowResult, _median_index
from bellwether.eval.metrics import (
    DEFAULT_QUANTILES,
    interval_coverage,
    interval_width,
    mae,
    mase,
    rmse,
    smape,
    weighted_quantile_loss,
)
from bellwether.forecast.base import Forecaster
from bellwether.forecast.residual import (
    CALENDAR_ONLY,
    DEFAULT_MIN_TRAIN_ORIGINS,
    WEATHER,
    FeatureSpec,
    ResidualQuantileCorrector,
    apply_correction,
    build_features,
)

log = logging.getLogger(__name__)

# A day-over-day temperature change needs yesterday's value, so the first day of the
# series can produce no weather features.
LAG_HOURS = 24


@dataclass(slots=True)
class CachedForecast:
    """One base-model forecast, kept so every arm scores the same numbers."""

    origin: int
    quantiles: np.ndarray  # (horizon, n_quantiles)


def cache_base_forecasts(
    forecaster: Forecaster,
    series: np.ndarray,
    origins: list[int],
    horizon: int,
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
) -> list[CachedForecast]:
    """Run the base model once over every origin."""
    cached = []
    for position, origin in enumerate(origins):
        if position % 50 == 0:
            log.info("base forecast %d/%d", position, len(origins))
        cached.append(
            CachedForecast(
                origin=origin,
                quantiles=forecaster.predict(series[:origin], horizon, quantile_levels),
            )
        )
    return cached


def usable_origins(
    series: np.ndarray,
    temperature: np.ndarray,
    horizon: int,
    initial_train_size: int,
    step: int = 24,
    season_length: int = 168,
) -> list[int]:
    """Origins where demand, its recent history, and the weather features are all present.

    Screened up front so every arm sees the same windows, and so a window is never dropped
    partway through for a reason that applies to only one arm. The weather requirement
    reaches back an extra 24 hours beyond the forecast window, since the day-over-day
    temperature change needs the same hours from the day before.
    """
    origins = []
    for origin in range(initial_train_size, series.size - horizon + 1, step):
        target = slice(origin, origin + horizon)
        lagged = slice(origin - LAG_HOURS, origin + horizon - LAG_HOURS)
        recent_history = series[max(0, origin - season_length * 2) : origin]

        if not np.all(np.isfinite(series[target])):
            continue
        if not np.all(np.isfinite(recent_history)):
            continue
        if origin < LAG_HOURS:
            continue
        if not np.all(np.isfinite(temperature[target])):
            continue
        if not np.all(np.isfinite(temperature[lagged])):
            continue
        origins.append(origin)
    return origins


def _calendar_columns(timestamps: np.ndarray, timezone: str) -> tuple[np.ndarray, np.ndarray]:
    """Local hour of day and a weekend flag for every point on the grid.

    Local rather than UTC: the daily load shape follows the working day where the load is,
    and in UTC a Texas evening peak would land in a different hour depending on the season.
    """
    local = pd.DatetimeIndex(timestamps).tz_localize("UTC").tz_convert(timezone)
    return local.hour.to_numpy(), (local.dayofweek.to_numpy() >= 5)


def run_weather_ablation(
    forecaster: Forecaster,
    series: np.ndarray,
    timestamps: np.ndarray,
    temperature: np.ndarray,
    *,
    series_id: str,
    timezone: str,
    horizon: int = 24,
    initial_train_size: int = 672,
    season_length: int = 168,
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    min_train_origins: int = DEFAULT_MIN_TRAIN_ORIGINS,
    cached: list[CachedForecast] | None = None,
) -> dict[str, BacktestResult]:
    """Score the base model and both corrected arms over one series.

    Returns a result per arm, keyed by model name, all covering the same origins.
    """
    origins = usable_origins(
        series, temperature, horizon, initial_train_size, season_length=season_length
    )
    if len(origins) <= min_train_origins:
        raise ValueError(
            f"{series_id}: {len(origins)} usable origins, need more than the "
            f"{min_train_origins} the corrector spends on warmup"
        )

    if cached is None:
        cached = cache_base_forecasts(forecaster, series, origins, horizon, quantile_levels)

    local_hours, is_weekend = _calendar_columns(timestamps, timezone)
    median_index = _median_index(quantile_levels)

    # Feature rows and residuals for every origin, built once and sliced per fit. The
    # corrector never sees more of this than the origins before the one it is forecasting.
    per_origin_features = {
        spec.name: [
            _features_for(spec, origin, horizon, local_hours, is_weekend, temperature)
            for origin in origins
        ]
        for spec in (CALENDAR_ONLY, WEATHER)
    }
    per_origin_residuals = [
        series[c.origin : c.origin + horizon] - c.quantiles[:, median_index] for c in cached
    ]

    # Warmup origins are excluded from every arm, including the uncorrected one, so the
    # comparison is over one window set rather than three.
    scored = list(range(min_train_origins, len(origins)))
    log.info(
        "%s: %d usable origins, %d scored after %d warmup",
        series_id,
        len(origins),
        len(scored),
        min_train_origins,
    )

    results = {
        forecaster.name: BacktestResult(forecaster.name, series_id, horizon, 0),
        f"{forecaster.name}+calendar": BacktestResult(
            f"{forecaster.name}+calendar", series_id, horizon, 0
        ),
        f"{forecaster.name}+weather": BacktestResult(
            f"{forecaster.name}+weather", series_id, horizon, 0
        ),
    }

    for position in scored:
        origin = origins[position]
        actual = series[origin : origin + horizon]
        history = series[:origin]
        base = cached[position].quantiles

        _score(
            results[forecaster.name],
            origin,
            actual,
            base,
            history,
            season_length,
            quantile_levels,
            median_index,
        )

        train_residuals = np.concatenate(per_origin_residuals[:position])
        for spec in (CALENDAR_ONLY, WEATHER):
            rows = per_origin_features[spec.name]
            corrector = ResidualQuantileCorrector(spec, quantile_levels).fit(
                np.vstack(rows[:position]), train_residuals
            )
            corrected = apply_correction(base[:, median_index], corrector.predict(rows[position]))
            _score(
                results[f"{forecaster.name}+{spec.name}"],
                origin,
                actual,
                corrected,
                history,
                season_length,
                quantile_levels,
                median_index,
            )

    return results


def _features_for(
    spec: FeatureSpec,
    origin: int,
    horizon: int,
    local_hours: np.ndarray,
    is_weekend: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    target = slice(origin, origin + horizon)
    lagged = slice(origin - LAG_HOURS, origin + horizon - LAG_HOURS)
    return build_features(
        spec,
        horizon_steps=np.arange(1, horizon + 1),
        local_hours=local_hours[target],
        is_weekend=is_weekend[target],
        temperature=temperature[target] if spec.use_weather else None,
        temperature_yesterday=temperature[lagged] if spec.use_weather else None,
    )


def _score(
    result: BacktestResult,
    origin: int,
    actual: np.ndarray,
    forecasts: np.ndarray,
    history: np.ndarray,
    season_length: int,
    quantile_levels: tuple[float, ...],
    median_index: int,
) -> None:
    median = forecasts[:, median_index]
    result.windows.append(
        WindowResult(
            origin_index=origin,
            mae=mae(actual, median),
            rmse=rmse(actual, median),
            smape=smape(actual, median),
            mase=mase(actual, median, history, season_length),
            wql=weighted_quantile_loss(actual, forecasts, quantile_levels),
            coverage_80=interval_coverage(actual, forecasts, quantile_levels, 0.1, 0.9),
            width_80=interval_width(forecasts, quantile_levels, 0.1, 0.9),
        )
    )
    result.n_windows += 1
