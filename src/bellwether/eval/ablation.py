"""Corrector ablation: what explains the errors a foundation model makes.

Five arms, scored on an identical origin set. The four corrected ones are a 2x2 of weather
against volatility, which exists because the two answer different questions: weather
columns describe *where* the residual sits, volatility columns describe *how far* it can
stray, and only the second is what an interval is made of.

1. **base**, the uncorrected foundation model.
2. **base + calendar**, a residual quantile regression on calendar features alone. The
   control. Rebuilding a predictive distribution from residual quantiles is itself a
   recalibration, and recalibration alone moves coverage with no covariate in it, so an arm
   that beats `base` has demonstrated nothing until it also beats this.
3. **base + weather**, adding temperature. Cut point error and left calibration alone.
4. **base + volatility**, adding the recent realised volatility of demand.
5. **base + weather+volatility**, both.

Two properties this file exists to guarantee:

* **No leakage.** The corrector at origin *i* is fitted only on residuals from origins
  strictly before *i*. It is refitted at every origin rather than fitted once, which is
  slower and is the only version of the experiment that means anything.
* **Identical windows.** Every arm is scored on the same origins, which excludes the
  warmup origins the corrector needs before it can produce anything. Comparing a corrected
  arm on 300 windows against a base arm on 360 would compare window sets, not models.

The base model's forecasts are computed once and cached across arms. Refitting the
corrector is cheap; re-running a foundation model over hundreds of origins is not, and
running it five times would change nothing about the result except the wall clock.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
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
from bellwether.explain.evidence import (
    WIDELY_OBSERVED_HOLIDAYS,
    holiday_names_in_window,
    holidays_in_window,
)
from bellwether.forecast.base import Forecaster
from bellwether.forecast.residual import (
    ALL_SPECS,
    DEFAULT_MIN_TRAIN_ORIGINS,
    FeatureSpec,
    HolidayClassScaleCorrector,
    HolidayHourScaleCorrector,
    HolidayScaleCorrector,
    QuantileScaleCorrector,
    ResidualQuantileCorrector,
    apply_correction,
    build_features,
)

# The scale arm is not feature-based, so it sits outside the FeatureSpec 2x2. It stretches
# the base model's own interval rather than replacing the distribution, which is the one
# correction here that cannot destroy the base model's conditioning.
SCALE_ARM = "scale"

# The scale arm plus a calendar the base model cannot see. Kept separate from `scale` so
# the holiday effect is measured against a corrector identical in every other respect.
HOLIDAY_ARM = "scale+holiday"

# The same calendar split by how widely each holiday is actually observed. Its control is
# HOLIDAY_ARM rather than SCALE_ARM: the question is whether splitting the shift helps, not
# whether shifting helps, and those were tangled in the first version of this experiment.
HOLIDAY_CLASS_ARM = "scale+holidayclass"

# The same calendar again, shaped over the hours of the day instead of applied flat across
# them. Its control is HOLIDAY_CLASS_ARM, for the same reason: the question is whether the
# shape helps, and it degrades to that arm exactly when no cell has evidence.
HOLIDAY_HOUR_ARM = "scale+holidayhour"

# Observance codes. Ordinary is zero so `codes > 0` recovers the plain holiday flag.
ORDINARY = 0
WIDELY_OBSERVED = 1
FEDERAL_ONLY = 2

log = logging.getLogger(__name__)

# A day-over-day temperature change needs yesterday's value, so the first day of the
# series can produce no weather features.
LAG_HOURS = 24


@dataclass(slots=True)
class CachedForecast:
    """One base-model forecast, kept so every arm scores the same numbers."""

    origin: int
    quantiles: np.ndarray  # (horizon, n_quantiles)


@dataclass(slots=True)
class AblationOutput:
    """Scores per arm, plus the forecasts behind them.

    `scored_origins` and every list in `forecasts` are index-aligned, so position *i* of
    any arm's forecast list belongs to `scored_origins[i]`. Warmup origins appear in
    neither.
    """

    results: dict[str, BacktestResult]
    scored_origins: list[int]
    forecasts: dict[str, list[np.ndarray]]


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
    temperature: np.ndarray | None,
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

    `temperature=None` drops the weather requirement, for experiments whose arms do not
    use it. That roughly doubles the usable period, since NCEI's archive ends eleven months
    before EIA's data does, and it means the run is on a **different window** from every
    weather-scored result. Any comparison across the two compares windows, not models.
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
        if temperature is not None:
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


def holiday_flags(timestamps: np.ndarray, timezone: str) -> np.ndarray:
    """Whether each hour falls on a US federal holiday in market-local time.

    Local, for the same reason as the hour of day: a holiday is a local calendar day, and
    in UTC the boundary would fall several hours into the wrong day.
    """
    local = pd.DatetimeIndex(timestamps).tz_localize("UTC").tz_convert(timezone)
    dates = local.normalize().tz_localize(None)
    holidays = holidays_in_window(dates[0], dates[-1]) if dates.size else pd.DatetimeIndex([])
    return np.isin(dates.to_numpy(), holidays.to_numpy())


def holiday_class_flags(timestamps: np.ndarray, timezone: str) -> np.ndarray:
    """Which observance class each hour falls in: 0 ordinary, 1 widely observed, 2 federal only.

    Codes rather than names, because the corrector estimates one offset per class and the
    class count is what has to stay small. Two classes over two years give each of them
    roughly 250 holiday hours to be estimated from; eleven named holidays would give 48.
    """
    local = pd.DatetimeIndex(timestamps).tz_localize("UTC").tz_convert(timezone)
    dates = local.normalize().tz_localize(None)
    if not dates.size:
        return np.zeros(len(timestamps), dtype=np.int8)

    named = holiday_names_in_window(dates[0], dates[-1])
    classes = {
        date: (WIDELY_OBSERVED if name in WIDELY_OBSERVED_HOLIDAYS else FEDERAL_ONLY)
        for date, name in named.items()
    }
    return np.array([classes.get(d.date(), ORDINARY) for d in dates], dtype=np.int8)


def run_corrector_ablation(
    forecaster: Forecaster,
    series: np.ndarray,
    timestamps: np.ndarray,
    temperature: np.ndarray | None,
    *,
    series_id: str,
    timezone: str,
    horizon: int = 24,
    initial_train_size: int = 672,
    season_length: int = 168,
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    min_train_origins: int = DEFAULT_MIN_TRAIN_ORIGINS,
    specs: Sequence[FeatureSpec] = ALL_SPECS,
    cached: list[CachedForecast] | None = None,
) -> AblationOutput:
    """Score the base model and every corrected arm over one series.

    Returns the per-arm scores alongside the forecasts they were computed from. The
    forecasts are kept because a summary metric cannot answer where a model fails, only
    whether it does, and the interval breaches that the explanation layer works from live
    in the individual hours rather than in the aggregate.
    """
    if temperature is None and any(spec.use_weather for spec in specs):
        raise ValueError("Weather arms were requested without a temperature series")

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
    is_holiday = holiday_flags(timestamps, timezone)
    holiday_class = holiday_class_flags(timestamps, timezone)
    median_index = _median_index(quantile_levels)

    # Feature rows and residuals for every origin, built once and sliced per fit. The
    # corrector never sees more of this than the origins before the one it is forecasting.
    per_origin_features = {
        spec.name: [
            _features_for(spec, origin, horizon, local_hours, is_weekend, temperature, series)
            for origin in origins
        ]
        for spec in specs
    }
    per_origin_residuals = [
        series[c.origin : c.origin + horizon] - c.quantiles[:, median_index] for c in cached
    ]
    # Stacked once and sliced per fit, so the scale corrector sees exactly the origins
    # before the one it is forecasting and nothing later.
    stacked_base = np.concatenate([c.quantiles for c in cached])
    stacked_actual = np.concatenate([series[c.origin : c.origin + horizon] for c in cached])
    stacked_holiday = np.concatenate([is_holiday[c.origin : c.origin + horizon] for c in cached])
    stacked_class = np.concatenate([holiday_class[c.origin : c.origin + horizon] for c in cached])
    stacked_hour = np.concatenate([local_hours[c.origin : c.origin + horizon] for c in cached])

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

    arm_names = [
        forecaster.name,
        *(f"{forecaster.name}+{spec.name}" for spec in specs),
        f"{forecaster.name}+{SCALE_ARM}",
        f"{forecaster.name}+{HOLIDAY_ARM}",
        f"{forecaster.name}+{HOLIDAY_CLASS_ARM}",
        f"{forecaster.name}+{HOLIDAY_HOUR_ARM}",
    ]
    results = {name: BacktestResult(name, series_id, horizon, 0) for name in arm_names}
    forecasts: dict[str, list[np.ndarray]] = {name: [] for name in arm_names}

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
        forecasts[forecaster.name].append(base)

        train_residuals = np.concatenate(per_origin_residuals[:position])
        for spec in specs:
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
            forecasts[f"{forecaster.name}+{spec.name}"].append(corrected)

        scaler = QuantileScaleCorrector(quantile_levels).fit(
            stacked_base[: position * horizon], stacked_actual[: position * horizon]
        )
        scaled = scaler.predict(base)
        _score(
            results[f"{forecaster.name}+{SCALE_ARM}"],
            origin,
            actual,
            scaled,
            history,
            season_length,
            quantile_levels,
            median_index,
        )
        forecasts[f"{forecaster.name}+{SCALE_ARM}"].append(scaled)

        window_holiday = is_holiday[origin : origin + horizon]
        holiday_scaler = HolidayScaleCorrector(quantile_levels).fit(
            stacked_base[: position * horizon],
            stacked_actual[: position * horizon],
            stacked_holiday[: position * horizon],
        )
        adjusted = holiday_scaler.predict(base, window_holiday)
        _score(
            results[f"{forecaster.name}+{HOLIDAY_ARM}"],
            origin,
            actual,
            adjusted,
            history,
            season_length,
            quantile_levels,
            median_index,
        )
        forecasts[f"{forecaster.name}+{HOLIDAY_ARM}"].append(adjusted)

        class_scaler = HolidayClassScaleCorrector(quantile_levels).fit(
            stacked_base[: position * horizon],
            stacked_actual[: position * horizon],
            stacked_class[: position * horizon],
        )
        by_class = class_scaler.predict(base, holiday_class[origin : origin + horizon])
        _score(
            results[f"{forecaster.name}+{HOLIDAY_CLASS_ARM}"],
            origin,
            actual,
            by_class,
            history,
            season_length,
            quantile_levels,
            median_index,
        )
        forecasts[f"{forecaster.name}+{HOLIDAY_CLASS_ARM}"].append(by_class)

        hour_scaler = HolidayHourScaleCorrector(quantile_levels).fit(
            stacked_base[: position * horizon],
            stacked_actual[: position * horizon],
            stacked_class[: position * horizon],
            stacked_hour[: position * horizon],
        )
        by_hour = hour_scaler.predict(
            base,
            holiday_class[origin : origin + horizon],
            local_hours[origin : origin + horizon],
        )
        _score(
            results[f"{forecaster.name}+{HOLIDAY_HOUR_ARM}"],
            origin,
            actual,
            by_hour,
            history,
            season_length,
            quantile_levels,
            median_index,
        )
        forecasts[f"{forecaster.name}+{HOLIDAY_HOUR_ARM}"].append(by_hour)

    return AblationOutput(
        results=results,
        scored_origins=[origins[position] for position in scored],
        forecasts=forecasts,
    )


def realised_volatility(series: np.ndarray, origin: int, lookback: int) -> float:
    """Standard deviation of hourly changes over the `lookback` hours before `origin`.

    Differences rather than levels: the level of demand says how big the market is, and
    the interval needs to know how much it moves. Strictly `[:origin]`, so this is
    information a forecaster standing at the origin genuinely has.
    """
    window = series[max(0, origin - lookback) : origin]
    if window.size < 2:
        return 0.0
    return float(np.std(np.diff(window)))


def _features_for(
    spec: FeatureSpec,
    origin: int,
    horizon: int,
    local_hours: np.ndarray,
    is_weekend: np.ndarray,
    temperature: np.ndarray,
    series: np.ndarray,
) -> np.ndarray:
    target = slice(origin, origin + horizon)
    lagged = slice(origin - LAG_HOURS, origin + horizon - LAG_HOURS)

    volatility_24 = volatility_168 = None
    if spec.use_volatility:
        # Constant across the window: one number describing the regime the forecast is
        # being made in, broadcast so it lines up with the per-hour columns.
        volatility_24 = np.full(horizon, realised_volatility(series, origin, 24))
        volatility_168 = np.full(horizon, realised_volatility(series, origin, 168))

    return build_features(
        spec,
        horizon_steps=np.arange(1, horizon + 1),
        local_hours=local_hours[target],
        is_weekend=is_weekend[target],
        temperature=temperature[target] if spec.use_weather else None,
        temperature_yesterday=temperature[lagged] if spec.use_weather else None,
        volatility_24=volatility_24,
        volatility_168=volatility_168,
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
