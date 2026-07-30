"""Rolling-origin backtesting.

The forecast origin advances through time and the model only ever sees data strictly
before it. This is the only evaluation protocol that reflects how the system runs in
production, and the only one that cannot leak the future into the training window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from bellwether.eval.metrics import (
    DEFAULT_QUANTILES,
    interval_coverage,
    mae,
    mase,
    rmse,
    smape,
    weighted_quantile_loss,
)
from bellwether.forecast.base import Forecaster

log = logging.getLogger(__name__)


@dataclass(slots=True)
class WindowResult:
    """Metrics for one forecast origin."""

    origin_index: int
    mae: float
    rmse: float
    smape: float
    mase: float
    wql: float
    coverage_80: float


@dataclass(slots=True)
class BacktestResult:
    """Aggregate result for one model on one series."""

    model_name: str
    series_id: str
    horizon: int
    n_windows: int
    windows: list[WindowResult] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Mean of each metric across windows, plus coverage error."""
        if not self.windows:
            return {}
        return {
            "mae": float(np.mean([w.mae for w in self.windows])),
            "rmse": float(np.mean([w.rmse for w in self.windows])),
            "smape": float(np.mean([w.smape for w in self.windows])),
            "mase": float(np.mean([w.mase for w in self.windows])),
            "wql": float(np.mean([w.wql for w in self.windows])),
            "coverage_80": float(np.mean([w.coverage_80 for w in self.windows])),
            # Signed gap from nominal 80%: negative means overconfident intervals.
            "coverage_80_error": float(np.mean([w.coverage_80 for w in self.windows]) - 0.80),
        }


def rolling_origin_backtest(
    forecaster: Forecaster,
    series: np.ndarray,
    *,
    series_id: str,
    horizon: int = 24,
    step: int = 24,
    initial_train_size: int | None = None,
    season_length: int = 168,
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    max_windows: int | None = None,
) -> BacktestResult:
    """Evaluate `forecaster` by advancing the origin through `series`.

    Args:
        series: full observed history, oldest first.
        horizon: points forecast at each origin.
        step: how far the origin advances between windows.
        initial_train_size: history before the first origin. Defaults to four seasons.
        season_length: seasonal period, used to scale MASE.
        max_windows: cap for quick runs during development.

    Windows containing non-finite actuals are skipped rather than imputed — a gap in EIA
    data is a real gap, and filling it would invent accuracy the model never earned.
    """
    series = np.asarray(series, dtype=float)
    train_size = initial_train_size if initial_train_size is not None else season_length * 4

    if series.size < train_size + horizon:
        raise ValueError(
            f"Series {series_id!r} has {series.size} points; need at least "
            f"{train_size + horizon} for one window"
        )

    result = BacktestResult(
        model_name=forecaster.name,
        series_id=series_id,
        horizon=horizon,
        n_windows=0,
    )

    origins = range(train_size, series.size - horizon + 1, step)
    skipped = 0

    for origin in origins:
        if max_windows is not None and result.n_windows >= max_windows:
            break

        history = series[:origin]
        actual = series[origin : origin + horizon]

        recent_history = history[-season_length * 2 :]
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(recent_history)):
            skipped += 1
            continue

        forecasts = forecaster.predict(history, horizon, quantile_levels)
        median = forecasts[:, _median_index(quantile_levels)]

        result.windows.append(
            WindowResult(
                origin_index=origin,
                mae=mae(actual, median),
                rmse=rmse(actual, median),
                smape=smape(actual, median),
                mase=mase(actual, median, history, season_length),
                wql=weighted_quantile_loss(actual, forecasts, quantile_levels),
                coverage_80=interval_coverage(actual, forecasts, quantile_levels, 0.1, 0.9),
            )
        )
        result.n_windows += 1

    if skipped:
        log.warning(
            "%s/%s: skipped %d window(s) containing gaps", forecaster.name, series_id, skipped
        )
    return result


def _median_index(quantile_levels: tuple[float, ...]) -> int:
    matches = [i for i, q in enumerate(quantile_levels) if np.isclose(q, 0.5)]
    if not matches:
        raise ValueError("Quantile levels must include the median (0.5) for point metrics")
    return matches[0]
