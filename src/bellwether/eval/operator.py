"""Comparison against the balancing authority's own day-ahead forecast.

EIA publishes each authority's day-ahead demand forecast as series type `DF` on Form
EIA-930. That is a far more meaningful bar than seasonal-naive: it is produced by the
people who operate the grid, with weather forecasts, unit commitment schedules, and
knowledge of local events that none of our models see.

Making the comparison fair requires care about horizon. The operator forecasts an entire
operating day before that day begins, so its effective horizon runs roughly 24 to 47 hours.
A rolling backtest forecasting 1 to 24 hours ahead from an arbitrary origin would win on
horizon alone and prove nothing. So origins are aligned to local midnight, giving both
forecasters the same information cutoff: everything up to the end of the previous local
day, nothing from the target day itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bellwether.eval.metrics import mae, mase, rmse, smape

# Local time zone of each balancing authority, used to find local midnight. The operating
# day is a local concept; using UTC midnight would offset the comparison by hours.
BA_TIMEZONES = {
    "CISO": "America/Los_Angeles",
    "ERCO": "America/Chicago",
    "PACE": "America/Denver",
}


def local_midnight_origins(
    timestamps: np.ndarray,
    timezone: str,
    min_index: int,
    horizon: int,
) -> list[int]:
    """Indices where local time is midnight, leaving room for a full forecast window.

    Handles daylight saving transitions implicitly: on a spring-forward day the local
    midnight still exists, and the window simply covers a 23 or 25 hour local day while
    remaining exactly `horizon` UTC hours. Both forecasters are scored over the identical
    index range, so neither is advantaged by the shift.
    """
    local = pd.DatetimeIndex(timestamps).tz_localize("UTC").tz_convert(timezone)
    is_midnight = local.hour == 0

    usable = np.flatnonzero(is_midnight)
    return [int(i) for i in usable if i >= min_index and i + horizon <= timestamps.size]


def evaluate_stored_forecast(
    actual: np.ndarray,
    forecast: np.ndarray,
    origins: list[int],
    *,
    horizon: int,
    season_length: int,
) -> dict[str, float]:
    """Score a recorded external forecast over the given windows.

    Only point metrics are returned. `DF` is a single number per hour with no stated
    uncertainty, so quantile loss and interval coverage do not apply, and inventing a
    distribution for it would be scoring a fiction.

    Windows where either the actual or the published forecast has a gap are skipped, the
    same rule the model backtest uses.
    """
    per_window: list[dict[str, float]] = []
    skipped = 0

    for origin in origins:
        window = slice(origin, origin + horizon)
        actual_window = actual[window]
        forecast_window = forecast[window]
        history = actual[:origin]

        if not (np.all(np.isfinite(actual_window)) and np.all(np.isfinite(forecast_window))):
            skipped += 1
            continue

        per_window.append(
            {
                "mae": mae(actual_window, forecast_window),
                "rmse": rmse(actual_window, forecast_window),
                "smape": smape(actual_window, forecast_window),
                "mase": mase(actual_window, forecast_window, history, season_length),
            }
        )

    if not per_window:
        raise ValueError("No evaluable windows: every one contained a gap")

    summary = {
        metric: float(np.mean([w[metric] for w in per_window]))
        for metric in ("mae", "rmse", "smape", "mase")
    }
    summary["windows"] = float(len(per_window))
    summary["skipped_windows"] = float(skipped)
    return summary
