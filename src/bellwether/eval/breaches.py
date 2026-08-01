"""Where a forecast fails, rather than how much.

A summary metric answers whether a model is good. It cannot answer the question an
operator actually has, which is when to stop trusting it. This module takes the hour-level
forecasts a backtest produces and asks three things of them:

1. **Where does the error live?** Error decomposed by local hour of day. Aggregate metrics
   average the evening ramp together with 4am, and the ramp is the whole problem.
2. **Where does the miscoverage live?** Coverage decomposed the same way. A model can hit
   80% overall while being badly wrong in a few hours, and that is a different defect from
   being uniformly slightly overconfident.
3. **What does a breach look like?** Consecutive breached hours chained into episodes, with
   a direction, a duration, and a severity. This is the unit the explanation layer works
   from: a brief explains an episode, not an hour.

Scored windows tile the timeline without overlapping, since origins advance by exactly the
horizon, so each target hour appears once and consecutive breached hours are genuinely
consecutive in time rather than an artifact of overlapping windows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bellwether.eval.metrics import DEFAULT_QUANTILES, _quantile_index

log = logging.getLogger(__name__)

# The interval whose breaches are counted. Matches the coverage metric reported everywhere
# else, so a breach here is exactly a miss of the 80% band reported there.
DEFAULT_LOWER = 0.1
DEFAULT_UPPER = 0.9


@dataclass(slots=True)
class HourlyRecords:
    """One row per forecast hour, as parallel arrays.

    Struct-of-arrays rather than a list of objects: every consumer here filters and
    aggregates over tens of thousands of hours, which is a vector operation.
    """

    timestamps: np.ndarray  # datetime64, UTC
    local_hours: np.ndarray  # int, 0..23 in market-local time
    months: np.ndarray  # int, 1..12 in market-local time
    horizon_steps: np.ndarray  # int, 1..horizon
    actual: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __len__(self) -> int:
        return int(self.actual.size)

    @property
    def error(self) -> np.ndarray:
        """Signed error. Positive means the forecast came in under the actual."""
        return self.actual - self.median

    @property
    def breached_above(self) -> np.ndarray:
        return self.actual > self.upper

    @property
    def breached_below(self) -> np.ndarray:
        return self.actual < self.lower

    @property
    def breached(self) -> np.ndarray:
        return self.breached_above | self.breached_below

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def exceedance(self) -> np.ndarray:
        """How far outside the interval, in the units of the series. Zero if inside."""
        return np.maximum.reduce(
            [self.actual - self.upper, self.lower - self.actual, np.zeros_like(self.actual)]
        )

    @property
    def exceedance_ratio(self) -> np.ndarray:
        """Exceedance as a multiple of interval width.

        Scale-free, so a breach in a 30 GW market and a 3 GW market are comparable, and so
        severity means "how badly the model was wrong about its own uncertainty" rather
        than "how big the market is".
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.width > 0, self.exceedance / self.width, np.nan)


def hourly_records(
    series: np.ndarray,
    timestamps: np.ndarray,
    origins: Sequence[int],
    forecasts: Sequence[np.ndarray],
    timezone: str,
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
) -> HourlyRecords:
    """Flatten per-window forecasts into one row per forecast hour."""
    if len(origins) != len(forecasts):
        raise ValueError(f"Got {len(origins)} origins and {len(forecasts)} forecasts")
    if not origins:
        raise ValueError("Need at least one scored origin")

    lo_idx = _quantile_index(quantile_levels, lower)
    hi_idx = _quantile_index(quantile_levels, upper)
    median_idx = _quantile_index(quantile_levels, 0.5)

    indices, medians, lowers, uppers, steps = [], [], [], [], []
    for origin, window in zip(origins, forecasts, strict=True):
        horizon = window.shape[0]
        indices.append(np.arange(origin, origin + horizon))
        medians.append(window[:, median_idx])
        lowers.append(window[:, lo_idx])
        uppers.append(window[:, hi_idx])
        steps.append(np.arange(1, horizon + 1))

    index = np.concatenate(indices)
    local = pd.DatetimeIndex(timestamps[index]).tz_localize("UTC").tz_convert(timezone)

    return HourlyRecords(
        timestamps=timestamps[index],
        local_hours=local.hour.to_numpy(),
        months=local.month.to_numpy(),
        horizon_steps=np.concatenate(steps),
        actual=series[index],
        median=np.concatenate(medians),
        lower=np.concatenate(lowers),
        upper=np.concatenate(uppers),
    )


RECORD_FIELDS = (
    "timestamps",
    "local_hours",
    "months",
    "horizon_steps",
    "actual",
    "median",
    "lower",
    "upper",
)


def pool_records(runs: Sequence[HourlyRecords]) -> HourlyRecords:
    """Concatenate several backtest runs into one record set, for profiles only.

    Origins advance by exactly the horizon, so within a single run every local hour is
    always forecast at the same lead time. Hour of day and horizon step are then the same
    variable under two names, and no amount of slicing one run can separate them. Pooling
    runs whose origins are offset by a fraction of a day crosses the two properly.

    The cost is that an hour appears once per run. That repetition is what makes the
    crossing work and what makes this unusable for `find_episodes`, which counts breaches
    and would count each of them several times.
    """
    if not runs:
        raise ValueError("Need at least one run to pool")
    return HourlyRecords(
        **{field: np.concatenate([getattr(run, field) for run in runs]) for field in RECORD_FIELDS}
    )


def profile_by_local_hour(records: HourlyRecords) -> list[dict]:
    """Error, coverage, and sharpness for each hour of the local day."""
    return [
        _slice_summary(records, records.local_hours == hour, {"hour": hour}) for hour in range(24)
    ]


def profile_by_month(records: HourlyRecords) -> list[dict]:
    """The same decomposition by calendar month, to separate season from time of day."""
    present = sorted(set(int(m) for m in records.months))
    return [_slice_summary(records, records.months == month, {"month": month}) for month in present]


def profile_by_horizon_step(records: HourlyRecords) -> list[dict]:
    """The same decomposition by how far ahead the hour was forecast."""
    present = sorted(set(int(s) for s in records.horizon_steps))
    return [
        _slice_summary(records, records.horizon_steps == step, {"step": step}) for step in present
    ]


def _slice_summary(records: HourlyRecords, mask: np.ndarray, label: dict) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {**label, "hours": 0}

    error = records.error[mask]
    return {
        **label,
        "hours": count,
        "mae": float(np.mean(np.abs(error))),
        # Signed, because a model that is late on the evening ramp is wrong in a
        # consistent direction and the absolute error hides that entirely.
        "bias": float(np.mean(error)),
        "coverage": float(np.mean(~records.breached[mask])),
        "breach_rate_above": float(np.mean(records.breached_above[mask])),
        "breach_rate_below": float(np.mean(records.breached_below[mask])),
        "mean_width": float(np.mean(records.width[mask])),
    }


@dataclass(slots=True)
class BreachEpisode:
    """A run of consecutive hours outside the interval, in one direction."""

    start: np.datetime64
    end: np.datetime64
    duration_hours: int
    direction: str  # "above" when demand exceeded the upper bound, else "below"
    peak_at: np.datetime64
    peak_exceedance: float
    peak_exceedance_ratio: float
    total_exceedance: float
    local_hour_start: int
    month: int

    def as_dict(self) -> dict:
        return {
            "start": str(self.start),
            "end": str(self.end),
            "duration_hours": self.duration_hours,
            "direction": self.direction,
            "peak_at": str(self.peak_at),
            "peak_exceedance": round(self.peak_exceedance, 1),
            "peak_exceedance_ratio": round(self.peak_exceedance_ratio, 3),
            "total_exceedance": round(self.total_exceedance, 1),
            "local_hour_start": self.local_hour_start,
            "month": self.month,
        }


def find_episodes(records: HourlyRecords, min_duration_hours: int = 1) -> list[BreachEpisode]:
    """Chain consecutive breached hours into episodes.

    A run is broken by a covered hour, by a change of direction, or by a gap in the
    timeline. The direction split matters: demand running above the upper bound and demand
    collapsing below the lower bound are different events with different causes, and
    merging them would produce an episode that no single explanation covers.

    The timeline gap check is what makes this safe across skipped windows. Windows
    containing a data gap are never scored, so consecutive rows can be days apart, and
    without the check two unrelated breaches either side of a gap would fuse into one long
    fictional episode.
    """
    if len(records) == 0:
        return []

    order = np.argsort(records.timestamps, kind="stable")
    timestamps = records.timestamps[order]
    breached = records.breached[order]
    above = records.breached_above[order]
    exceedance = records.exceedance[order]
    ratio = records.exceedance_ratio[order]
    local_hours = records.local_hours[order]
    months = records.months[order]

    one_hour = np.timedelta64(1, "h")
    episodes: list[BreachEpisode] = []
    start: int | None = None

    for i in range(len(timestamps)):
        contiguous = (
            start is not None
            and breached[i]
            and above[i] == above[start]
            and timestamps[i] - timestamps[i - 1] == one_hour
        )
        if contiguous:
            continue

        if start is not None:
            episodes.append(
                _build_episode(start, i, timestamps, above, exceedance, ratio, local_hours, months)
            )
            start = None
        if breached[i]:
            start = i

    if start is not None:
        episodes.append(
            _build_episode(
                start, len(timestamps), timestamps, above, exceedance, ratio, local_hours, months
            )
        )

    return [e for e in episodes if e.duration_hours >= min_duration_hours]


def _build_episode(
    start: int,
    stop: int,
    timestamps: np.ndarray,
    above: np.ndarray,
    exceedance: np.ndarray,
    ratio: np.ndarray,
    local_hours: np.ndarray,
    months: np.ndarray,
) -> BreachEpisode:
    span = slice(start, stop)
    peak = start + int(np.argmax(exceedance[span]))
    return BreachEpisode(
        start=timestamps[start],
        end=timestamps[stop - 1],
        duration_hours=stop - start,
        direction="above" if above[start] else "below",
        peak_at=timestamps[peak],
        peak_exceedance=float(exceedance[peak]),
        peak_exceedance_ratio=float(ratio[peak]),
        total_exceedance=float(np.sum(exceedance[span])),
        local_hour_start=int(local_hours[start]),
        month=int(months[start]),
    )


def episode_summary(episodes: Sequence[BreachEpisode], total_hours: int) -> dict:
    """Headline numbers for a set of episodes."""
    if not episodes:
        return {"episodes": 0, "breached_hours": 0, "breached_fraction": 0.0}

    durations = np.array([e.duration_hours for e in episodes])
    ratios = np.array([e.peak_exceedance_ratio for e in episodes])
    breached_hours = int(durations.sum())

    return {
        "episodes": len(episodes),
        "breached_hours": breached_hours,
        "breached_fraction": breached_hours / total_hours if total_hours else 0.0,
        "above": sum(1 for e in episodes if e.direction == "above"),
        "below": sum(1 for e in episodes if e.direction == "below"),
        "mean_duration_hours": float(durations.mean()),
        "max_duration_hours": int(durations.max()),
        # The tail is the point: a long episode is a different operational problem from a
        # single stray hour, and the mean hides how often the long ones happen.
        "episodes_over_3h": int(np.sum(durations > 3)),
        "mean_peak_ratio": float(np.nanmean(ratios)),
        "max_peak_ratio": float(np.nanmax(ratios)),
    }
