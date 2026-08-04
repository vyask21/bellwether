"""Degrading an hourly series to a coarser cadence, and interpolating it back.

This exists for one control. NDFD publishes CONUS temperature every three hours while
everything else in this project is hourly, so an arm built on forecast temperature differs
from the published weather arm in **two** ways at once: the forecast is wrong, and it is
coarse. An ablation that changes two things measures neither.

The control is the observed series put through the coarseness and none of the error. Then

* forecast against degraded-observation isolates **forecast error**;
* degraded-observation against the published hourly arm isolates **resolution loss**.

The degradation has to match how the forecast series is actually built, not merely
resemble it: same stamp positions, same interpolation, same refusal to extrapolate past
the last stamp. Anything else leaves a difference between the arms that is neither of the
two things being measured. This is the same discipline as the calendar-only control, which
is what stopped a recalibration being read as a weather effect.
"""

from __future__ import annotations

import numpy as np

# NDFD's published cadence for CONUS temperature. Not a tuning knob: it is a property of
# the archive, checked in `docs/DATA_SOURCES.md` against both the 2.5 km and 5 km products.
FORECAST_CADENCE_HOURS = 3


def degrade_to_cadence(
    series: np.ndarray,
    timestamps: np.ndarray,
    cadence_hours: int = FORECAST_CADENCE_HOURS,
    origin_hour: int = 0,
) -> np.ndarray:
    """Keep only the hours a forecast would have published, then interpolate back.

    Stamps are anchored to `origin_hour` so they land on the same clock hours the forecast
    series does. Anchoring to the start of the array instead would put the two on different
    phases, and the arms would then differ by half a step of temperature as well as by
    everything under test.

    Hours outside the stamps that survive are interpolated linearly. Hours before the first
    stamp or after the last stay NaN: past the end of the stamps there is nothing to
    interpolate between, and continuing the last slope is inventing a reading.
    """
    if series.shape != timestamps.shape:
        raise ValueError(f"series {series.shape} and timestamps {timestamps.shape} differ")
    if cadence_hours < 1:
        raise ValueError(f"cadence_hours must be at least 1, got {cadence_hours}")

    hour_of_day = timestamps.astype("datetime64[h]").astype(int) % 24
    on_stamp = (hour_of_day - origin_hour) % cadence_hours == 0
    keep = on_stamp & np.isfinite(series)

    out = np.full(series.size, np.nan, dtype=float)
    if keep.sum() < 2:
        return out

    positions = np.flatnonzero(keep)
    inside = np.zeros(series.size, dtype=bool)
    inside[positions[0] : positions[-1] + 1] = True
    out[inside] = np.interp(
        np.flatnonzero(inside).astype(float), positions.astype(float), series[positions]
    )
    return out


def shared_coverage(*arrays: np.ndarray) -> np.ndarray:
    """A mask true where every arm has a value.

    Applied to all of them before scoring, so the arms are compared on one window set. Left
    to itself, each arm would drop a different set of origins, and a difference in the
    scores would then be partly a difference in which days each was asked about. That
    mistake has already cost this project a published result.
    """
    if not arrays:
        raise ValueError("Need at least one array")
    mask = np.isfinite(arrays[0])
    for array in arrays[1:]:
        mask &= np.isfinite(array)
    return mask
