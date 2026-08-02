"""Candidate explanations for a breach episode, computed from stored data.

Every number a brief can cite is produced here, in Python, from the database. The language
model receives these findings and narrates them. It never computes, estimates, or infers a
quantity, which is what makes each numeric claim in a brief traceable to a specific
measurement.

The three evidence kinds were chosen by looking at what the detector actually surfaced
rather than by guessing in advance, and the exercise reordered the roadmap. Of the five
worst episodes per market:

* **Holidays** explain several of the largest below-bound episodes. Thanksgiving 2024 in
  CISO ran 25 hours below the interval, Christmas Eve into Christmas Day 22 hours, Memorial
  Day 2025 in ERCO 12 hours. Chronos-Bolt sees only demand history and has no calendar, so
  a holiday is precisely the thing it cannot anticipate.
* **Temperature anomalies** explain the above-bound winter episodes, which is what the
  original roadmap expected of the evidence layer and is only part of the story.
* **Data quality** is the finding nobody planned for. The single most severe episode in the
  whole analysis, 6.31 interval widths, is an EIA value of 11,819 MW sitting between two
  hours near 29,900. A grid does not shed and recover 60% of its load in two hours. An
  explanation layer without this screen would have written a confident brief about it.

The two sources originally planned for this layer, nuclear outages and energy disruptions,
are not used. Disruptions is not an API route at all, and a reactor going offline does not
move demand, which is the series being forecast. See docs/DATA_SOURCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from bellwether.eval.breaches import BreachEpisode
from bellwether.forecast.residual import DEGREE_DAY_BASE_C

# How far back to look for "normal" when judging whether a temperature is unusual. Two
# weeks tracks the season as it moves, which a fixed monthly climatology would not, and is
# short enough that "unusually cold" means what an operator would mean by it. A longer
# baseline is not available anyway: the weather overlap is about thirteen months, so there
# is no second winter to average against.
BASELINE_DAYS = 14

# Degrees Celsius away from the trailing baseline before a temperature is worth citing.
# Below this the anomaly is smaller than the spread between the stations feeding the market
# average and would not survive being questioned.
TEMPERATURE_ANOMALY_THRESHOLD_C = 3.0

# A single hour this far from the mean of its two neighbours, when those neighbours agree
# with each other, is a reporting artifact rather than grid behaviour. Load does not move
# this fast: the three instances found across 52,562 hours are drops of 34%, 60% and 80%
# with full recovery the following hour.
SPIKE_DEVIATION = 0.30
SPIKE_NEIGHBOUR_AGREEMENT = 0.10

EvidenceKind = Literal["holiday", "temperature", "data_quality"]


@dataclass(frozen=True, slots=True)
class Evidence:
    """One candidate explanation, with the measurement behind it.

    `summary` is a sentence a brief may quote. `facts` holds the numbers it rests on, so a
    generated claim can be checked against the value it came from rather than trusted.
    """

    kind: EvidenceKind
    summary: str
    facts: dict = field(default_factory=dict)
    # Higher sorts first. Not a probability, and deliberately not presented as one: it
    # orders candidates for a human or a model to read, and nothing here estimates how
    # likely an explanation is to be correct.
    strength: float = 0.0

    @property
    def is_disqualifying(self) -> bool:
        """Whether this finding means the episode should not be explained at all."""
        return self.kind == "data_quality"


def find_data_spikes(series: np.ndarray) -> np.ndarray:
    """Indices where a single hour contradicts both its neighbours.

    Returns positions in `series`. Deliberately conservative: it fires only when the two
    neighbouring hours agree with each other and the hour between them does not, which is
    the signature of a bad reading rather than a fast ramp.

    Note what this does **not** do. EIA's Terms of Service forbid modifying its content and
    still calling it EIA data, so nothing here rewrites or interpolates the stored value.
    The flag is derived, lives outside the observations table, and its only effect is to
    stop a brief being written about a number that is not real.
    """
    if series.size < 3:
        return np.array([], dtype=int)

    previous, following = series[:-2], series[2:]
    middle = series[1:-1]
    neighbour_mean = (previous + following) / 2.0

    with np.errstate(invalid="ignore", divide="ignore"):
        deviation = np.abs(middle - neighbour_mean) / np.abs(neighbour_mean)
        agreement = np.abs(previous - following) / np.abs(neighbour_mean)

    suspect = (deviation > SPIKE_DEVIATION) & (agreement < SPIKE_NEIGHBOUR_AGREEMENT)
    return np.flatnonzero(np.nan_to_num(suspect, nan=0.0).astype(bool)) + 1


def holidays_in_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """US federal holidays overlapping a local-time window.

    Federal holidays only, which under-counts: the Friday after Thanksgiving and Christmas
    Eve both depress load heavily and neither is federal. An episode starting on Christmas
    Eve is still caught here, because the window it spans reaches into Christmas Day.
    """
    calendar = USFederalHolidayCalendar()
    # Pad so a holiday adjacent to the window is still visible to the caller.
    return calendar.holidays(
        start=start.normalize() - pd.Timedelta(days=1), end=end.normalize() + pd.Timedelta(days=1)
    )


def gather_evidence(
    episode: BreachEpisode,
    timestamps: np.ndarray,
    series: np.ndarray,
    temperature: np.ndarray,
    timezone: str,
    spike_indices: np.ndarray | None = None,
) -> list[Evidence]:
    """Assemble candidate explanations for one episode, strongest first.

    A data-quality finding always sorts first when present, because it changes the question
    from "what caused this" to "this did not happen".
    """
    index = pd.DatetimeIndex(timestamps)
    start = pd.Timestamp(episode.start)
    end = pd.Timestamp(episode.end)
    window = np.flatnonzero((index >= start) & (index <= end))
    if window.size == 0:
        raise ValueError(f"Episode {episode.start} to {episode.end} is outside the series")

    found: list[Evidence] = []
    found.extend(_data_quality_evidence(episode, index, series, window, spike_indices))
    found.extend(_holiday_evidence(episode, index, window, timezone))
    found.extend(_temperature_evidence(episode, index, temperature, window))

    return sorted(found, key=lambda e: (e.is_disqualifying, e.strength), reverse=True)


def _data_quality_evidence(
    episode: BreachEpisode,
    index: pd.DatetimeIndex,
    series: np.ndarray,
    window: np.ndarray,
    spike_indices: np.ndarray | None,
) -> list[Evidence]:
    spikes = find_data_spikes(series) if spike_indices is None else spike_indices
    hit = np.intersect1d(spikes, window)
    if hit.size == 0:
        return []

    position = int(hit[0])
    neighbours = (series[position - 1] + series[position + 1]) / 2.0
    return [
        Evidence(
            kind="data_quality",
            summary=(
                f"The reported demand at {index[position]:%Y-%m-%d %H:%M} UTC is "
                f"{series[position]:,.0f} MW, against about {neighbours:,.0f} MW in the "
                "hours either side. A single-hour move of this size with immediate "
                "recovery is a reporting artifact, not grid behaviour, so this episode "
                "should not be explained as an event."
            ),
            facts={
                "period": str(index[position]),
                "reported_mw": float(series[position]),
                "neighbour_mean_mw": float(neighbours),
                "deviation_fraction": float(abs(series[position] - neighbours) / abs(neighbours)),
                "affected_hours": int(hit.size),
            },
            strength=1.0,
        )
    ]


def _holiday_evidence(
    episode: BreachEpisode,
    index: pd.DatetimeIndex,
    window: np.ndarray,
    timezone: str,
) -> list[Evidence]:
    local = index[window].tz_localize("UTC").tz_convert(timezone)
    holidays = holidays_in_window(local[0].tz_localize(None), local[-1].tz_localize(None))
    if holidays.empty:
        return []

    local_dates = {stamp.date() for stamp in local}
    overlapping = [h for h in holidays if h.date() in local_dates]
    if not overlapping:
        return []

    names = ", ".join(f"{h:%Y-%m-%d}" for h in overlapping)
    # A holiday depresses load, so it explains demand falling below the band, not above it.
    # Reporting it against an above-bound episode would be citing a fact that argues the
    # wrong way.
    consistent = episode.direction == "below"
    return [
        Evidence(
            kind="holiday",
            summary=(
                f"The episode covers a US federal holiday ({names} local). Holidays depress "
                "commercial and industrial load, and the forecaster sees only demand "
                "history with no calendar, so it cannot anticipate one."
                + (
                    ""
                    if consistent
                    else (
                        " Note the episode runs above the interval, which a holiday"
                        " does not explain."
                    )
                )
            ),
            facts={
                "holiday_dates": [str(h.date()) for h in overlapping],
                "episode_direction": episode.direction,
                "consistent_with_direction": consistent,
            },
            strength=0.9 if consistent else 0.2,
        )
    ]


def _temperature_evidence(
    episode: BreachEpisode,
    index: pd.DatetimeIndex,
    temperature: np.ndarray,
    window: np.ndarray,
) -> list[Evidence]:
    during = temperature[window]
    if not np.any(np.isfinite(during)):
        return []

    baseline_start = window[0] - BASELINE_DAYS * 24
    if baseline_start < 0:
        return []
    baseline = temperature[baseline_start : window[0]]
    if not np.any(np.isfinite(baseline)):
        return []

    observed = float(np.nanmean(during))
    normal = float(np.nanmean(baseline))
    anomaly = observed - normal
    if abs(anomaly) < TEMPERATURE_ANOMALY_THRESHOLD_C:
        return []

    # Which way the anomaly pushes demand depends on the regime, not on its sign. A
    # summer market cooling off sheds air conditioning and demand falls; a winter market
    # warming up sheds heating and demand also falls. Reading "colder" as "more demand"
    # unconditionally called two genuine PACE summer episodes inconsistent when a mild
    # spell was exactly the explanation.
    #
    # Degree-days handle every regime, including the mild band where neither applies:
    # demand follows the change in total heating plus cooling load, whichever the weather
    # moved.
    load_change = _degree_days(during) - _degree_days(baseline)
    expected = "above" if load_change > 0 else "below"
    consistent = episode.direction == expected

    if load_change > 0:
        mechanism = (
            "Unusual heat raises cooling load."
            if anomaly > 0
            else "Unusual cold raises heating load."
        )
    else:
        mechanism = (
            "Milder than usual, so less heating load."
            if anomaly > 0
            else "Cooler than usual, so less cooling load."
        )

    return [
        Evidence(
            kind="temperature",
            summary=(
                f"Population-weighted temperature averaged {observed:.1f} C during the "
                f"episode against {normal:.1f} C over the preceding {BASELINE_DAYS} days, "
                f"an anomaly of {anomaly:+.1f} C. "
                + mechanism
                + (
                    ""
                    if consistent
                    else (
                        f" Note this points to demand {expected} the interval and the"
                        f" episode ran {episode.direction} it, so the anomaly does not"
                        " explain this episode."
                    )
                )
            ),
            facts={
                "episode_mean_c": observed,
                "baseline_mean_c": normal,
                "anomaly_c": anomaly,
                "degree_day_change": load_change,
                "baseline_days": BASELINE_DAYS,
                "episode_direction": episode.direction,
                "expected_direction": expected,
                "consistent_with_direction": consistent,
            },
            strength=min(0.85, 0.3 + abs(anomaly) / 20.0) if consistent else 0.2,
        )
    ]


def _degree_days(temperature: np.ndarray) -> float:
    """Mean heating plus cooling degrees, the load a temperature series implies.

    One number standing for "how much work the weather is asking the grid to do", so that
    the direction of a demand response follows from comparing two of them rather than from
    the sign of a temperature change.
    """
    cooling = np.maximum(temperature - DEGREE_DAY_BASE_C, 0.0)
    heating = np.maximum(DEGREE_DAY_BASE_C - temperature, 0.0)
    return float(np.nanmean(cooling + heating))
