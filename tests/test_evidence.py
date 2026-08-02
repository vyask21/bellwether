"""Evidence gathering for breach episodes.

Two properties carry most of the weight. A data-quality finding must outrank everything,
because it changes the question from "what caused this" to "this did not happen". And
evidence that argues the wrong way must be reported as such rather than quietly dropped or,
worse, presented as support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bellwether.eval.breaches import BreachEpisode
from bellwether.explain.evidence import (
    BASELINE_DAYS,
    Evidence,
    find_data_spikes,
    gather_evidence,
    holidays_in_window,
)

START = np.datetime64("2025-01-15T00", "ns")


def _grid(hours: int, start: np.datetime64 = START) -> np.ndarray:
    return np.array([start + np.timedelta64(h, "h") for h in range(hours)])


def _episode(
    start_hour: int,
    duration: int = 4,
    direction: str = "above",
    grid_start: np.datetime64 = START,
) -> BreachEpisode:
    start = grid_start + np.timedelta64(start_hour, "h")
    end = start + np.timedelta64(duration - 1, "h")
    return BreachEpisode(
        start=start,
        end=end,
        duration_hours=duration,
        direction=direction,
        peak_at=start,
        peak_exceedance=1000.0,
        peak_exceedance_ratio=1.2,
        total_exceedance=4000.0,
        local_hour_start=12,
        month=1,
    )


class TestDataSpikes:
    def test_finds_a_single_hour_collapse_between_agreeing_neighbours(self):
        """The real case: CISO went 29,881 to 11,819 to 29,886 MW in consecutive hours."""
        series = np.full(200, 29_900.0)
        series[100] = 11_819.0

        assert find_data_spikes(series).tolist() == [100]

    def test_ignores_a_genuine_ramp(self):
        """Load climbing steadily is not an artifact, however steep."""
        series = np.linspace(10_000.0, 40_000.0, 200)
        assert find_data_spikes(series).size == 0

    def test_ignores_a_step_change_that_persists(self):
        """A sustained shift is grid behaviour; the neighbours must agree to flag one hour."""
        series = np.concatenate([np.full(100, 30_000.0), np.full(100, 15_000.0)])
        assert find_data_spikes(series).size == 0

    def test_a_short_series_is_handled(self):
        assert find_data_spikes(np.array([1.0, 2.0])).size == 0

    def test_nans_do_not_raise_or_register(self):
        series = np.full(50, 100.0)
        series[20] = np.nan
        assert 20 not in find_data_spikes(series).tolist()


class TestHolidayLookup:
    def test_finds_thanksgiving(self):
        found = holidays_in_window(pd.Timestamp("2024-11-27"), pd.Timestamp("2024-11-29"))
        assert pd.Timestamp("2024-11-28") in found

    def test_an_ordinary_week_has_none(self):
        found = holidays_in_window(pd.Timestamp("2025-03-10"), pd.Timestamp("2025-03-14"))
        assert found.empty


class TestGatherEvidence:
    def _inputs(self, hours: int = 24 * 40):
        timestamps = _grid(hours)
        series = np.full(hours, 30_000.0)
        temperature = np.full(hours, 15.0)
        return timestamps, series, temperature

    def test_a_cold_snap_is_reported_for_an_above_bound_episode(self):
        timestamps, series, temperature = self._inputs()
        start = 24 * 30
        temperature[start : start + 6] = -5.0  # 20 C below the trailing baseline

        found = gather_evidence(
            _episode(start, duration=6, direction="above"),
            timestamps,
            series,
            temperature,
            "America/Chicago",
        )
        temp = [e for e in found if e.kind == "temperature"]
        assert temp, "a 20 C anomaly should be reported"
        assert temp[0].facts["anomaly_c"] == pytest.approx(-20.0, abs=0.1)
        assert temp[0].facts["consistent_with_direction"] is True
        assert "heating load" in temp[0].summary

    def test_a_small_anomaly_is_not_reported(self):
        """Below the station-spread threshold it would not survive being questioned."""
        timestamps, series, temperature = self._inputs()
        start = 24 * 30
        temperature[start : start + 6] = 16.0

        found = gather_evidence(
            _episode(start, duration=6), timestamps, series, temperature, "America/Chicago"
        )
        assert not [e for e in found if e.kind == "temperature"]

    def test_evidence_arguing_the_wrong_way_is_flagged_not_hidden(self):
        """A heat wave does not explain demand falling below the band.

        Reporting it as support would be the explanation layer inventing a causal story.
        """
        timestamps, series, temperature = self._inputs()
        start = 24 * 30
        temperature[start : start + 6] = 40.0

        found = gather_evidence(
            _episode(start, duration=6, direction="below"),
            timestamps,
            series,
            temperature,
            "America/Chicago",
        )
        temp = [e for e in found if e.kind == "temperature"][0]
        assert temp.facts["consistent_with_direction"] is False
        assert "does not explain" in temp.summary
        assert temp.strength < 0.3

    def test_a_cool_spell_in_a_hot_market_explains_demand_falling(self):
        """The regime bug, found on PACE and fixed.

        Reading "colder" as "more demand" unconditionally is wrong in a summer-peaking
        market: a cool spell sheds air conditioning and demand drops. Two genuine PACE
        summer episodes were called inconsistent before degree-days replaced the sign test.
        """
        timestamps, series, temperature = self._inputs()
        temperature[:] = 27.0  # a hot baseline, well into cooling territory
        start = 24 * 30
        temperature[start : start + 6] = 17.5

        found = gather_evidence(
            _episode(start, duration=6, direction="below"),
            timestamps,
            series,
            temperature,
            "America/Denver",
        )
        temp = [e for e in found if e.kind == "temperature"][0]
        assert temp.facts["expected_direction"] == "below"
        assert temp.facts["consistent_with_direction"] is True
        assert "less cooling load" in temp.summary
        assert temp.strength > 0.5

    def test_a_mild_spell_in_a_cold_market_explains_demand_falling(self):
        """The mirror case: warming up in winter sheds heating load."""
        timestamps, series, temperature = self._inputs()
        temperature[:] = 0.0
        start = 24 * 30
        temperature[start : start + 6] = 12.0

        found = gather_evidence(
            _episode(start, duration=6, direction="below"),
            timestamps,
            series,
            temperature,
            "America/Chicago",
        )
        temp = [e for e in found if e.kind == "temperature"][0]
        assert temp.facts["expected_direction"] == "below"
        assert temp.facts["consistent_with_direction"] is True
        assert "less heating load" in temp.summary

    def test_a_holiday_is_reported_for_a_below_bound_episode(self):
        """Thanksgiving 2024 produced a 25 hour below-bound episode in CISO."""
        timestamps = _grid(24 * 40, np.datetime64("2024-11-10T00", "ns"))
        series = np.full(timestamps.size, 30_000.0)
        temperature = np.full(timestamps.size, 15.0)
        start = 24 * 18  # 2024-11-28 UTC

        found = gather_evidence(
            _episode(start, duration=12, direction="below", grid_start=timestamps[0]),
            timestamps,
            series,
            temperature,
            "America/Los_Angeles",
        )
        holiday = [e for e in found if e.kind == "holiday"]
        assert holiday, "Thanksgiving should be found"
        assert holiday[0].facts["consistent_with_direction"] is True
        assert "2024-11-28" in holiday[0].facts["holiday_dates"]

    def test_data_quality_outranks_every_other_explanation(self):
        """The finding that changes the question rather than answering it."""
        timestamps, series, temperature = self._inputs()
        start = 24 * 30
        temperature[start : start + 6] = -5.0  # a real cold snap too
        series[start + 1] = 11_000.0  # and a bad reading

        found = gather_evidence(
            _episode(start, duration=6, direction="above"),
            timestamps,
            series,
            temperature,
            "America/Chicago",
        )
        assert found[0].kind == "data_quality"
        assert found[0].is_disqualifying
        assert any(e.kind == "temperature" for e in found), "other evidence is still reported"

    def test_ordinary_conditions_produce_no_evidence(self):
        """Silence is a valid answer. An unexplained episode must not be given a story."""
        timestamps, series, temperature = self._inputs()
        found = gather_evidence(
            _episode(24 * 30, duration=6), timestamps, series, temperature, "America/Chicago"
        )
        assert found == []

    def test_an_episode_outside_the_series_is_rejected(self):
        timestamps, series, temperature = self._inputs()
        stray = _episode(0, grid_start=np.datetime64("2030-01-01T00", "ns"))

        with pytest.raises(ValueError, match="outside the series"):
            gather_evidence(stray, timestamps, series, temperature, "America/Chicago")

    def test_no_baseline_means_no_temperature_claim(self):
        """Too early in the series to say what normal looks like, so it says nothing."""
        hours = 24 * (BASELINE_DAYS - 2)
        timestamps = _grid(hours)
        series = np.full(hours, 30_000.0)
        temperature = np.full(hours, -20.0)

        found = gather_evidence(
            _episode(hours - 6, duration=4), timestamps, series, temperature, "America/Chicago"
        )
        assert not [e for e in found if e.kind == "temperature"]

    def test_missing_temperature_is_silent_rather_than_fatal(self):
        """The weather archive ends months before the demand data does."""
        timestamps, series, temperature = self._inputs()
        temperature[:] = np.nan

        found = gather_evidence(
            _episode(24 * 30, duration=6), timestamps, series, temperature, "America/Chicago"
        )
        assert not [e for e in found if e.kind == "temperature"]


class TestEvidenceRecord:
    def test_only_data_quality_disqualifies(self):
        assert Evidence(kind="data_quality", summary="x").is_disqualifying
        assert not Evidence(kind="temperature", summary="x").is_disqualifying
        assert not Evidence(kind="holiday", summary="x").is_disqualifying
