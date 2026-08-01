"""Breach detection and error decomposition.

The episode chaining is where the bugs would be: it walks a timeline and has to break runs
on three separate conditions. Getting any of them wrong produces episodes that look
entirely plausible, which is why each condition has its own test.
"""

from __future__ import annotations

import numpy as np
import pytest

from bellwether.eval.breaches import (
    HourlyRecords,
    episode_summary,
    find_episodes,
    hourly_records,
    pool_records,
    profile_by_horizon_step,
    profile_by_local_hour,
    profile_by_month,
)
from bellwether.eval.metrics import DEFAULT_QUANTILES

START = np.datetime64("2025-06-01T00", "ns")


def _records(
    actual: list[float],
    lower: list[float] | None = None,
    upper: list[float] | None = None,
    start: np.datetime64 = START,
    gaps_after: set[int] | None = None,
) -> HourlyRecords:
    """Hand-built records so every expected breach is countable by eye."""
    n = len(actual)
    offsets, cursor = [], 0
    for i in range(n):
        offsets.append(cursor)
        cursor += 25 if gaps_after and i in gaps_after else 1

    timestamps = np.array([start + np.timedelta64(o, "h") for o in offsets])
    return HourlyRecords(
        timestamps=timestamps,
        # Derived from this run's own start, not the module default, so a staggered run
        # genuinely lands on different local hours than an unstaggered one.
        local_hours=np.array(
            [(start.astype("datetime64[h]").astype(int) + o) % 24 for o in offsets]
        ),
        months=np.full(n, 6),
        horizon_steps=np.array([(i % 24) + 1 for i in range(n)]),
        actual=np.array(actual, dtype=float),
        median=np.full(n, 100.0),
        lower=np.array(lower if lower else [90.0] * n, dtype=float),
        upper=np.array(upper if upper else [110.0] * n, dtype=float),
    )


class TestRecordProperties:
    def test_breach_direction_is_split(self):
        records = _records([100.0, 120.0, 80.0])
        assert records.breached.tolist() == [False, True, True]
        assert records.breached_above.tolist() == [False, True, False]
        assert records.breached_below.tolist() == [False, False, True]

    def test_a_value_exactly_on_the_bound_is_covered(self):
        """Matches interval_coverage, which uses inclusive comparisons."""
        records = _records([110.0, 90.0])
        assert not records.breached.any()

    def test_exceedance_is_zero_inside_and_positive_outside(self):
        records = _records([100.0, 125.0, 85.0])
        assert records.exceedance.tolist() == [0.0, 15.0, 5.0]

    def test_exceedance_ratio_is_scale_free(self):
        """The same relative miss in a big and a small market scores the same."""
        big = _records([1200.0], lower=[900.0], upper=[1100.0])
        small = _records([120.0], lower=[90.0], upper=[110.0])
        assert big.exceedance_ratio[0] == pytest.approx(small.exceedance_ratio[0])

    def test_signed_error_points_the_right_way(self):
        records = _records([130.0])
        assert records.error[0] == pytest.approx(30.0), "under-forecast should be positive"


class TestEpisodes:
    def test_consecutive_breaches_form_one_episode(self):
        episodes = find_episodes(_records([100.0, 120.0, 121.0, 122.0, 100.0]))
        assert len(episodes) == 1
        assert episodes[0].duration_hours == 3
        assert episodes[0].direction == "above"

    def test_a_covered_hour_breaks_the_run(self):
        episodes = find_episodes(_records([120.0, 100.0, 120.0]))
        assert [e.duration_hours for e in episodes] == [1, 1]

    def test_a_direction_change_breaks_the_run(self):
        """Demand overshooting and demand collapsing are different events.

        Merged, they would produce one episode that no single explanation covers.
        """
        episodes = find_episodes(_records([120.0, 121.0, 80.0, 79.0]))
        assert len(episodes) == 2
        assert [e.direction for e in episodes] == ["above", "below"]
        assert [e.duration_hours for e in episodes] == [2, 2]

    def test_a_timeline_gap_breaks_the_run(self):
        """Windows with missing data are never scored, so rows can be days apart.

        Without this check, two unrelated breaches either side of a skipped window would
        fuse into one long fictional episode.
        """
        episodes = find_episodes(_records([120.0, 121.0], gaps_after={0}))
        assert len(episodes) == 2
        assert all(e.duration_hours == 1 for e in episodes)

    def test_an_episode_records_its_peak_not_its_first_hour(self):
        episodes = find_episodes(_records([115.0, 140.0, 112.0]))
        assert len(episodes) == 1
        assert episodes[0].peak_exceedance == pytest.approx(30.0)
        assert episodes[0].peak_at == episodes[0].start + np.timedelta64(1, "h")

    def test_total_exceedance_accumulates_over_the_episode(self):
        episodes = find_episodes(_records([115.0, 120.0]))
        assert episodes[0].total_exceedance == pytest.approx(15.0)

    def test_a_trailing_breach_is_not_dropped(self):
        """The final run has no covered hour after it to close it."""
        episodes = find_episodes(_records([100.0, 120.0, 121.0]))
        assert len(episodes) == 1
        assert episodes[0].duration_hours == 2

    def test_a_leading_breach_is_not_dropped(self):
        episodes = find_episodes(_records([120.0, 100.0]))
        assert len(episodes) == 1
        assert episodes[0].start == START

    def test_short_episodes_can_be_filtered_out(self):
        episodes = find_episodes(
            _records([120.0, 100.0, 120.0, 121.0, 122.0]), min_duration_hours=3
        )
        assert [e.duration_hours for e in episodes] == [3]

    def test_no_breaches_gives_no_episodes(self):
        assert find_episodes(_records([100.0] * 10)) == []

    def test_empty_records_are_handled(self):
        empty = _records([])
        assert find_episodes(empty) == []


class TestEpisodeSummary:
    def test_counts_directions_and_durations(self):
        episodes = find_episodes(_records([120.0, 121.0, 100.0, 80.0]))
        summary = episode_summary(episodes, total_hours=4)

        assert summary["episodes"] == 2
        assert summary["above"] == 1
        assert summary["below"] == 1
        assert summary["breached_hours"] == 3
        assert summary["breached_fraction"] == pytest.approx(0.75)
        assert summary["max_duration_hours"] == 2

    def test_empty_summary_does_not_divide_by_zero(self):
        assert episode_summary([], total_hours=0)["episodes"] == 0


class TestProfiles:
    def test_hour_profile_covers_the_whole_day(self):
        profile = profile_by_local_hour(_records([100.0] * 48))
        assert len(profile) == 24
        assert {row["hour"] for row in profile} == set(range(24))

    def test_hour_profile_isolates_a_single_bad_hour(self):
        """The point of the decomposition: an aggregate would average this away."""
        actual = [100.0] * 48
        for i in range(len(actual)):
            if (START.astype("datetime64[h]").astype(int) + i) % 24 == 18:
                actual[i] = 200.0

        profile = {row["hour"]: row for row in profile_by_local_hour(_records(actual))}
        assert profile[18]["coverage"] == 0.0
        assert profile[18]["mae"] == pytest.approx(100.0)
        assert profile[17]["coverage"] == 1.0

    def test_bias_keeps_its_sign(self):
        """A model consistently late on a ramp is wrong in one direction."""
        profile = profile_by_local_hour(_records([130.0] * 24))
        assert all(row["bias"] == pytest.approx(30.0) for row in profile if row["hours"])

    def test_month_profile_reports_only_months_present(self):
        assert [row["month"] for row in profile_by_month(_records([100.0] * 5))] == [6]

    def test_horizon_profile_reports_each_step(self):
        profile = profile_by_horizon_step(_records([100.0] * 48))
        assert [row["step"] for row in profile] == list(range(1, 25))

    def test_empty_slice_reports_zero_hours_rather_than_dividing(self):
        records = _records([100.0] * 3)
        profile = {row["hour"]: row for row in profile_by_local_hour(records)}
        empty = [row for row in profile.values() if row["hours"] == 0]
        assert empty and "mae" not in empty[0]


class TestPooling:
    """Pooling staggered runs is what makes a diurnal profile mean anything.

    Within one run, origins advance by exactly the horizon, so each local hour is always
    forecast at the same lead time and the two variables cannot be told apart. The first
    version of this analysis reported a diurnal profile that was entirely a horizon-step
    profile, and it looked completely reasonable.
    """

    def test_pooling_crosses_hour_of_day_against_horizon_step(self):
        run_a = _records([100.0] * 24)
        run_b = _records([100.0] * 24, start=START + np.timedelta64(6, "h"))
        # Within either run, one local hour maps to exactly one horizon step.
        for run in (run_a, run_b):
            pairs = set(zip(run.local_hours.tolist(), run.horizon_steps.tolist(), strict=True))
            assert len(pairs) == len({h for h, _ in pairs})

        pooled = pool_records([run_a, run_b])
        steps_at_hour = {
            hour: set(pooled.horizon_steps[pooled.local_hours == hour].tolist())
            for hour in set(pooled.local_hours.tolist())
        }
        assert any(len(steps) > 1 for steps in steps_at_hour.values())

    def test_pooling_concatenates_every_field(self):
        pooled = pool_records([_records([100.0] * 5), _records([100.0] * 7)])
        assert len(pooled) == 12
        assert pooled.local_hours.size == 12
        assert pooled.horizon_steps.size == 12

    def test_pooling_repeats_hours_which_is_why_episodes_must_not_use_it(self):
        """A breach covered by two offsets appears twice, so counting it twice is wrong."""
        run = _records([120.0])
        pooled = pool_records([run, run])

        assert len(find_episodes(run)) == 1
        assert sum(e.duration_hours for e in find_episodes(pooled)) == 2

    def test_rejects_an_empty_pool(self):
        with pytest.raises(ValueError, match="at least one run"):
            pool_records([])


class TestHourlyRecords:
    def _forecasts(self, n_windows: int, horizon: int = 24) -> list[np.ndarray]:
        spread = np.linspace(-10.0, 10.0, len(DEFAULT_QUANTILES))
        return [100.0 + np.tile(spread, (horizon, 1)) for _ in range(n_windows)]

    def _timestamps(self, n: int) -> np.ndarray:
        return np.array([START + np.timedelta64(i, "h") for i in range(n)])

    def test_flattens_windows_into_hours(self):
        series = np.full(200, 100.0)
        records = hourly_records(series, self._timestamps(200), [24, 48], self._forecasts(2), "UTC")
        assert len(records) == 48

    def test_horizon_step_restarts_at_each_origin(self):
        series = np.full(200, 100.0)
        records = hourly_records(series, self._timestamps(200), [24, 48], self._forecasts(2), "UTC")
        assert records.horizon_steps[0] == 1
        assert records.horizon_steps[23] == 24
        assert records.horizon_steps[24] == 1

    def test_local_hours_follow_the_market_timezone(self):
        """A Texas evening peak must not land in a different hour by season."""
        series = np.full(200, 100.0)
        utc = hourly_records(series, self._timestamps(200), [24], self._forecasts(1), "UTC")
        central = hourly_records(
            series, self._timestamps(200), [24], self._forecasts(1), "America/Chicago"
        )
        assert utc.local_hours[0] != central.local_hours[0]

    def test_rejects_mismatched_origins_and_forecasts(self):
        series = np.full(200, 100.0)
        with pytest.raises(ValueError, match="origins and"):
            hourly_records(series, self._timestamps(200), [24, 48], self._forecasts(1), "UTC")

    def test_rejects_an_empty_origin_set(self):
        series = np.full(200, 100.0)
        with pytest.raises(ValueError, match="at least one"):
            hourly_records(series, self._timestamps(200), [], [], "UTC")
