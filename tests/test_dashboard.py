"""The dashboard builds every chart from committed files.

Not a UI test. It checks the thing that actually breaks: a results file changes shape, or
an arm is renamed, and a chart silently draws nothing. Every builder here is called with
the real committed data and asserted to produce marks.

Skipped wholesale when the snapshot has not been exported, since a fresh clone has no
Parquet and failing there would only teach people to ignore the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"
sys.path.insert(0, str(DASHBOARD))

pytest.importorskip("altair")
pytest.importorskip("streamlit")

import loaders as data  # noqa: E402
import viz  # noqa: E402


def _has_snapshot() -> bool:
    """All three markets, not merely one.

    Cross-market charts normalise by each market's mean demand, which comes from the
    snapshot, so a partial export produces charts that are silently missing a series
    rather than obviously broken. That is the failure this predicate exists to catch.
    """
    exported = {
        p.stem.removeprefix("forecasts_")
        for p in (DASHBOARD.parent / "snapshot").glob("forecasts_*.parquet")
    }
    return set(data.MARKETS) <= exported


needs_snapshot = pytest.mark.skipif(not _has_snapshot(), reason="snapshot not fully exported")


class TestTheme:
    def test_it_enables_without_a_deprecation_error(self):
        viz.enable_theme()

    def test_the_categorical_range_is_the_validated_order(self):
        """Slot order is the colourblind-safety mechanism, not a preference."""
        assert viz.theme()["config"]["range"]["category"] == viz.SERIES
        assert viz.SERIES[:3] == ["#2a78d6", "#eb6834", "#1baf7a"]

    def test_every_market_has_a_fixed_hue(self):
        """Colour follows the entity, so filtering one market never repaints the others."""
        assert set(viz.MARKET_COLOURS) == set(data.MARKETS)
        assert len(set(viz.MARKET_COLOURS.values())) == len(data.MARKETS)


class TestResultFrames:
    @needs_snapshot
    def test_coverage_and_width_covers_all_three_markets(self):
        frame = data.coverage_width_frame()
        assert set(frame["market"]) == set(data.MARKETS), "weather_ablation lost a market"
        assert (frame["coverage"].between(50, 100)).all()
        assert (frame["width_pct"] > 0).all()

    @pytest.mark.parametrize(
        ("field", "axis", "buckets"),
        [("mae", "by_local_hour", 24), ("coverage", "by_month", 11)],
    )
    def test_profiles_have_one_row_per_bucket_per_market(self, field, axis, buckets):
        frame = data.profile_frame(field, axis)
        assert not frame.empty
        for market in data.MARKETS:
            assert len(frame[frame["market"] == market]) == buckets, market

    @needs_snapshot
    def test_every_checkpoint_is_present_with_a_width_in_every_market(self):
        """The width is the load-bearing column. The July entries had none, and a coverage
        comparison without one is the reading this whole section warns against."""
        frame = data.model_comparison_frame()
        assert len(frame) == len(data.MARKETS) * len(data.MODEL_ARMS)
        assert (frame["width_pct"] > 0).all()
        assert (frame["coverage"].between(50, 100)).all()

    @needs_snapshot
    def test_the_checkpoints_agree_on_which_market_is_hardest_to_calibrate(self):
        """Findings 21 and 23 together. If a re-measurement ever breaks the shared ordering,
        the prose claiming the defect belongs to the grid is no longer supported.

        Written over however many checkpoints `MODEL_ARMS` carries rather than over two,
        because the point of the claim is that it keeps surviving another one.
        """
        frame = data.model_comparison_frame()
        ordering = {
            arm: tuple(group.sort_values("coverage")["market"])
            for arm, group in frame.groupby("arm")
        }
        assert len(ordering) == len(data.MODEL_ARMS)
        assert len(set(ordering.values())) == 1, ordering
        assert next(iter(ordering.values()))[-1] == "CISO", "CISO is best calibrated for all"

    def test_the_small_checkpoint_keeps_most_of_the_gain_in_every_market(self):
        """Finding 23, and the numbers section 4 states in prose. A table that drifted out
        of this range would leave the page asserting figures the data no longer carries."""
        frame = data.retention_frame()
        assert len(frame) == len(data.MARKETS)
        kept = frame[["MASE gain kept (%)", "WQL gain kept (%)"]]
        assert kept.min().min() >= 85, "the page says 89 to 95%"
        assert kept.max().max() <= 100, "keeping more than all of it would be a new finding"
        assert (frame["80% width vs base (%)"] > 0).all(), "less sharp in every market"

    def test_the_longer_context_helps_in_every_market_by_the_amount_the_page_says(self):
        """Finding 24, and the three figures section 4 states in prose. The page names 1.1
        to 4.3%, so a re-measurement that moved the range would leave it asserting numbers
        the data no longer carries."""
        frame = data.context_frame()
        assert len(frame) == len(data.MARKETS)
        gain = frame["MASE better by (%)"]
        assert (gain > 0).all(), "the page says the extra history helps everywhere"
        assert gain.min() >= 1.0 and gain.max() <= 4.5, "the page says 1.1 to 4.3%"
        assert (frame["Coverage change (pts)"] > 0).all(), "coverage improves in all three"

    def test_only_erco_gets_its_coverage_without_paying_width(self):
        """The load-bearing claim of the passage, and the one most likely to be wrong after
        a re-run. If a second market ever narrows, the prose calling ERCO "the only place on
        this page" is false; if ERCO stops narrowing, the whole passage loses its point."""
        frame = data.context_frame().set_index("Market")
        narrowed = frame.index[frame["80% width change (%)"] < 0].tolist()
        assert narrowed == ["ERCO"], f"the page says ERCO alone narrows, got {narrowed}"
        assert frame.loc["ERCO", "MASE better by (%)"] == frame["MASE better by (%)"].max()

    def test_the_long_arm_does_not_overtake_either_chronos_checkpoint(self):
        """The conclusion the section closes on. Written against the raw record rather than
        the frame, because the frame reports the change and this is about the level."""
        backtest = data.results("backtest_results.json")
        for series_id, arms in backtest.items():
            long = arms["timesfm_2p5_200m_long"]["mase"]
            assert long > arms["chronos_bolt_base"]["mase"], series_id
            assert long > arms["chronos_bolt_small"]["mase"], series_id

    def test_the_forecast_ablation_has_all_three_arms_in_every_market(self):
        frame = data.forecast_frame()
        assert set(frame["Market"]) == set(data.MARKETS)
        for market in data.MARKETS:
            arms = frame[frame["Market"] == market]["Temperature"]
            assert list(arms) == list(data.FORECAST_ARMS.values()), market

    def test_the_control_is_identical_in_every_pass(self):
        """What makes the three arms comparable at all. Each pass re-scores the calendar-only
        control, and they agree only if all three saw the same windows. If this ever drifts,
        the percentages in the section are differences between different window sets."""
        for series_id, arms in data.results("forecast_ablation.json").items():
            controls = {
                arm: metrics["smape"] for arm, metrics in arms.items() if arm.endswith("+calendar")
            }
            assert len(controls) == 3, series_id
            assert len(set(controls.values())) == 1, (series_id, controls)

    def test_a_real_forecast_keeps_most_of_the_gain_on_the_market_that_has_one(self):
        """The finding the new half of section 6 is built on. ERCO is the only market with a
        weather effect large enough to lose anything measurable, so it is where the claim
        lives: four fifths of the ceiling, and resolution costing nothing."""
        frame = data.forecast_frame().set_index(["Market", "Temperature"])
        erco = frame.loc["ERCO"]["sMAPE change (%)"]
        observed, degraded, forecast = (erco[label] for label in data.FORECAST_ARMS.values())
        assert observed < forecast < 0, "the forecast arm has left the gap it was measured in"
        assert abs(forecast / observed) > 0.75, "less than three quarters of the gain survives"
        assert abs(degraded - observed) < 0.2, "the coarse cadence has started costing something"

    def test_the_hour_profile_covers_every_hour_in_every_market(self):
        frame = data.hour_profile_frame()
        for market in data.MARKETS:
            hours = sorted(frame[frame["market"] == market]["hour"])
            assert hours == list(range(24)), market

    def test_the_holiday_shape_is_not_flat(self):
        """The finding the chart is drawn to show. A scalar shift being the wrong shape is
        the claim, and a flat profile in any market would withdraw it there."""
        frame = data.hour_profile_frame()
        for market in data.MARKETS:
            offsets = frame[frame["market"] == market]["offset"]
            overnight = offsets[frame["hour"].isin(range(5))].mean()
            deepest = offsets.min()
            assert deepest < overnight * 2, market

    def test_holiday_offsets_carry_both_observance_classes(self):
        holiday = data.results("holiday_arm.json")
        for series_id, entry in holiday.items():
            learned = entry["learned_offsets"]
            assert {"widely observed", "federal only", "pooled"} <= set(learned), series_id

    def test_the_minor_offset_is_smaller_everywhere(self):
        """The finding the section is built on. If it inverts, the prose is now wrong."""
        holiday = data.results("holiday_arm.json")
        for series_id, entry in holiday.items():
            learned = entry["learned_offsets"]
            assert abs(learned["federal only"]) < abs(learned["widely observed"]), series_id


class TestChartsRender:
    def _marks(self, chart) -> str:
        return chart.to_json()

    @needs_snapshot
    def test_coverage_and_width_are_two_plots_not_one(self):
        """A dual axis here would invent a relationship between two unrelated scales."""
        chart = viz.coverage_and_width(data.coverage_width_frame())
        assert len(chart.vconcat) == 2

    def test_profile_renders(self):
        frame = data.profile_frame("mae", "by_local_hour")
        assert "mark" in self._marks(viz.profile(frame, "bucket", "Hour", "value", "MAE"))

    def test_learned_offsets_renders(self):
        holiday = data.results("holiday_arm.json")
        rows = [
            {"market": sid.split(":")[0], "observance": label, "offset": value}
            for sid, entry in holiday.items()
            for label, value in entry["learned_offsets"].items()
            if label != "pooled"
        ]
        assert "mark" in self._marks(viz.learned_offsets(pd.DataFrame(rows)))

    def test_paired_holidays_renders(self):
        entry = data.results("holiday_arm.json")["CISO:D"]
        rows = [
            {
                "date": r["date"],
                "name": r["date"],
                "observance": r["observance"],
                "change": r["change_vs_scale"],
                "direction": "better" if r["change_vs_scale"] < 0 else "worse",
            }
            for r in entry["per_holiday"]
        ]
        assert "mark" in self._marks(viz.paired_holidays(pd.DataFrame(rows)))


@needs_snapshot
class TestSnapshot:
    def test_observations_cover_the_full_demand_history(self):
        frame = data.observations("CISO")
        assert len(frame) > 17_000
        assert frame["demand_mw"].notna().all()

    def test_temperature_is_absent_for_about_half_the_grid(self):
        """NCEI's archive ends before EIA's. A chart must drop those hours, not fill them."""
        frame = data.observations("CISO")
        covered = frame["temperature_c"].notna().mean()
        assert 0.4 < covered < 0.7, covered

    def test_every_arm_is_present_in_the_snapshot(self):
        frame = pd.read_parquet(DASHBOARD.parent / "snapshot" / "forecasts_CISO.parquet")
        assert set(frame["arm"]) == {"base", "scale", "holiday", "holidayclass"}

    def test_the_band_is_ordered(self):
        """Every consumer of a quantile forecast assumes the levels are sorted."""
        frame = data.forecasts("CISO", "scale")
        assert (frame["q10"] <= frame["q50"]).all()
        assert (frame["q50"] <= frame["q90"]).all()

    def test_an_unknown_arm_is_rejected_rather_than_returning_nothing(self):
        """The bug this replaced: the loader filtered on full identifiers while the
        snapshot stored short labels, so every forecast chart drew an empty frame and
        nothing anywhere said so."""
        with pytest.raises(ValueError, match="Unknown snapshot arm"):
            data.forecasts("CISO", "chronos_bolt_base+scale")

    def test_a_forecast_window_renders(self):
        window = data.forecasts("CISO", "scale")
        origin = sorted(window["origin"].unique())[len(window["origin"].unique()) // 2]
        slice_ = window[window["origin"] == origin]
        assert len(slice_) == 24
        chart = viz.forecast_window(data.observations("CISO"), slice_, "test")
        assert "mark" in chart.to_json()

    def test_it_clips_the_observed_series_rather_than_inlining_two_years(self):
        """Altair embeds its data in the spec, so an unclipped frame ships 17,521 rows of
        JSON to draw 48 of them. Altair's own row cap turns that into an error rather than
        a slow page, which is how this was caught."""
        window = data.forecasts("CISO", "scale")
        slice_ = window[window["origin"] == sorted(window["origin"].unique())[10]]
        chart = viz.forecast_window(data.observations("CISO"), slice_, "test")

        # Altair hoists each frame into a top-level `datasets` map rather than inlining it
        # per layer, so the sizes are read from there.
        sizes = sorted(len(rows) for rows in chart.to_dict()["datasets"].values())
        assert sizes == [24, 24 + viz.LEAD_IN_HOURS], sizes
