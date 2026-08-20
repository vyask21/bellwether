"""The reactor registry and what counts as a supply shock.

The registry is hand-authored because EIA's outage route carries no balancing-authority
field, so nothing upstream will ever fail if it drifts. These tests are the only thing that
holds it to what was verified against the route.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bellwether.ingest.eia import OutageRow, _parse_outage_row
from bellwether.ingest.nuclear import (
    EXCLUDED_PLANTS,
    MARKETS_WITHOUT_NUCLEAR,
    NUCLEAR_PLANTS,
    facility_ids,
    find_outage_steps,
    market_of,
    markets_with_nuclear,
    plants_for,
)

START = date(2025, 3, 1)


def _rows(outages, facility="6145", generator="1", start=START):
    """One unit's daily outage series, one value per consecutive day."""
    return [
        OutageRow(
            period=start + timedelta(days=i),
            facility_id=facility,
            generator=generator,
            capacity_mw=1205.0,
            outage_mw=value,
            percent_outage=None if value is None else 100.0 * value / 1205.0,
        )
        for i, value in enumerate(outages)
    ]


class TestRegistry:
    def test_pace_has_no_reactors_and_says_so(self):
        """The finding's central limitation. If this ever passes by accident, the whole
        two-market framing in RESULTS.md is wrong and should be rewritten, not patched."""
        assert plants_for("PACE") == ()
        assert "PACE" in MARKETS_WITHOUT_NUCLEAR

    def test_the_markets_that_do_have_reactors(self):
        assert set(markets_with_nuclear()) == {"ERCO", "CISO"}
        assert len(plants_for("ERCO")) == 2
        assert len(plants_for("CISO")) == 1

    def test_palo_verde_is_excluded_on_purpose_and_recorded(self):
        """It is the largest nuclear plant in the country and CAISO imports from it, which
        makes it the obvious wrong addition. Its output reaches California as interchange,
        where `TI` already accounts for it."""
        assert "6008" not in facility_ids()
        excluded = {entry[0] for entry in EXCLUDED_PLANTS}
        assert "6008" in excluded, "the exclusion must stay recorded, not just absent"

    def test_every_plant_maps_back_to_its_market(self):
        for plant in NUCLEAR_PLANTS:
            assert market_of(plant.facility_id) == plant.market
        assert market_of("9999") is None

    def test_facility_ids_are_unique(self):
        assert len(set(facility_ids())) == len(facility_ids())


class TestOutageSteps:
    def test_a_unit_going_out_is_a_loss(self):
        steps = find_outage_steps(_rows([0.0, 0.0, 1205.0, 1205.0]))
        assert len(steps) == 1
        assert steps[0].direction == "loss"
        assert steps[0].change_mw == pytest.approx(1205.0)
        assert steps[0].day == START + timedelta(days=2)
        assert steps[0].market == "ERCO"

    def test_a_unit_returning_is_a_return(self):
        steps = find_outage_steps(_rows([1205.0, 1205.0, 0.0]))
        assert len(steps) == 1
        assert steps[0].direction == "return"
        assert steps[0].change_mw == pytest.approx(-1205.0)

    def test_both_ends_of_an_outage_are_events(self):
        """A return to service is as unforeseen as a loss: the model holds no outage
        calendar either way. Counting only the onset would halve the population for no
        reason a forecaster would recognise."""
        steps = find_outage_steps(_rows([0.0, 1205.0, 1205.0, 1205.0, 0.0]))
        assert [s.direction for s in steps] == ["loss", "return"]

    def test_a_gradual_ramp_below_the_threshold_is_not_an_event(self):
        """Load-following and slow derates are not supply shocks. Four 300 MW days reach
        the same place as one 1,200 MW day and mean something entirely different."""
        steps = find_outage_steps(_rows([0.0, 300.0, 600.0, 900.0, 1200.0]), step_mw=400.0)
        assert steps == []

    def test_the_threshold_is_honoured(self):
        assert find_outage_steps(_rows([0.0, 500.0]), step_mw=400.0)
        assert find_outage_steps(_rows([0.0, 500.0]), step_mw=600.0) == []

    def test_a_missing_day_breaks_the_comparison_rather_than_bridging_it(self):
        """An absent row means the route did not report. Bridging the gap would invent a
        step whose size depends on how long the gap was, which is a fact about the outage
        history and not about any single day."""
        rows = _rows([0.0, 0.0, 1205.0])
        del rows[1]  # leave day 0 and day 2, a one-day hole
        assert find_outage_steps(rows) == []

    def test_units_are_tracked_separately(self):
        """Two units at one plant share a facility id. Pooling them would net a loss on
        one against a return on the other and see nothing happen."""
        rows = _rows([0.0, 1205.0], generator="1") + _rows([1205.0, 0.0], generator="2")
        steps = find_outage_steps(rows)
        assert {s.direction for s in steps} == {"loss", "return"}
        assert len(steps) == 2

    def test_unmapped_facilities_are_dropped(self):
        assert find_outage_steps(_rows([0.0, 1205.0], facility="6008")) == []

    def test_missing_values_do_not_produce_a_step(self):
        assert find_outage_steps(_rows([0.0, None, 1205.0])) == []

    def test_steps_come_back_in_day_order(self):
        rows = _rows([0.0, 1205.0, 1205.0, 0.0], generator="1") + _rows(
            [0.0, 0.0, 1205.0, 1205.0], generator="2"
        )
        steps = find_outage_steps(rows)
        assert [s.day for s in steps] == sorted(s.day for s in steps)


class TestParsing:
    def test_a_row_parses_into_its_typed_shape(self):
        row = _parse_outage_row(
            {
                "period": "2025-04-21",
                "facility": 6145,
                "generator": "1",
                "capacity": "1205",
                "outage": "1205",
                "percentOutage": "100",
            }
        )
        assert row.period == date(2025, 4, 21)
        assert row.facility_id == "6145"
        assert row.capacity_mw == pytest.approx(1205.0)

    def test_absent_numbers_stay_none_rather_than_becoming_zero(self):
        """Zero outage and unreported outage are different claims, and a supply shock is
        measured as a change, so silently reading one as the other would manufacture a
        1,205 MW step out of a reporting gap."""
        row = _parse_outage_row(
            {
                "period": "2025-04-21",
                "facility": 6145,
                "generator": "1",
                "capacity": None,
                "outage": "",
                "percentOutage": None,
            }
        )
        assert row.capacity_mw is None
        assert row.outage_mw is None
        assert row.percent_outage is None


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_outages  # noqa: E402


class TestStatistics:
    """The tests behind the finding. No scipy here, so these are hand-rolled and the only
    thing standing between a hand-rolled exact test and a published p-value is this class.
    """

    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            ((10, 10, 10, 10), 1.0),  # perfectly balanced
            ((9, 1, 1, 9), 0.0011),  # strong association
            ((0, 0, 5, 5), 1.0),  # an empty group is not evidence
        ],
    )
    def test_fisher_matches_known_values(self, table, expected):
        assert analyze_outages._fisher(*table) == pytest.approx(expected, abs=5e-4)

    def test_fisher_survives_a_table_too_big_for_exact_factorials(self):
        """The first draft used `math.comb` on n in the tens of thousands and would have
        hung rather than failed. Log-gamma is why this returns at all."""
        assert analyze_outages._fisher(80, 420, 100, 900) < 0.01
        assert analyze_outages._fisher(50, 450, 100, 900) == pytest.approx(1.0, abs=0.05)

    def test_sign_test_matches_the_holiday_arms(self):
        assert analyze_outages._sign_test(9, 10) == pytest.approx(0.0215, abs=1e-4)
        assert analyze_outages._sign_test(5, 10) == pytest.approx(1.0)
        assert analyze_outages._sign_test(0, 0) == 1.0

    def test_permutation_test_separates_a_planted_effect_from_noise(self):
        days = [date(2025, 3, 1) + timedelta(days=i) for i in range(60)]
        counts = np.array([i % 10 for i in range(60)])

        def frame(values):
            return pd.DataFrame({"below": values, "above": 0, "hours": 24}, index=days)

        class Step:
            def __init__(self, day):
                self.day = day

        events = [Step(days[i]) for i in range(20)]

        planted = counts.copy()
        planted[:20] = 0
        effect = analyze_outages._burden_test(frame(planted), events, permutations=2000)
        assert effect["p_value"] < 0.01
        assert effect["observed_difference"] < 0

        noise = analyze_outages._burden_test(frame(counts), events, permutations=2000)
        assert noise["p_value"] > 0.05

    def test_a_permutation_p_value_is_never_zero(self):
        """A finite permutation test cannot establish p = 0, and reporting it would claim
        more than 20,000 shuffles can support."""
        days = [date(2025, 3, 1) + timedelta(days=i) for i in range(40)]
        values = np.array([0] * 20 + [10] * 20)

        class Step:
            def __init__(self, day):
                self.day = day

        result = analyze_outages._burden_test(
            pd.DataFrame({"below": values, "above": 0, "hours": 24}, index=days),
            [Step(days[i]) for i in range(20)],
            permutations=500,
        )
        assert result["p_value"] > 0
