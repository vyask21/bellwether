"""Brief generation, and the verification that makes its constraint real.

The project's design constraint is that the language model never produces a number. That
is only true if something checks, so most of this file is adversarial: it feeds the
verifier fabricated briefs and asserts they are rejected. A verifier that passes
everything would let the constraint quietly lapse while every test still went green.

None of this needs an API key. The model call is exercised through a stub.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bellwether.eval.breaches import BreachEpisode
from bellwether.explain.brief import (
    BriefContext,
    generate_brief,
    render_prompt,
    traces_to,
    verify_brief,
)
from bellwether.explain.evidence import Evidence


def _episode(direction: str = "above") -> BreachEpisode:
    return BreachEpisode(
        start=np.datetime64("2025-01-05T22", "ns"),
        end=np.datetime64("2025-01-07T00", "ns"),
        duration_hours=26,
        direction=direction,
        peak_at=np.datetime64("2025-01-06T14", "ns"),
        peak_exceedance=12384.0,
        peak_exceedance_ratio=1.4,
        total_exceedance=180000.0,
        local_hour_start=16,
        month=1,
    )


def _context(evidence: list[Evidence] | None = None) -> BriefContext:
    if evidence is None:
        evidence = [
            Evidence(
                kind="temperature",
                summary="Temperature averaged 2.8 C against 15.4 C, an anomaly of -12.6 C.",
                facts={"episode_mean_c": 2.8, "baseline_mean_c": 15.4, "anomaly_c": -12.6},
                strength=0.85,
            )
        ]
    return BriefContext(market="ERCO", episode=_episode(), evidence=evidence)


class TestCitableValues:
    def test_episode_measurements_are_citable(self):
        allowed = _context().citable_numbers()
        assert 26.0 in allowed
        assert 12384.0 in allowed
        assert 1.4 in allowed

    def test_evidence_facts_are_citable(self):
        allowed = _context().citable_numbers()
        assert 2.8 in allowed
        assert -12.6 in allowed

    def test_a_fraction_may_be_cited_as_a_percentage(self):
        """0.604 and 60% are the same measurement; forbidding the readable one helps nobody."""
        context = _context(
            [Evidence(kind="data_quality", summary="x", facts={"deviation_fraction": 0.604})]
        )
        allowed = context.citable_numbers()
        assert 0.604 in allowed
        assert any(abs(v - 60.4) < 1e-9 for v in allowed)

    def test_booleans_are_not_citable_quantities(self):
        """bool subclasses int, so True would otherwise become the number 1."""
        without = _context([Evidence(kind="holiday", summary="x", facts={})]).citable_numbers()
        with_flag = _context(
            [Evidence(kind="holiday", summary="x", facts={"consistent_with_direction": True})]
        ).citable_numbers()

        assert with_flag == without, "a boolean fact must contribute no citable number"

    def test_dates_are_citable_strings_not_loose_integers(self):
        literals = _context().citable_strings()
        assert any("2025-01-05" in literal for literal in literals)


class TestTracing:
    def test_an_exact_value_traces(self):
        assert traces_to(12384.0, {12384.0})

    def test_a_rounded_value_traces(self):
        """A brief that says 'about 12,400 MW' is still citing the measurement."""
        assert traces_to(12400.0, {12384.0})
        assert traces_to(12000.0, {12384.0})

    def test_a_nearby_but_wrong_value_does_not_trace(self):
        assert not traces_to(12385.0, {12384.0})

    def test_an_invented_value_does_not_trace(self):
        assert not traces_to(50000.0, {12384.0, 1.4, 26.0})

    def test_decimal_rounding_traces(self):
        assert traces_to(-12.6, {-12.63})
        assert traces_to(1.4, {1.4012})


class TestVerification:
    def test_a_faithful_brief_passes(self):
        context = _context()
        text = (
            "ERCOT demand ran 26 hours above its forecast band, peaking 12,384 MW outside "
            "it. Temperature averaged 2.8 C against a baseline of 15.4 C."
        )
        assert verify_brief(text, context).ok

    def test_a_fabricated_number_is_caught(self):
        """The failure this whole module exists to prevent."""
        context = _context()
        text = "Demand ran 26 hours above the band, costing an estimated 45,000 MWh."

        result = verify_brief(text, context)
        assert not result.ok
        assert "45,000" in result.unverified

    def test_a_plausible_but_uncomputed_figure_is_caught(self):
        """The dangerous case: arithmetic on supplied values, which reads as a measurement."""
        context = _context()
        # 12,384 / 26 is a real calculation on real inputs and is still not a measurement.
        text = "Demand averaged 476 MW above the band across the episode."

        assert not verify_brief(text, context).ok

    def test_rounding_a_supplied_number_passes(self):
        context = _context()
        assert verify_brief("Demand peaked about 12,400 MW outside the band.", context).ok

    @pytest.mark.parametrize(
        "written",
        [
            "2025-01-05T22:00:00.000000000",
            "2025-01-05T22:00",
            "2025-01-05T22",
            "2025-01-05 22:00",
            "2025-01-05",
        ],
    )
    def test_a_timestamp_may_be_written_several_ways(self, written: str):
        """A writer given a numpy repr will sensibly shorten it; that must not be a rejection."""
        context = _context()
        assert verify_brief(f"The episode began {written} and ran 26 hours.", context).ok

    def test_a_date_not_in_the_evidence_is_caught(self):
        """Masking must not become a blanket exemption for anything date-shaped."""
        context = _context()
        result = verify_brief("A similar episode occurred on 2024-03-17.", context)
        assert not result.ok

    def test_prose_without_numbers_passes(self):
        context = _context()
        assert verify_brief("Unusual cold drove demand above the forecast band.", context).ok

    def test_percentages_derived_from_a_fraction_pass(self):
        context = _context(
            [Evidence(kind="data_quality", summary="x", facts={"deviation_fraction": 0.6})]
        )
        assert verify_brief("The reading was 60% below its neighbours.", context).ok

    def test_every_offending_number_is_reported_not_just_the_first(self):
        context = _context()
        result = verify_brief("Costs were 45,000 and 91,000 respectively.", context)
        assert len(result.unverified) == 2


class TestPrompt:
    def test_the_prompt_carries_the_evidence(self):
        prompt = render_prompt(_context())
        assert "12384" in prompt.replace(",", "")
        assert "anomaly_c" in prompt

    def test_an_episode_without_evidence_says_so(self):
        prompt = render_prompt(_context([]))
        assert "No candidate explanation" in prompt

    def test_the_prompt_is_the_only_source_of_numbers(self):
        """The allowlist is sound only if the model saw nothing the context lacks."""
        context = _context()
        prompt = render_prompt(context)
        for value in (12384.0, 26.0, 2.8):
            assert str(value).rstrip("0").rstrip(".") in prompt.replace(",", "")


class _StubClient:
    """Stands in for `anthropic.Anthropic`, returning scripted payloads."""

    def __init__(self, payloads: list[dict], stop_reason: str = "end_turn") -> None:
        self._payloads = list(payloads)
        self._stop_reason = stop_reason
        self.prompts: list[str] = []
        self.messages = self

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        payload = self._payloads.pop(0)

        class _Block:
            type = "text"
            text = json.dumps(payload)

        class _Message:
            stop_reason = self._stop_reason
            content = [_Block()]

        return _Message()


class TestGeneration:
    def _payload(self, body: str) -> dict:
        return {
            "headline": "Cold snap drove ERCOT demand above band",
            "body": body,
            "cause_known": True,
        }

    def test_a_verified_brief_is_returned(self):
        client = _StubClient([self._payload("Demand ran 26 hours above the band.")])
        brief = generate_brief(_context(), client)

        assert brief.verification.ok
        assert "26 hours" in brief.body

    def test_an_unverifiable_brief_is_retried_with_the_failure_named(self):
        client = _StubClient(
            [
                self._payload("Demand cost roughly 45,000 MWh."),
                self._payload("Demand ran 26 hours above the band."),
            ]
        )
        brief = generate_brief(_context(), client)

        assert brief.verification.ok
        assert "45,000" in client.prompts[1], "the retry should name what was rejected"

    def test_a_persistently_unverifiable_brief_raises_rather_than_returning(self):
        """Returning it with a warning would push the decision onto whoever reads the log."""
        client = _StubClient([self._payload("Cost 45,000 MWh."), self._payload("Cost 91,000 MWh.")])

        with pytest.raises(ValueError, match="cited numbers not present"):
            generate_brief(_context(), client)

    def test_a_refusal_is_surfaced_rather_than_indexed_into(self):
        """Safety classifiers return empty or partial content; indexing it would crash."""
        client = _StubClient([self._payload("x")], stop_reason="refusal")

        with pytest.raises(RuntimeError, match="declined"):
            generate_brief(_context(), client)


class TestDisqualification:
    def test_a_data_artifact_episode_is_flagged_on_the_context(self):
        context = _context(
            [Evidence(kind="data_quality", summary="artifact", facts={}, strength=1.0)]
        )
        assert context.is_disqualified

    def test_an_ordinary_episode_is_not(self):
        assert not _context().is_disqualified


class TestSystemConstants:
    """The nominal interval is a property of the detector, not a model-produced number."""

    def test_the_nominal_coverage_level_is_citable(self):
        assert verify_brief("Demand fell outside the 80% band.", _context()).ok

    def test_the_quantile_bounds_are_citable(self):
        assert verify_brief("The band spans the 10th to 90th percentile.", _context()).ok

    def test_a_nearby_percentage_is_still_rejected(self):
        """Allowing 80 must not quietly allow every two-digit number near it."""
        assert not verify_brief("Demand fell outside the 85% band.", _context()).ok
