"""Turn evidence into prose, and prove the prose invented nothing.

Two paths write briefs, and both are checked the same way.

`render_template_brief` is **the shipped one**: deterministic Python, no API key, no
network, no dependency. `evidence.py` already writes each finding as a sentence, so a
brief is mostly a matter of choosing which findings to include and what to say when there
are none. That takes domain judgement rather than fluency, and judgement is cheaper to
encode than to verify.

`generate_brief` is the model path, kept for whoever wants better prose later. **It has
never been run against the live API** and needs an `anthropic` client passed in; the
package is not a dependency of this project. Its tests exercise it through a stub.

`verify_brief` is what makes either honest. Every numeric token in a brief must trace to a
value the context supplied, exactly or as a correct rounding. Anything else is rejected
rather than shown, because a number a reader cannot trace back is worse than no brief: it
looks like a measurement.

That check matters on the template path too, and not as ceremony. The templates
interpolate measurements, so an edit that computes one inline, a rate or a total or a unit
conversion, would otherwise reach a reader as a fabricated number wearing a template's
authority.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bellwether.eval.breaches import DEFAULT_LOWER, DEFAULT_UPPER, BreachEpisode
from bellwether.explain.evidence import Evidence

MODEL = "claude-opus-5"

# Rounding depths a citation may use. A brief written by a person rounds: 12,384 MW becomes
# "about 12,400 MW". That stays traceable, so it is allowed, and anything outside this
# range is not a rounding of a supplied value in any useful sense.
ROUNDING_PLACES = range(-4, 5)

# Numbers written with commas, decimals, or a leading sign. Deliberately greedy about
# commas so "12,384" is one token rather than two, which would otherwise let a fabricated
# figure slip past as two innocuous-looking small integers.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Tolerance for float comparison, well below the precision of anything cited here.
_EPSILON = 1e-9

# Constants of the system rather than measurements of an episode. A brief will naturally
# say "the 80% band", and rejecting that would be the verifier failing on a number nothing
# generated. Derived from the interval the detector actually uses, so changing that changes
# what a brief may cite, rather than leaving a stale literal behind.
SYSTEM_CONSTANTS: set[float] = {
    round((DEFAULT_UPPER - DEFAULT_LOWER) * 100.0),  # nominal coverage, 80
    DEFAULT_LOWER * 100.0,  # lower quantile as a percent
    DEFAULT_UPPER * 100.0,  # upper quantile as a percent
    DEFAULT_LOWER,
    DEFAULT_UPPER,
}


@dataclass(slots=True)
class BriefContext:
    """Everything the model may see, and therefore everything it may cite."""

    market: str
    episode: BreachEpisode
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def is_disqualified(self) -> bool:
        """Whether the evidence says this episode should not be explained at all."""
        return any(item.is_disqualifying for item in self.evidence)

    def citable_numbers(self) -> set[float]:
        """Every numeric value the brief is permitted to contain.

        Drawn from the episode and from each evidence item's facts, so the allowlist is
        derived from what the model was actually shown rather than maintained by hand
        beside it. A hand-maintained list would drift, and it would drift in the direction
        of permitting more.
        """
        values: set[float] = {
            float(self.episode.duration_hours),
            float(self.episode.peak_exceedance),
            float(self.episode.peak_exceedance_ratio),
            float(self.episode.total_exceedance),
            float(self.episode.local_hour_start),
            float(self.episode.month),
        }
        for item in self.evidence:
            values.update(_numeric_leaves(item.facts))

        # A fraction may be cited as a percentage. This is the one derivation allowed,
        # because "0.604" and "60%" are the same measurement in different clothes and
        # forbidding the readable one would only push briefs toward the unreadable one.
        values.update(value * 100.0 for value in list(values) if 0.0 < abs(value) < 1.0)
        return values | SYSTEM_CONSTANTS

    def citable_strings(self) -> set[str]:
        """Literal strings, mostly timestamps, that legitimately contain digits.

        Masked out before numbers are extracted. Without this, every timestamp in a brief
        would surface as three or four unexplained integers.

        Each timestamp contributes several renderings rather than one. A writer given
        `2025-01-05T22:00:00.000000000` will sensibly write `2025-01-05 22:00`, and
        rejecting that would push briefs toward machine-formatted dates nobody wants to
        read. Every accepted form is derived from the same instant, so this widens how a
        timestamp may be written without widening which timestamps are citable.
        """
        literals = {self.market}
        for stamp in (self.episode.start, self.episode.end, self.episode.peak_at):
            literals.update(_timestamp_forms(stamp))
        for item in self.evidence:
            for literal in _string_leaves(item.facts):
                literals.add(literal)
                literals.update(_timestamp_forms(literal))
        return {literal for literal in literals if literal}


def _timestamp_forms(stamp: Any) -> set[str]:
    """Renderings of one instant that a brief may legitimately use.

    Returns an empty set for anything that is not a timestamp, so ordinary strings in the
    facts pass through untouched.
    """
    try:
        moment = pd.Timestamp(str(stamp))
    except (ValueError, TypeError):
        return set()
    if pd.isna(moment):
        return set()
    return {
        str(stamp),
        moment.isoformat(),
        f"{moment:%Y-%m-%dT%H:%M}",
        f"{moment:%Y-%m-%dT%H}",
        f"{moment:%Y-%m-%d %H:%M}",
        f"{moment:%Y-%m-%d %H:00}",
        f"{moment:%Y-%m-%d}",
    }


def _numeric_leaves(value: Any) -> set[float]:
    """Every number anywhere in a nested facts structure."""
    if isinstance(value, bool):
        # bool is an int subclass, and "True" is not a citable quantity.
        return set()
    if isinstance(value, int | float):
        return {float(value)}
    if isinstance(value, dict):
        return set().union(*(_numeric_leaves(v) for v in value.values())) if value else set()
    if isinstance(value, list | tuple):
        return set().union(*(_numeric_leaves(v) for v in value)) if value else set()
    return set()


def _string_leaves(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_string_leaves(v) for v in value.values())) if value else set()
    if isinstance(value, list | tuple):
        return set().union(*(_string_leaves(v) for v in value)) if value else set()
    return set()


@dataclass(slots=True)
class Verification:
    """The result of checking a brief against what it was allowed to cite."""

    ok: bool
    unverified: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.ok:
            return "every number traces to supplied evidence"
        return "unverified numbers: " + ", ".join(self.unverified)


def traces_to(written: float, allowed: set[float]) -> bool:
    """Whether a written number is a supplied value, or a correct rounding of one."""
    for value in allowed:
        if abs(written - value) < _EPSILON:
            return True
        if any(abs(round(value, places) - written) < _EPSILON for places in ROUNDING_PLACES):
            return True
    return False


def verify_brief(text: str, context: BriefContext) -> Verification:
    """Check that every number in `text` traces to something in `context`.

    Literal strings are masked before extraction, so a date contributes no loose integers.
    Everything that survives must be a supplied value or a rounding of one.
    """
    masked = text
    # Longest first, so a longer literal is consumed before a shorter one that is a
    # prefix of it and would otherwise leave a fragment behind.
    for literal in sorted(context.citable_strings(), key=len, reverse=True):
        masked = masked.replace(literal, " ")

    allowed = context.citable_numbers()
    unverified = []
    for match in _NUMBER.finditer(masked):
        token = match.group()
        written = float(token.replace(",", ""))
        if not traces_to(written, allowed):
            unverified.append(token)

    return Verification(ok=not unverified, unverified=unverified)


MARKET_NAMES = {
    "CISO": "California ISO",
    "ERCO": "ERCOT",
    "PACE": "PacifiCorp East",
}


def render_template_brief(context: BriefContext) -> Brief:
    """Assemble a brief in Python, with no model involved.

    This is the shipped path. `evidence.py` already writes each finding as a sentence, so
    a brief is mostly a matter of choosing which findings to include, in what order, and
    what to say when there is nothing to report. That needs judgement about the domain,
    not fluency, and judgement is cheaper to encode than to verify.

    The result is passed through the same verifier the model path uses. That is not
    ceremony: the templates interpolate measurements, and a future edit that computes one
    inline, a rate or a total or a unit conversion, would be caught here rather than
    reaching a reader as a fabricated number wearing a template's authority.
    """
    market = MARKET_NAMES.get(context.market, context.market)
    episode = context.episode
    disqualifying = [item for item in context.evidence if item.is_disqualifying]
    supporting = [
        item
        for item in context.evidence
        if not item.is_disqualifying and item.facts.get("consistent_with_direction", True)
    ]
    contrary = [
        item
        for item in context.evidence
        if not item.is_disqualifying and item.facts.get("consistent_with_direction", True) is False
    ]

    if disqualifying:
        headline = f"{market} demand for this hour is not a real value"
        body = " ".join(
            [
                disqualifying[0].summary,
                "The forecast was not wrong here, and this episode should be excluded "
                "rather than explained.",
            ]
        )
        return _verified(headline, body, cause_known=False, context=context)

    span = _describe_span(episode, market)

    if supporting:
        headline = f"{market} demand ran {episode.duration_hours} hours "
        headline += "above forecast" if episode.direction == "above" else "below forecast"
        headline += f" {_headline_cause(supporting[0])}"
        body = " ".join([span, *(item.summary for item in supporting)])
        return _verified(headline, body, cause_known=True, context=context)

    headline = (
        f"{market} demand ran {episode.duration_hours} hours "
        f"{'above' if episode.direction == 'above' else 'below'} forecast, cause unexplained"
    )
    parts = [span, "No stored evidence explains this episode."]
    if contrary:
        # Naming what was checked and rejected is more useful than silence: it tells the
        # reader which explanations they do not need to re-investigate.
        parts.append(
            "The following was considered and does not account for it: "
            + " ".join(item.summary for item in contrary)
        )
    return _verified(headline, " ".join(parts), cause_known=False, context=context)


def _describe_span(episode: BreachEpisode, market: str) -> str:
    direction = "above" if episode.direction == "above" else "below"
    return (
        f"{market} demand ran {direction} its 80% forecast band for "
        f"{episode.duration_hours} consecutive hours from {_readable(episode.start)}, "
        f"reaching {episode.peak_exceedance:,.0f} MW outside the band at its peak."
    )


def _readable(stamp: Any) -> str:
    """A timestamp a person would write. Must stay among the forms the verifier accepts."""
    return f"{pd.Timestamp(str(stamp)):%Y-%m-%d %H:%M} UTC"


def _headline_cause(item: Evidence) -> str:
    if item.kind == "holiday":
        return "on a public holiday"
    if item.kind == "temperature":
        load_change = item.facts.get("degree_day_change")
        if load_change is None:
            # Without the degree-day change there is no way to tell which direction the
            # weather pushed demand, and defaulting would let a hard freeze be announced
            # as mild. Say less instead.
            return "during unusual weather"
        if load_change > 0:
            # Weather asking more of the grid than usual: heat or cold, depending which.
            return (
                "during unusual heat"
                if item.facts.get("anomaly_c", 0.0) > 0
                else "during unusual cold"
            )
        # Weather asking less of it. Whether that arrived as a cool spell in summer or a
        # mild one in winter does not change the headline, and naming both would.
        return "in unusually mild weather"
    return f"attributed to {item.kind}"


def _verified(headline: str, body: str, cause_known: bool, context: BriefContext) -> Brief:
    """Build a Brief and check it, so a template edit cannot smuggle in a new number."""
    return Brief(
        headline=headline,
        body=body,
        cause_known=cause_known,
        verification=verify_brief(f"{headline} {body}", context),
    )


SYSTEM_PROMPT = """\
You write short operational briefs about electricity demand forecast errors.

An interval breach is an hour where actual demand fell outside the forecast's stated 80%
confidence band. You are given one episode of consecutive breached hours and the evidence
a separate analysis computed for it.

Rules, in order of importance:

1. Never state a number that is not in the evidence you were given. Do not compute,
   estimate, convert units, sum, average, or infer any quantity. If you want to express a
   magnitude the evidence does not contain, describe it in words instead. Rounding a
   supplied number is acceptable.
2. Do not assert a cause the evidence does not support. Evidence marked as inconsistent
   with the episode's direction argues against itself being the cause; say so rather than
   omitting it.
3. If the evidence says the episode is a data artifact, the brief's job is to say the
   reported demand is not real and should not be explained as a grid event.
4. Prefer saying you do not know. An episode with no evidence gets a brief that says the
   cause is unexplained.

Write for a grid operator reading quickly. No preamble, no restating the question."""


def render_prompt(context: BriefContext) -> str:
    """The user message: the episode, the evidence, and nothing else.

    Numbers reach the model only through here, which is what makes the allowlist derived
    from this same context a sound check rather than a coincidence.
    """
    payload = {
        "market": context.market,
        "episode": {
            "start": str(context.episode.start),
            "end": str(context.episode.end),
            "duration_hours": context.episode.duration_hours,
            "direction": context.episode.direction,
            "peak_exceedance_mw": round(context.episode.peak_exceedance, 1),
            "peak_exceedance_interval_widths": round(context.episode.peak_exceedance_ratio, 2),
            "peak_at": str(context.episode.peak_at),
        },
        "evidence": [
            {"kind": item.kind, "finding": item.summary, "measurements": item.facts}
            for item in context.evidence
        ]
        or "No candidate explanation was found for this episode.",
    }
    return (
        "Write a brief for this breach episode.\n\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        "Respond with a headline of at most twelve words and a body of at most four "
        "sentences."
    )


BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "body": {"type": "string"},
        "cause_known": {
            "type": "boolean",
            "description": "False when the evidence does not establish a cause.",
        },
    },
    "required": ["headline", "body", "cause_known"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class Brief:
    """A generated brief and the result of checking it."""

    headline: str
    body: str
    cause_known: bool
    verification: Verification

    @property
    def text(self) -> str:
        return f"{self.headline}\n\n{self.body}"


def generate_brief(context: BriefContext, client: Any, max_attempts: int = 2) -> Brief:
    """Ask the model for a brief and verify it before returning.

    A brief that cites an unverifiable number is retried once with the failure named, then
    raised rather than returned. Returning it with a warning attached would put the
    decision on whoever reads the log, and the whole point of the constraint is that
    nobody has to.

    `client` is an `anthropic.Anthropic`. It is a parameter rather than a module global so
    the verification path can be exercised without one.
    """
    prompt = render_prompt(context)
    last: Verification | None = None

    for _attempt in range(max_attempts):
        message = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": BRIEF_SCHEMA}},
            messages=[{"role": "user", "content": _with_retry_note(prompt, last)}],
        )

        # Safety classifiers can decline; content is empty or partial when they do, so
        # this has to be checked before indexing into it.
        if message.stop_reason == "refusal":
            raise RuntimeError(f"Model declined to write a brief for {context.market}")

        payload = json.loads("".join(b.text for b in message.content if b.type == "text"))
        verification = verify_brief(f"{payload['headline']} {payload['body']}", context)
        if verification.ok:
            return Brief(
                headline=payload["headline"],
                body=payload["body"],
                cause_known=payload["cause_known"],
                verification=verification,
            )
        last = verification

    raise ValueError(
        f"Brief for {context.market} cited numbers not present in the evidence after "
        f"{max_attempts} attempts: {last.describe() if last else 'unknown'}"
    )


def _with_retry_note(prompt: str, failure: Verification | None) -> str:
    if failure is None:
        return prompt
    return (
        f"{prompt}\n\nYour previous attempt was rejected. It contained "
        f"{failure.describe()}. Every number must come from the evidence above."
    )
