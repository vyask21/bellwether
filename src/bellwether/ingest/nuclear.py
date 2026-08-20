"""Which reactors sit in which market, and what counts as a supply shock.

EIA's `generator-nuclear-outages` route is faceted by `facility` and `generator` and
carries no balancing-authority field, so the mapping below is this project's knowledge
rather than the agency's. It is hand-authored, deliberately short, and covers only the
three markets this project tracks. The route lists 66 facilities; the other 63 are in
markets nothing here forecasts.

## Why a step change is the event, and why "planned" is not a useful category

It is tempting to separate scheduled refuelling from unplanned trips and keep only the
trips, on the grounds that a refuelling outage is known in advance. That reasoning is
about the operator's knowledge, not the forecaster's. **The model is handed a history of
net generation and nothing else.** It holds no outage calendar, so a unit dropping 1,300 MW
overnight is equally unforeseen whether a control room scheduled it a year ago or a relay
tripped at 03:00. Both are step changes in supply that the recent history does not
anticipate.

So the event here is a **day on which a unit's outage jumps**, in either direction. A
return to service is as unforeseen as a loss and is counted with the same threshold, which
roughly doubles the population and costs nothing in honesty. Once a unit has been out for a
day, the outage is inside the model's context window and is no longer a surprise, which is
why the episode's later days are not events.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from bellwether.ingest.eia import OutageRow

# A unit dropping or returning this much in a day is treated as a supply shock. The units
# here run 503 to 1,340 MW, and partial derates cluster near half a unit, so a threshold
# below about 400 MW would start counting load-following rather than outages.
DEFAULT_STEP_MW = 400.0


@dataclass(frozen=True, slots=True)
class NuclearPlant:
    facility_id: str
    name: str
    market: str


# Verified against the route's own facility facet on 2026-08-20.
NUCLEAR_PLANTS: tuple[NuclearPlant, ...] = (
    NuclearPlant("6099", "Diablo Canyon", "CISO"),
    NuclearPlant("6145", "Comanche Peak", "ERCO"),
    NuclearPlant("6251", "South Texas Project", "ERCO"),
)

# Stated rather than left as an absence, because it is the finding's main limitation and a
# reader should meet it here rather than infer it from an empty list. PacifiCorp East has
# no nuclear generation at all, so no amount of work makes this a three-market result.
MARKETS_WITHOUT_NUCLEAR: tuple[str, ...] = ("PACE",)

# Excluded on purpose, and recorded so the exclusion is not rediscovered as an omission.
# Palo Verde is the largest nuclear plant in the country and CAISO imports from it, which
# makes it a tempting fourth entry. It belongs to AZPS. Its output reaches California
# through interchange, where this project already accounts for it as `TI`, and counting it
# as CISO generation would double-count it.
EXCLUDED_PLANTS: tuple[tuple[str, str, str], ...] = (
    ("6008", "Palo Verde", "AZPS, not CISO; reaches California as interchange"),
)


def plants_for(market: str) -> tuple[NuclearPlant, ...]:
    """The reactors in one market, which is empty for PACE and says so."""
    return tuple(plant for plant in NUCLEAR_PLANTS if plant.market == market)


def markets_with_nuclear() -> tuple[str, ...]:
    return tuple(dict.fromkeys(plant.market for plant in NUCLEAR_PLANTS))


def facility_ids() -> tuple[str, ...]:
    return tuple(plant.facility_id for plant in NUCLEAR_PLANTS)


def market_of(facility_id: str) -> str | None:
    for plant in NUCLEAR_PLANTS:
        if plant.facility_id == facility_id:
            return plant.market
    return None


@dataclass(frozen=True, slots=True)
class OutageStep:
    """A day on which one unit's outage jumped, in either direction."""

    day: date
    market: str
    facility_id: str
    facility_name: str
    generator: str
    change_mw: float  # positive when generation was lost, negative when it returned
    outage_mw: float  # the level after the step
    direction: str  # "loss" or "return"

    def as_dict(self) -> dict:
        return {
            "day": self.day.isoformat(),
            "market": self.market,
            "plant": self.facility_name,
            "unit": self.generator,
            "change_mw": round(self.change_mw, 1),
            "outage_mw": round(self.outage_mw, 1),
            "direction": self.direction,
        }


def find_outage_steps(
    rows: Iterable[OutageRow],
    step_mw: float = DEFAULT_STEP_MW,
) -> list[OutageStep]:
    """Find days where a unit's outage changed by more than `step_mw`.

    A missing day breaks the comparison rather than being bridged: an absent row means the
    route did not report, and treating the gap as continuity would invent a step change
    across it whose size depends on how long the gap was.
    """
    names = {plant.facility_id: plant.name for plant in NUCLEAR_PLANTS}
    by_unit: dict[tuple[str, str], list[OutageRow]] = {}
    for row in rows:
        by_unit.setdefault((row.facility_id, row.generator), []).append(row)

    steps: list[OutageStep] = []
    for (facility_id, generator), unit_rows in by_unit.items():
        market = market_of(facility_id)
        if market is None:
            continue
        ordered = sorted(unit_rows, key=lambda r: r.period)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if (current.period - previous.period).days != 1:
                continue
            if previous.outage_mw is None or current.outage_mw is None:
                continue
            change = current.outage_mw - previous.outage_mw
            if abs(change) < step_mw:
                continue
            steps.append(
                OutageStep(
                    day=current.period,
                    market=market,
                    facility_id=facility_id,
                    facility_name=names.get(facility_id, facility_id),
                    generator=generator,
                    change_mw=change,
                    outage_mw=current.outage_mw,
                    direction="loss" if change > 0 else "return",
                )
            )
    return sorted(steps, key=lambda s: (s.day, s.facility_id, s.generator))


def steps_by_market(steps: Sequence[OutageStep]) -> dict[str, list[OutageStep]]:
    grouped: dict[str, list[OutageStep]] = {market: [] for market in markets_with_nuclear()}
    for step in steps:
        grouped.setdefault(step.market, []).append(step)
    return grouped
