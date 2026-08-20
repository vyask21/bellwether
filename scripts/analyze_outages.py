"""Do nuclear outages show up as net-generation forecast breaches?

Usage: python scripts/analyze_outages.py ERCO [--out docs/outage_analysis.json]

Finding 25 made this askable by forecasting a supply-side series. A reactor tripping does
not move demand, but it moves generation by 500 to 1,340 MW in a single day, so if supply
shocks are visible to a forecaster at all they should be visible here.

## What counts as an event

A day on which one unit's outage changed by more than `--step-mw`, in either direction. See
`bellwether.ingest.nuclear` for why scheduled refuelling is not excluded: the model holds no
outage calendar, so a planned shutdown is exactly as unforeseen to it as an unplanned one.

## The two tests, and which one is discriminating

The **rate test** asks whether a breach is more likely on an event day than on another day,
counted one observation per **day** rather than per hour: breached hours arrive in runs, and
counting them would treat a single six-hour episode as six independent facts. On its own it
is still weak, because it is confounded: this project's intervals run under nominal and
their miscoverage is seasonal, while refuelling clusters in the shoulder seasons. A raw rate
difference could be that overlap and nothing else. It is reported against two controls, all
other days and the non-event days of the same calendar months, so the confound is visible.

The **direction test** is the one that discriminates. A unit going out removes supply, so
generation should fall short of the forecast and breach the **lower** bound; a unit
returning should breach the **upper**. Seasonal miscalibration predicts no such pairing, so
if the breaches on event days are randomly signed the association is spurious however large
the rate difference is. This is the same shape as the volatility check in finding 10, which
is what caught that target being an artifact of its method.
"""

from __future__ import annotations

import argparse
import json
import logging
from math import comb, exp, lgamma
from pathlib import Path

import numpy as np
import pandas as pd

from bellwether.eval.ablation import cache_base_forecasts, usable_origins
from bellwether.eval.breaches import hourly_records
from bellwether.eval.operator import BA_TIMEZONES
from bellwether.ingest.nuclear import (
    DEFAULT_STEP_MW,
    MARKETS_WITHOUT_NUCLEAR,
    find_outage_steps,
    plants_for,
)
from bellwether.storage.db import connect
from bellwether.storage.queries import load_outage_rows, load_series

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--step-mw", type=float, default=DEFAULT_STEP_MW)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--out", default="docs/outage_analysis.json")
    args = parser.parse_args()

    market = args.respondent
    if market in MARKETS_WITHOUT_NUCLEAR:
        raise SystemExit(
            f"{market} has no nuclear generation, so there is nothing here to measure. "
            "That is the finding's limitation, not an error."
        )
    plants = plants_for(market)
    if not plants:
        raise SystemExit(f"No reactors mapped to {market!r}")

    timezone = BA_TIMEZONES[market]

    with connect(read_only=True) as conn:
        series = load_series(conn, market, "NG")
        demand = load_series(conn, market, "D")
        outage_rows = load_outage_rows(conn)

    # Clipped to demand's span so this sits on the same window set as finding 25.
    series = series.clip(demand.timestamps[0], demand.timestamps[-1])

    from bellwether.forecast.chronos import ChronosBolt

    origins = usable_origins(series.values, None, args.horizon, initial_train_size=672)
    print(f"{series.series_id}: {len(origins)} usable origins, {len(plants)} plant(s)")

    cached = cache_base_forecasts(ChronosBolt(), series.values, origins, args.horizon)
    records = hourly_records(
        series.values,
        series.timestamps,
        [c.origin for c in cached],
        [c.quantiles for c in cached],
        timezone,
    )

    daily = _daily_breaches(records, timezone)
    steps = [s for s in find_outage_steps(outage_rows, args.step_mw) if s.market == market]
    steps = [s for s in steps if s.day in daily.index]

    report = {
        "market": market,
        "plants": [{"id": p.facility_id, "name": p.name} for p in plants],
        "step_mw": args.step_mw,
        "scored_days": int(len(daily)),
        "scored_hours": int(daily.hours.sum()),
        "events": [s.as_dict() for s in steps],
        "rate_test": _rate_test(daily, steps),
        "burden_test": _burden_test(daily, steps),
        "direction_test": _direction_test(daily, steps),
        "per_event": _per_event(daily, steps),
        # Persisted so a later question about these days costs arithmetic rather than
        # another pass of the base model over 686 windows.
        "daily": [
            {
                "day": day.isoformat(),
                "below": int(row.below),
                "above": int(row.above),
                "hours": int(row.hours),
            }
            for day, row in daily.iterrows()
        ],
    }

    _print(report)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing[market] = report
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(f"\nWrote {out_path}")


def _daily_breaches(records, timezone: str) -> pd.DataFrame:
    """One row per local calendar day: hours scored, and hours outside each bound.

    Local rather than UTC because the outage route publishes a calendar day, and a day is
    only meaningful in somebody's clock. This is the join's coarsest link and is stated in
    the results as such.
    """
    local = pd.DatetimeIndex(records.timestamps).tz_localize("UTC").tz_convert(timezone)
    below = records.actual < records.lower
    above = records.actual > records.upper
    frame = pd.DataFrame(
        {
            "day": local.date,
            "below": below.astype(int),
            "above": above.astype(int),
            "hours": 1,
        }
    )
    return frame.groupby("day").sum()


def _rate_test(daily: pd.DataFrame, steps) -> dict:
    """Is a breach more likely on an event day?

    The unit is a **day**, not an hour. Breached hours arrive in runs, so counting hours
    would treat one six-hour episode as six independent observations and shrink the p-value
    by roughly that factor. A day either contained a breach or it did not, and days are the
    grain at which the outage route reports anyway.

    Mean breached hours per day rides along as a descriptive, because a day-level binary
    throws away how hard the day broke.
    """
    event_days = {s.day for s in steps}
    is_event = daily.index.isin(event_days)
    breached_hours = (daily.below + daily.above).values
    any_breach = breached_hours > 0

    # Matched control: the non-event days of the months that contain events, so a seasonal
    # overlap between refuelling and miscoverage cannot masquerade as an outage effect.
    months = {(d.year, d.month) for d in event_days}
    in_month = np.array([(d.year, d.month) in months for d in daily.index])
    matched = in_month & ~is_event

    def group(mask: np.ndarray) -> tuple[int, int, float]:
        days = int(mask.sum())
        broke = int(any_breach[mask].sum())
        mean_hours = float(breached_hours[mask].mean()) if days else 0.0
        return days, broke, mean_hours

    ev_days, ev_broke, ev_hours = group(is_event)
    ot_days, ot_broke, ot_hours = group(~is_event)
    mt_days, mt_broke, mt_hours = group(matched)

    return {
        "event_days": ev_days,
        "event_days_with_breach": ev_broke,
        "event_breach_rate": _safe_ratio(ev_broke, ev_days),
        "all_other_days_rate": _safe_ratio(ot_broke, ot_days),
        "same_month_control_rate": _safe_ratio(mt_broke, mt_days),
        "mean_breached_hours_event": round(ev_hours, 2),
        "mean_breached_hours_other": round(ot_hours, 2),
        "mean_breached_hours_same_month": round(mt_hours, 2),
        "p_value_vs_all_other": _fisher(ev_broke, ev_days - ev_broke, ot_broke, ot_days - ot_broke),
        "p_value_vs_same_month": _fisher(
            ev_broke, ev_days - ev_broke, mt_broke, mt_days - mt_broke
        ),
    }


def _burden_test(daily: pd.DataFrame, steps, permutations: int = 20_000) -> dict:
    """How many hours a day breaks, rather than whether it broke at all.

    The day-level binary in `_rate_test` is close to saturated: with intervals running
    under nominal, a breach lands somewhere in nearly every 24-hour day, so a yes/no
    question cannot see much. Counting breached hours keeps the resolution the binary
    throws away.

    Tested by permutation against the same-month control rather than by a t-test. The
    per-day counts are bounded at 0 and 24, heavily zero-inflated and plainly not normal,
    and the event group is a couple of dozen days; shuffling the labels assumes none of
    that. Seeded, so re-running the script reproduces the p-value exactly.
    """
    event_days = {s.day for s in steps}
    is_event = daily.index.isin(event_days)
    months = {(d.year, d.month) for d in event_days}
    in_month = np.array([(d.year, d.month) in months for d in daily.index])
    pool = in_month  # event days plus the non-event days of the same months
    breached = (daily.below + daily.above).values.astype(float)

    treated = breached[is_event]
    control = breached[pool & ~is_event]
    if treated.size == 0 or control.size == 0:
        return {"observed_difference": None, "p_value": 1.0, "permutations": 0}

    observed = float(treated.mean() - control.mean())
    combined = np.concatenate([treated, control])
    n_treated = treated.size

    rng = np.random.default_rng(20260820)
    extreme = 0
    for _ in range(permutations):
        rng.shuffle(combined)
        difference = combined[:n_treated].mean() - combined[n_treated:].mean()
        if abs(difference) >= abs(observed) - 1e-12:
            extreme += 1

    return {
        "event_mean_hours": round(float(treated.mean()), 3),
        "control_mean_hours": round(float(control.mean()), 3),
        "observed_difference": round(observed, 3),
        # +1 in both parts: the observed labelling is itself one of the arrangements, and
        # omitting it can report p = 0 for something a finite test never established.
        "p_value": round((extreme + 1) / (permutations + 1), 4),
        "permutations": permutations,
    }


def _direction_test(daily: pd.DataFrame, steps) -> dict:
    """Does a loss breach the lower bound and a return the upper?

    Counted once per event rather than per hour: hours within a day are not independent,
    and an event that breaches for six hours is one observation of the pairing, not six.
    An event with no breach in either direction is not a trial, since it says nothing about
    sign; this is a sign test on the events that moved something.
    """
    agree = disagree = 0
    for step in steps:
        if step.day not in daily.index:
            continue
        row = daily.loc[step.day]
        expected, opposite = ("below", "above") if step.direction == "loss" else ("above", "below")
        if row[expected] == row[opposite]:
            continue
        if row[expected] > row[opposite]:
            agree += 1
        else:
            disagree += 1

    trials = agree + disagree
    return {
        "as_predicted": agree,
        "against": disagree,
        "trials": trials,
        "p_value": _sign_test(agree, trials),
    }


def _per_event(daily: pd.DataFrame, steps) -> list[dict]:
    rows = []
    for step in steps:
        if step.day not in daily.index:
            continue
        row = daily.loc[step.day]
        rows.append(
            {
                **step.as_dict(),
                "hours_below": int(row.below),
                "hours_above": int(row.above),
                "hours_scored": int(row.hours),
            }
        )
    return rows


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def _fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on a 2x2 table, without scipy.

    Exact rather than chi-square because the event group is tens of hours against tens of
    thousands, which is exactly where the asymptotic approximation stops being safe.
    """
    n = a + b + c + d
    if n == 0 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        return 1.0
    row1, col1 = a + b, a + c

    # Log-gamma rather than exact `comb`. The tables here run to hundreds of days, and
    # binomial coefficients at that size are enormous integers whose exact arithmetic is
    # slow enough to matter; the logs are exact to well past the precision anyone reads.
    def log_choose(n_: int, k_: int) -> float:
        return lgamma(n_ + 1) - lgamma(k_ + 1) - lgamma(n_ - k_ + 1)

    denominator = log_choose(n, col1)

    def probability(x: int) -> float:
        return exp(log_choose(row1, x) + log_choose(n - row1, col1 - x) - denominator)

    observed = probability(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    # Floating comparison needs slack or the observed table can miss its own tail.
    total = sum(p for x in range(lo, hi + 1) if (p := probability(x)) <= observed * (1 + 1e-9))
    return min(1.0, total)


def _sign_test(successes: int, trials: int) -> float:
    """Two-sided sign test against a fair coin. Same test as the holiday arms use."""
    if trials == 0:
        return 1.0
    extreme = min(successes, trials - successes)
    tail = sum(comb(trials, k) for k in range(extreme + 1)) / 2**trials
    return min(1.0, 2 * tail)


def _print(report: dict) -> None:
    rate, direction = report["rate_test"], report["direction_test"]
    print(f"\n{report['market']}: {report['scored_days']} days, {report['scored_hours']:,} hours")
    print(f"  events (>= {report['step_mw']:.0f} MW step): {rate['event_days']}")
    print(f"  breach rate on event days      : {_pct(rate['event_breach_rate'])}")
    print(f"  breach rate on all other days  : {_pct(rate['all_other_days_rate'])}")
    print(f"  breach rate, same-month control: {_pct(rate['same_month_control_rate'])}")
    print(f"  p vs all other  : {rate['p_value_vs_all_other']:.4f}")
    print(f"  p vs same month : {rate['p_value_vs_same_month']:.4f}")
    burden = report["burden_test"]
    if burden.get("observed_difference") is not None:
        print(
            f"  breached hours/day: event {burden['event_mean_hours']:.2f} vs "
            f"control {burden['control_mean_hours']:.2f}, "
            f"difference {burden['observed_difference']:+.2f} "
            f"(permutation p = {burden['p_value']:.4f})"
        )
    print(
        f"  direction: {direction['as_predicted']} as predicted, "
        f"{direction['against']} against, of {direction['trials']} "
        f"(p = {direction['p_value']:.4f})"
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


if __name__ == "__main__":
    main()
