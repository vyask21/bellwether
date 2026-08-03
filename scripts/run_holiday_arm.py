"""Test the holiday corrector on the full demand history, one paired observation per holiday.

Usage: python scripts/run_holiday_arm.py ERCO [--out docs/holiday_arm.json]

## Why this runs on a different window from every other experiment

The holiday arm uses no weather, so it is not bound by the NCEI archive ending eleven
months before EIA's data does. Dropping that constraint roughly doubles the usable period
and the number of holidays in it. **Nothing here is comparable to a weather-scored result**;
both arms below are scored on this window and only against each other.

## Why the statistic is a paired per-holiday difference

Coverage on holiday hours is hopeless as evidence. Hours inside one holiday are one event,
so the effective sample is the holiday count, and at roughly ten a year the standard error
on a coverage rate stays near ten points however long the history is. Resolving a
five-point effect at two standard errors would need about 256 holidays, which is 25 years.

A paired comparison does work. Each holiday contributes one number: the change in mean
absolute error when the shift is applied. Nine holidays all improving is a sign test at
p = 0.004, and that is a real result at a sample size this experiment can actually reach.
Consistency across holidays is the evidence; the size of any single one is not.

## What the class arm is being asked

The pooled arm was measured, found to improve 28 of 33 widely-observed holidays and 10 of
27 federal-only ones, and declined on the strength of a corrected p of 0.124 over three
markets. The class arm splits the offset along that line. Its control is therefore the
**pooled arm**, not the scale arm: the open question is whether splitting helps, and
scoring it against `+scale` would hand it credit for the shift itself.

Both comparisons are reported below because both are needed. Against `+scale` says whether
a holiday calendar is worth having at all; against `+holiday` says whether this version of
it is better than the last one.

**Read the p-values as exploratory.** The class split was chosen after seeing which
holidays the pooled arm failed on. The grouping itself comes from outside this repository
(private-sector paid time off, `WIDELY_OBSERVED_HOLIDAYS`), but the decision to look came
from the failures, and no amount of care about the former fixes the latter. A clean test
needs holidays this window does not contain.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from bellwether.eval.ablation import (
    FEDERAL_ONLY,
    HOLIDAY_ARM,
    HOLIDAY_CLASS_ARM,
    SCALE_ARM,
    WIDELY_OBSERVED,
    cache_base_forecasts,
    holiday_class_flags,
    holiday_flags,
    run_corrector_ablation,
    usable_origins,
)
from bellwether.eval.metrics import DEFAULT_QUANTILES, _quantile_index
from bellwether.eval.operator import BA_TIMEZONES
from bellwether.forecast.residual import HolidayClassScaleCorrector
from bellwether.storage.db import connect
from bellwether.storage.queries import load_series

CLASS_LABELS = {WIDELY_OBSERVED: "widely observed", FEDERAL_ONLY: "federal only"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--out", default="docs/holiday_arm.json")
    args = parser.parse_args()

    timezone = BA_TIMEZONES[args.respondent]
    with connect(read_only=True) as conn:
        series = load_series(conn, args.respondent, "D")

    from bellwether.forecast.chronos import ChronosBolt

    base = ChronosBolt()
    # No temperature: the weather arms are excluded, so the weather window does not bind.
    origins = usable_origins(series.values, None, 24, initial_train_size=672)
    print(f"{series.series_id}: {len(origins)} usable origins on the full demand history")

    cached = cache_base_forecasts(base, series.values, origins, 24)
    output = run_corrector_ablation(
        base,
        series.values,
        series.timestamps,
        None,
        series_id=series.series_id,
        timezone=timezone,
        specs=(),  # base, +scale, +scale+holiday only
        cached=cached,
    )

    flags = holiday_flags(series.timestamps, timezone)
    classes = holiday_class_flags(series.timestamps, timezone)
    index = np.concatenate([np.arange(o, o + 24) for o in output.scored_origins])
    median = _quantile_index(DEFAULT_QUANTILES, 0.5)

    local = pd.DatetimeIndex(series.timestamps[index]).tz_localize("UTC").tz_convert(timezone)
    holiday_dates = np.array([d.date() for d in local])
    on_holiday = flags[index]
    hour_class = classes[index]
    actual = series.values[index]

    arms = {
        "scale": f"{base.name}+{SCALE_ARM}",
        "pooled": f"{base.name}+{HOLIDAY_ARM}",
        "class": f"{base.name}+{HOLIDAY_CLASS_ARM}",
    }
    errors = {}
    for key, arm in arms.items():
        predicted = np.concatenate([w[:, median] for w in output.forecasts[arm]])
        errors[key] = np.abs(actual - predicted)

    per_holiday = []
    for date in sorted({d for d, h in zip(holiday_dates, on_holiday, strict=True) if h}):
        mask = on_holiday & (holiday_dates == date)
        row = {
            "date": str(date),
            "hours": int(mask.sum()),
            "observance": CLASS_LABELS[int(hour_class[mask][0])],
        }
        for key in arms:
            row[f"mae_{key}"] = float(errors[key][mask].mean())
        # Two paired differences per holiday: against the shift-free control, and against
        # the pooled shift this arm exists to improve on.
        row["change_vs_scale"] = row["mae_class"] - row["mae_scale"]
        row["change_vs_pooled"] = row["mae_class"] - row["mae_pooled"]
        row["pooled_change_vs_scale"] = row["mae_pooled"] - row["mae_scale"]
        per_holiday.append(row)

    n = len(per_holiday)
    print(
        f"\n{'holiday':<12}{'observance':>16}{'+scale':>10}{'+holiday':>10}"
        f"{'+class':>10}{'class-pooled':>14}"
    )
    for h in per_holiday:
        mark = "  better" if h["change_vs_pooled"] < 0 else "  worse"
        print(
            f"{h['date']:<12}{h['observance']:>16}{h['mae_scale']:>10.0f}"
            f"{h['mae_pooled']:>10.0f}{h['mae_class']:>10.0f}"
            f"{h['change_vs_pooled']:>+14.0f}{mark}"
        )

    summary = {}
    against_scale = (("class arm", "change_vs_scale"), ("pooled arm", "pooled_change_vs_scale"))
    for label, field in against_scale:
        counts = _counts_by_class(per_holiday, field)
        summary[field] = counts
        print(f"\n{label} against +scale, by observance:")
        _print_counts(counts)

    head_to_head = _counts_by_class(per_holiday, "change_vs_pooled")
    summary["change_vs_pooled"] = head_to_head
    print("\nclass arm against pooled arm, by observance:")
    _print_counts(head_to_head)

    pooled_mae_change = float(np.mean(errors["class"][on_holiday] - errors["scale"][on_holiday]))
    versus_pooled = float(np.mean(errors["class"][on_holiday] - errors["pooled"][on_holiday]))
    print(f"\nholiday-hour MAE change, class arm vs +scale:  {pooled_mae_change:+.0f} MW")
    print(f"holiday-hour MAE change, class arm vs pooled:  {versus_pooled:+.0f} MW")

    # Refit once on everything the last origin saw, purely to report what the arm learned.
    # Not used for scoring: the scored arms each refit on their own history.
    stacked_base = np.concatenate([c.quantiles for c in cached])
    final = HolidayClassScaleCorrector(DEFAULT_QUANTILES).fit(
        stacked_base,
        np.concatenate([series.values[c.origin : c.origin + 24] for c in cached]),
        np.concatenate([classes[c.origin : c.origin + 24] for c in cached]),
    )
    learned = {CLASS_LABELS[code]: value for code, value in final.offsets.items()}
    print("\noffsets learned on the full window:")
    print(f"  pooled{'':<16}{final.offset:>+10.0f} MW")
    for label, value in learned.items():
        print(f"  {label:<22}{value:>+10.0f} MW")

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing[series.series_id] = {
        "window": "full demand history, no weather constraint",
        "origins": len(origins),
        "scored_windows": len(output.scored_origins),
        "holidays": n,
        "sign_tests": summary,
        "class_mae_change_vs_scale": pooled_mae_change,
        "class_mae_change_vs_pooled": versus_pooled,
        "learned_offsets": {"pooled": final.offset, **learned},
        "per_holiday": per_holiday,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(f"\nwritten to {out_path}")


def _counts_by_class(per_holiday: list[dict], field: str) -> dict[str, dict[str, float]]:
    """Improvement counts and a sign test, overall and within each observance class."""
    groups = {"all": per_holiday}
    for label in CLASS_LABELS.values():
        groups[label] = [h for h in per_holiday if h["observance"] == label]

    counts = {}
    for label, rows in groups.items():
        improved, trials = sum(1 for h in rows if h[field] < 0), len(rows)
        counts[label] = {
            "improved": improved,
            "holidays": trials,
            "sign_test_p": _sign_test(improved, trials),
        }
    return counts


def _print_counts(counts: dict[str, dict[str, float]]) -> None:
    for label, c in counts.items():
        print(
            f"  {label:<16}improved on {c['improved']:>2} of {c['holidays']:>2}"
            f"   p = {c['sign_test_p']:.4f}"
        )


def _sign_test(successes: int, trials: int) -> float:
    """Two-sided sign test against a fair coin.

    The question is whether the shift helps *consistently*, not whether any one holiday
    moved a lot. Under the null that it is as likely to hurt as help, the count of
    improvements is binomial with p = 0.5, and that needs no assumption about how large
    the effect is or how it is distributed, which at this sample size is the point.
    """
    if trials == 0:
        return 1.0
    extreme = min(successes, trials - successes)
    tail = sum(comb(trials, k) for k in range(extreme + 1)) / 2**trials
    return min(1.0, 2 * tail)


if __name__ == "__main__":
    main()
