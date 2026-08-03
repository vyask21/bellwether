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
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from bellwether.eval.ablation import (
    HOLIDAY_ARM,
    SCALE_ARM,
    cache_base_forecasts,
    holiday_flags,
    run_corrector_ablation,
    usable_origins,
)
from bellwether.eval.metrics import DEFAULT_QUANTILES, _quantile_index
from bellwether.eval.operator import BA_TIMEZONES
from bellwether.storage.db import connect
from bellwether.storage.queries import load_series


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
    index = np.concatenate([np.arange(o, o + 24) for o in output.scored_origins])
    median = _quantile_index(DEFAULT_QUANTILES, 0.5)

    local = pd.DatetimeIndex(series.timestamps[index]).tz_localize("UTC").tz_convert(timezone)
    holiday_dates = np.array([d.date() for d in local])
    on_holiday = flags[index]
    actual = series.values[index]

    errors = {}
    for arm in (f"{base.name}+{SCALE_ARM}", f"{base.name}+{HOLIDAY_ARM}"):
        predicted = np.concatenate([w[:, median] for w in output.forecasts[arm]])
        errors[arm] = np.abs(actual - predicted)

    control, treatment = errors[f"{base.name}+{SCALE_ARM}"], errors[f"{base.name}+{HOLIDAY_ARM}"]
    per_holiday = []
    for date in sorted({d for d, h in zip(holiday_dates, on_holiday, strict=True) if h}):
        mask = on_holiday & (holiday_dates == date)
        before, after = float(control[mask].mean()), float(treatment[mask].mean())
        per_holiday.append(
            {
                "date": str(date),
                "hours": int(mask.sum()),
                "mae_before": before,
                "mae_after": after,
                "change": after - before,
            }
        )

    improved = sum(1 for h in per_holiday if h["change"] < 0)
    n = len(per_holiday)
    print(f"\n{'holiday':<12}{'hours':>6}{'MAE before':>12}{'MAE after':>11}{'change':>10}")
    for h in per_holiday:
        mark = "  better" if h["change"] < 0 else "  worse"
        print(
            f"{h['date']:<12}{h['hours']:>6}{h['mae_before']:>12.0f}"
            f"{h['mae_after']:>11.0f}{h['change']:>+10.0f}{mark}"
        )

    print(f"\nimproved on {improved} of {n} holidays")
    print(f"two-sided sign test p = {_sign_test(improved, n):.4f}")
    overall = np.mean(treatment[on_holiday]) - np.mean(control[on_holiday])
    print(f"pooled holiday MAE change: {overall:+.0f} MW")

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing[series.series_id] = {
        "window": "full demand history, no weather constraint",
        "origins": len(origins),
        "scored_windows": len(output.scored_origins),
        "holidays": n,
        "improved": improved,
        "sign_test_p": _sign_test(improved, n),
        "pooled_mae_change": float(overall),
        "per_holiday": per_holiday,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(f"\nwritten to {out_path}")


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
