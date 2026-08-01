"""Where the forecast fails: error by hour of day, and interval breaches as episodes.

Usage: python scripts/analyze_breaches.py ERCO [--stagger 4]

Runs the ablation to get hour-level forecasts, then decomposes them. Roughly 3 minutes per
market per stagger offset, because it is the ablation plus arithmetic.

The default arm is the calendar-recalibrated one rather than the raw base model. Breach
detection on a miscalibrated interval measures the calibration error as much as the events:
at ERCO's raw 76.4% coverage a detector fires on 24% of hours while claiming 20%, so a
fifth of what it reports is the model being wrong about itself rather than the grid doing
something worth explaining.

## Why `--stagger` exists

Origins advance by exactly the horizon, so every local hour is always forecast at the same
lead time: local hour `(h0 + step) % 24` for a fixed `h0`. Hour of day and horizon step are
therefore the same variable wearing two labels, and a single run cannot tell "the evening
ramp is hard" apart from "three hours ahead is hard". The first run of this script produced
a diurnal profile that was entirely a horizon-step profile, and it looked completely
reasonable.

`--stagger N` repeats the run at N origin offsets spaced `24/N` hours apart. Each run is
internally still step-24, but pooling them crosses hour of day against horizon step so the
two can be read separately.

Episodes come from the offset-0 run alone. Pooled runs overlap by construction, so an hour
appears up to N times, which is correct for a profile and would double-count a breach.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from bellwether.eval.ablation import cache_base_forecasts, run_corrector_ablation, usable_origins
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
from bellwether.eval.operator import BA_TIMEZONES
from bellwether.storage.db import connect
from bellwether.storage.queries import load_market_temperature, load_series

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

DEFAULT_ARM = "chronos_bolt_base+calendar"

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--out", default="docs/breach_analysis.json")
    parser.add_argument(
        "--stagger",
        type=int,
        default=4,
        help="Origin offsets to pool, decoupling hour of day from horizon step.",
    )
    parser.add_argument(
        "--top-episodes", type=int, default=10, help="How many worst episodes to record."
    )
    args = parser.parse_args()

    if 24 % args.stagger:
        raise SystemExit(f"--stagger must divide 24, got {args.stagger}")

    with connect(read_only=True) as conn:
        series = load_series(conn, args.respondent, "D")
        temperature = load_market_temperature(conn, args.respondent, series.timestamps)

    from bellwether.forecast.chronos import ChronosBolt

    base = ChronosBolt()
    timezone = BA_TIMEZONES[args.respondent]

    # Every arm is profiled from the same ablation pass. The base model is the expensive
    # part and it is already shared across arms, so covering all of them costs arithmetic
    # and makes the seasonal comparison a single run rather than one run per arm.
    pooled: dict[str, list[HourlyRecords]] = {}
    episodes = None
    summary = None

    for offset in range(0, 24, 24 // args.stagger):
        train_size = 672 + offset
        origins = usable_origins(series.values, temperature, 24, initial_train_size=train_size)
        print(f"{series.series_id}: offset +{offset}h, {len(origins)} usable origins")

        cached = cache_base_forecasts(base, series.values, origins, 24)
        output = run_corrector_ablation(
            base,
            series.values,
            series.timestamps,
            temperature,
            series_id=series.series_id,
            timezone=timezone,
            initial_train_size=train_size,
            cached=cached,
        )
        if args.arm not in output.forecasts:
            raise SystemExit(f"Unknown arm {args.arm!r}; have {sorted(output.forecasts)}")

        for arm, windows in output.forecasts.items():
            records = hourly_records(
                series.values, series.timestamps, output.scored_origins, windows, timezone
            )
            pooled.setdefault(arm, []).append(records)

            # Episodes come from one non-overlapping run of the selected arm. Pooled runs
            # cover the same hour several times over, which is what makes the profile
            # crossing work and what would make an episode count fiction.
            if offset == 0 and arm == args.arm:
                episodes = find_episodes(records)
                summary = episode_summary(episodes, total_hours=len(records))

    profiles = {
        arm: {
            "hours_pooled": len(pool_records(runs)),
            "stagger": args.stagger,
            "by_local_hour": profile_by_local_hour(pool_records(runs)),
            "by_month": profile_by_month(pool_records(runs)),
            "by_horizon_step": profile_by_horizon_step(pool_records(runs)),
        }
        for arm, runs in pooled.items()
    }

    records = pool_records(pooled[args.arm])
    selected = profiles[args.arm]
    _print_report(
        series.series_id,
        args.arm,
        records,
        summary,
        selected["by_local_hour"],
        selected["by_month"],
        selected["by_horizon_step"],
        episodes,
        args.stagger,
    )
    _print_seasonal_comparison(profiles)

    worst = sorted(episodes, key=lambda e: e.total_exceedance, reverse=True)[: args.top_episodes]
    profiles[args.arm]["episodes"] = summary
    profiles[args.arm]["worst_episodes"] = [e.as_dict() for e in worst]

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing.setdefault(series.series_id, {}).update(profiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(f"\nwritten to {out_path}")


# Months holding fewer hours than this are the ragged ends of the scored period and are
# excluded from the spread, where a handful of hours would otherwise set the range.
MIN_MONTH_HOURS = 1000


def _print_seasonal_comparison(profiles: dict) -> None:
    """The direct test of whether conditioning the interval flattens its seasonal error.

    Coverage spread is the quantity the whole exercise is about: an interval that adapts to
    the season should hold roughly nominal coverage in every month rather than under-cover
    in one and over-cover in another.
    """
    print(f"\n{'arm':<38} {'coverage range':>20} {'spread':>8} {'width spread':>13}")
    for arm, profile in profiles.items():
        full = [row for row in profile["by_month"] if row["hours"] >= MIN_MONTH_HOURS]
        if not full:
            continue
        coverage = [row["coverage"] for row in full]
        widths = [row["mean_width"] for row in full]
        print(
            f"{arm:<38} {min(coverage):>9.1%}..{max(coverage):<9.1%} "
            f"{100 * (max(coverage) - min(coverage)):>6.0f}pt "
            f"{100 * (max(widths) / min(widths) - 1):>12.0f}%"
        )


def _print_report(
    series_id, arm, records, summary, hours, months, steps, episodes, stagger
) -> None:
    print(f"\n{'=' * 78}\n{series_id}  arm={arm}  {len(records):,} hours\n{'=' * 78}")
    print(
        f"profiles pool {stagger} origin offsets, so hour of day and horizon step vary "
        f"independently.\nEpisodes come from the offset-0 run alone."
    )

    print(
        f"\ncoverage {np.mean(~records.breached):.1%}   "
        f"episodes {summary['episodes']}   "
        f"breached hours {summary['breached_hours']:,} ({summary['breached_fraction']:.1%})   "
        f"above {summary['above']} / below {summary['below']}"
    )
    print(
        f"episode duration: mean {summary['mean_duration_hours']:.1f}h  "
        f"max {summary['max_duration_hours']}h  "
        f"over 3h: {summary['episodes_over_3h']}   "
        f"peak severity: mean {summary['mean_peak_ratio']:.2f}  "
        f"max {summary['max_peak_ratio']:.2f} interval widths"
    )

    print(
        f"\n{'local hour':>10} {'MAE':>8} {'bias':>8} {'coverage':>9} {'above':>7} "
        f"{'below':>7} {'width':>8}"
    )
    for row in hours:
        if not row["hours"]:
            continue
        flag = (
            "  <-- worst"
            if row["hour"] == max(hours, key=lambda r: r.get("mae", 0))["hour"]
            else ""
        )
        print(
            f"{row['hour']:>8}:00 {row['mae']:>8.0f} {row['bias']:>8.0f} "
            f"{row['coverage']:>8.1%} {row['breach_rate_above']:>6.1%} "
            f"{row['breach_rate_below']:>6.1%} {row['mean_width']:>8.0f}{flag}"
        )

    print(f"\n{'month':>10} {'MAE':>8} {'bias':>8} {'coverage':>9} {'width':>8}")
    for row in months:
        print(
            f"{MONTH_NAMES[row['month'] - 1]:>10} {row['mae']:>8.0f} {row['bias']:>8.0f} "
            f"{row['coverage']:>8.1%} {row['mean_width']:>8.0f}"
        )

    first, last = steps[0], steps[-1]
    print(
        f"\nhorizon step 1: MAE {first['mae']:.0f}, coverage {first['coverage']:.1%}   "
        f"step {last['step']}: MAE {last['mae']:.0f}, coverage {last['coverage']:.1%}"
    )

    print("\nworst episodes by total exceedance:")
    for episode in sorted(episodes, key=lambda e: e.total_exceedance, reverse=True)[:5]:
        print(
            f"  {str(episode.start)[:13]}  {episode.duration_hours:>2}h  "
            f"{episode.direction:<5}  peak {episode.peak_exceedance:>8.0f} MW "
            f"({episode.peak_exceedance_ratio:.2f} widths)  "
            f"from local {episode.local_hour_start:02d}:00"
        )


if __name__ == "__main__":
    main()
