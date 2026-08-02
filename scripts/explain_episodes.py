"""Attribute the worst breach episodes to evidence computed from stored data.

Usage: python scripts/explain_episodes.py ERCO [--top 5] [--arm chronos_bolt_base+scale]

Reads the episodes recorded by `analyze_breaches.py` and asks what stored data says about
each. Fast: no model runs, only queries.

Every number printed is computed here in Python. Nothing in this path asks a language model
for a quantity, which is the project's standing constraint and what makes a later generated
brief checkable line by line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bellwether.eval.breaches import BreachEpisode
from bellwether.eval.operator import BA_TIMEZONES
from bellwether.explain.evidence import find_data_spikes, gather_evidence
from bellwether.storage.db import connect
from bellwether.storage.queries import load_market_temperature, load_series

DEFAULT_ARM = "chronos_bolt_base+scale"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--analysis", default="docs/breach_analysis.json")
    args = parser.parse_args()

    analysis = json.loads(Path(args.analysis).read_text())
    series_id = f"{args.respondent}:D"
    recorded = analysis[series_id][args.arm]["worst_episodes"][: args.top]

    with connect(read_only=True) as conn:
        series = load_series(conn, args.respondent, "D")
        temperature = load_market_temperature(conn, args.respondent, series.timestamps)

    # Computed once rather than per episode: it is a scan of the whole series.
    spikes = find_data_spikes(series.values)
    timezone = BA_TIMEZONES[args.respondent]

    print(f"{series_id}  arm={args.arm}  {len(recorded)} worst episodes")
    print(f"data spikes in series: {spikes.size}\n")

    unexplained = 0
    for record in recorded:
        episode = _rebuild(record)
        evidence = gather_evidence(
            episode, series.timestamps, series.values, temperature, timezone, spike_indices=spikes
        )

        print(
            f"{'=' * 78}\n{record['start'][:16]} UTC  {record['duration_hours']}h "
            f"{record['direction']}  peak {record['peak_exceedance']:,.0f} MW "
            f"({record['peak_exceedance_ratio']:.2f} widths)"
        )
        if not evidence:
            unexplained += 1
            print("  no evidence found")
        for item in evidence:
            marker = "!!" if item.is_disqualifying else "  "
            print(f"{marker} [{item.kind} {item.strength:.2f}] {item.summary}")
        print()

    print(f"{unexplained} of {len(recorded)} episodes had no candidate explanation")


def _rebuild(record: dict) -> BreachEpisode:
    """Reconstruct an episode from its recorded JSON form."""
    return BreachEpisode(
        start=np.datetime64(record["start"]),
        end=np.datetime64(record["end"]),
        duration_hours=record["duration_hours"],
        direction=record["direction"],
        peak_at=np.datetime64(record["peak_at"]),
        peak_exceedance=record["peak_exceedance"],
        peak_exceedance_ratio=record["peak_exceedance_ratio"],
        total_exceedance=record["total_exceedance"],
        local_hour_start=record["local_hour_start"],
        month=record["month"],
    )


if __name__ == "__main__":
    main()
