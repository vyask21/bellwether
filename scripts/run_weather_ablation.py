"""Run the weather ablation for one market and append the result to a JSON file.

Usage: python scripts/run_corrector_ablation.py ERCO [--out docs/weather_ablation.json]

Written incrementally, one market per invocation, so a long run can be resumed rather
than restarted. Roughly 5 minutes per market on CPU: the base model forecasts once over
every origin, then the correctors refit at each of them.

Note what is *not* configurable here. The origin set comes from where demand and weather
are both present, and all three arms are scored on it identically. There is no flag to
score one arm on more windows than another, because that comparison would be meaningless.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from bellwether.eval.ablation import cache_base_forecasts, run_corrector_ablation, usable_origins
from bellwether.eval.operator import BA_TIMEZONES
from bellwether.storage.db import connect
from bellwether.storage.queries import load_market_temperature, load_series

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--series-type", default="D")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--out", default="docs/weather_ablation.json")
    args = parser.parse_args()

    with connect(read_only=True) as conn:
        series = load_series(conn, args.respondent, args.series_type)
        temperature = load_market_temperature(conn, args.respondent, series.timestamps)

    from bellwether.forecast.chronos import ChronosBolt

    base = ChronosBolt()

    origins = usable_origins(series.values, temperature, args.horizon, initial_train_size=672)
    print(
        f"{series.series_id}: {series.values.size:,} hours, "
        f"{np.isfinite(temperature).sum():,} with temperature, {len(origins)} usable origins"
    )

    started = time.perf_counter()
    cached = cache_base_forecasts(base, series.values, origins, args.horizon)
    print(f"  base forecasts cached in {time.perf_counter() - started:.0f}s")

    started = time.perf_counter()
    output = run_corrector_ablation(
        base,
        series.values,
        series.timestamps,
        temperature,
        series_id=series.series_id,
        timezone=BA_TIMEZONES[args.respondent],
        horizon=args.horizon,
        cached=cached,
    )
    elapsed = time.perf_counter() - started

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    market = existing.setdefault(series.series_id, {})

    for name, result in output.results.items():
        summary = result.summary()
        summary["windows"] = result.n_windows
        market[name] = summary
        print(
            f"  {name:<28} MASE {summary['mase']:.3f}  "
            f"WQL {summary['wql']:.4f}  sMAPE {summary['smape']:.2f}%  "
            f"cov {summary['coverage_80']:.1%}  ({summary['windows']} windows)"
        )

    market["_meta"] = {
        "origins": len(origins),
        "scored_windows": len(output.scored_origins),
        "correction_seconds": round(elapsed, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
