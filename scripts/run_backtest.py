"""Run the full backtest for one market and append the result to a JSON file.

Split per market and written incrementally so a long run can be resumed rather than
restarted, and so a partial run still leaves usable results on disk.

Usage: python scripts/run_backtest.py CISO [--chronos] [--chronos-small] [--timesfm]
       [--timesfm-long] [--series-type NG] [--match-series D]
       [--out docs/backtest_results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from bellwether.eval.backtest import rolling_origin_backtest
from bellwether.forecast.baseline import DailySeasonalNaive, WeeklySeasonalNaive
from bellwether.storage.db import connect
from bellwether.storage.queries import load_series

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--series-type", default="D")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--chronos", action="store_true")
    parser.add_argument("--chronos-small", action="store_true")
    parser.add_argument("--timesfm", action="store_true")
    parser.add_argument(
        "--timesfm-long",
        action="store_true",
        help="TimesFM on its own context ceiling rather than Chronos's, as a separate arm",
    )
    parser.add_argument("--out", default="docs/backtest_results.json")
    parser.add_argument(
        "--match-series",
        help="Clip to another series type's span, so both are scored on one window set",
    )
    args = parser.parse_args()

    with connect(read_only=True) as conn:
        series = load_series(conn, args.respondent, args.series_type)
        if args.match_series:
            reference = load_series(conn, args.respondent, args.match_series)
            series = series.clip(reference.timestamps[0], reference.timestamps[-1])

    models: list = [WeeklySeasonalNaive(), DailySeasonalNaive()]
    if args.chronos:
        from bellwether.forecast.chronos import ChronosBolt

        models.append(ChronosBolt())
    if args.chronos_small:
        from bellwether.forecast.chronos import ChronosBoltSmall

        models.append(ChronosBoltSmall())
    if args.timesfm:
        from bellwether.forecast.timesfm import TimesFM

        models.append(TimesFM())
    if args.timesfm_long:
        # Roughly seven times the matched arm's per-window cost: the compiled graph is a
        # fixed size, so every window pays for the full context whether it can fill it or
        # not. About an hour a market against the matched arm's eight minutes.
        from bellwether.forecast.timesfm import LONG_CONTEXT, TimesFM

        models.append(TimesFM(context_limit=LONG_CONTEXT))

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    market = existing.setdefault(series.series_id, {})

    for model in models:
        if model.name in market:
            print(f"  {model.name}: already recorded, skipping")
            continue

        started = time.perf_counter()
        result = rolling_origin_backtest(
            model,
            series.values,
            series_id=series.series_id,
            horizon=args.horizon,
            max_windows=None,
        )
        elapsed = time.perf_counter() - started

        summary = result.summary()
        summary["windows"] = result.n_windows
        summary["seconds"] = round(elapsed, 1)
        market[model.name] = summary

        # Written after each model so an interrupted run keeps what it finished.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))

        print(
            f"  {model.name:<22} MASE {summary['mase']:.3f}  "
            f"WQL {summary['wql']:.4f}  sMAPE {summary['smape']:.2f}%  "
            f"cov {summary['coverage_80']:.1%}  ({elapsed:.0f}s)"
        )


if __name__ == "__main__":
    main()
