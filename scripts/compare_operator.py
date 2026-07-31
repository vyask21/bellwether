"""Compare our models against the balancing authority's own day-ahead forecast.

Both are scored over identical windows, aligned to local midnight, so each forecaster has
the same information cutoff: everything through the end of the previous local day.

Usage: python scripts/compare_operator.py CISO [--chronos]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from bellwether.eval.backtest import rolling_origin_backtest
from bellwether.eval.operator import (
    BA_TIMEZONES,
    evaluate_stored_forecast,
    local_midnight_origins,
)
from bellwether.forecast.baseline import DailySeasonalNaive
from bellwether.storage.db import connect
from bellwether.storage.queries import load_aligned_series

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

SEASON_LENGTH = 168
HORIZON = 24
TRAIN_SIZE = SEASON_LENGTH * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--chronos", action="store_true")
    parser.add_argument("--out", default="docs/operator_comparison.json")
    args = parser.parse_args()

    # Both series must sit on one grid. Loading them separately gives each its own
    # grid derived from its own bounds, which silently offsets the comparison.
    with connect(read_only=True) as conn:
        timestamps, values = load_aligned_series(conn, args.respondent, ["D", "DF"])
    actual, operator_forecast = values["D"], values["DF"]
    series_id = f"{args.respondent}:D"

    timezone = BA_TIMEZONES[args.respondent]
    origins = local_midnight_origins(timestamps, timezone, min_index=TRAIN_SIZE, horizon=HORIZON)
    print(f"{series_id}: {len(origins)} local-midnight origins ({timezone})")
    print(f"  DF gaps: {np.mean(~np.isfinite(operator_forecast)):.2%}")

    results: dict[str, dict] = {}

    op = evaluate_stored_forecast(
        actual,
        operator_forecast,
        origins,
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
    )
    results["operator_day_ahead"] = op
    print(
        f"  {'operator_day_ahead':<22} MASE {op['mase']:.3f}  sMAPE {op['smape']:.2f}%  "
        f"MAE {op['mae']:,.0f}  ({int(op['windows'])} windows, "
        f"{int(op['skipped_windows'])} skipped)"
    )

    models: list = [DailySeasonalNaive()]
    if args.chronos:
        from bellwether.forecast.chronos import ChronosBolt

        models.append(ChronosBolt())

    for model in models:
        started = time.perf_counter()
        result = rolling_origin_backtest(
            model,
            actual,
            series_id=series_id,
            horizon=HORIZON,
            season_length=SEASON_LENGTH,
            origins=origins,
        )
        summary = result.summary()
        summary["windows"] = result.n_windows
        results[model.name] = summary
        print(
            f"  {model.name:<22} MASE {summary['mase']:.3f}  sMAPE {summary['smape']:.2f}%  "
            f"MAE {summary['mae']:,.0f}  cov {summary['coverage_80']:.1%}  "
            f"({time.perf_counter() - started:.0f}s)"
        )

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing[series_id] = results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
