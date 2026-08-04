"""Score weather correction three ways: forecast, degraded observation, observed.

Usage: python scripts/run_forecast_ablation.py ERCO [--out docs/forecast_ablation.json]

Every weather number this project has published was measured with observed temperature,
which hands the model perfect knowledge of tomorrow. This run replaces it with the forecast
a forecaster would actually have had, and answers what that costs.

## Three arms, because the obvious two confound

NDFD publishes CONUS temperature every three hours. Everything else here is hourly. So a
forecast arm differs from the published weather arm in two ways at once, forecast error and
temporal resolution, and a comparison between those two attributes both to whichever the
reader already believed.

* **observed**   the published hourly arm, unchanged, as the reference.
* **degraded**   the same observations at the forecast's cadence, interpolated back.
* **forecast**   NDFD, restricted to runs issued at or before each window's origin.

forecast against degraded isolates forecast error. degraded against observed isolates
resolution loss. Reporting forecast against observed alone would be a two-variable
comparison presented as one number, which is the mistake the calendar-only control exists
to prevent and which reading coverage without width once reversed outright.

The calendar-only control runs in all three, and should score identically in each: it uses
no temperature, so a difference between its three copies means the arms were not scored on
one window set, and the run is invalid rather than interesting.

## What to expect

The forecast arm should lose to the observed one. That is the finding, not a failure.
Against observations the forecast runs r = 0.977 at +24 hours with RMSE 1.84 C, so it is a
good forecast and not a perfect one, and finding 7's weather gain was an upper bound
measured with foresight nobody has.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bellwether.eval.ablation import (  # noqa: E402
    cache_base_forecasts,
    run_corrector_ablation,
    usable_origins,
)
from bellwether.eval.operator import BA_TIMEZONES  # noqa: E402
from bellwether.eval.resolution import degrade_to_cadence, shared_coverage  # noqa: E402
from bellwether.forecast.residual import CALENDAR_ONLY, WEATHER  # noqa: E402
from bellwether.storage.db import connect  # noqa: E402
from bellwether.storage.queries import (  # noqa: E402
    load_market_forecast_temperature,
    load_market_temperature,
    load_series,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# Only the two specs this experiment is about. The volatility arms answer a different
# question and would triple a run that is already three runs.
SPECS = (CALENDAR_ONLY, WEATHER)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--series-type", default="D")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--out", default="docs/forecast_ablation.json")
    args = parser.parse_args()

    with connect(read_only=True) as conn:
        series = load_series(conn, args.respondent, args.series_type)
        observed = load_market_temperature(conn, args.respondent, series.timestamps)
        forecast = load_market_forecast_temperature(conn, args.respondent, series.timestamps)

    degraded = degrade_to_cadence(observed, series.timestamps)

    # One window set for all three arms, applied before origins are chosen rather than
    # after. Each arm left to itself would drop a different set of days, and the scores
    # would then differ partly by which days each arm was asked about.
    shared = shared_coverage(observed, degraded, forecast)
    variants = {
        "observed": np.where(shared, observed, np.nan),
        "degraded": np.where(shared, degraded, np.nan),
        "forecast": np.where(shared, forecast, np.nan),
    }

    origins = usable_origins(
        series.values, variants["observed"], args.horizon, initial_train_size=672
    )
    print(
        f"{series.series_id}: {series.values.size:,} hours, "
        f"{int(shared.sum()):,} with all three temperature series, {len(origins)} usable origins"
    )
    if not origins:
        print(
            "\nNo usable origins. Every hour of a window needs a forecast, and a window\n"
            "opening at 00:00 UTC is served by the previous day's 12:00 UTC run, so it\n"
            "spans leads 12 to 36. If the ingest was run with a lower --max-step, re-run:\n"
            "  python scripts/ingest_ndfd.py --start 2024-07-31 --end 2026-07-31 "
            "--max-step 36 --skip-stored"
        )
        return 1

    from bellwether.forecast.chronos import ChronosBolt

    base = ChronosBolt()
    started = time.perf_counter()
    cached = cache_base_forecasts(base, series.values, origins, args.horizon)
    print(f"  base forecasts cached in {time.perf_counter() - started:.0f}s")

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    market = existing.setdefault(series.series_id, {})

    for label, temperature in variants.items():
        started = time.perf_counter()
        output = run_corrector_ablation(
            base,
            series.values,
            series.timestamps,
            temperature,
            series_id=series.series_id,
            timezone=BA_TIMEZONES[args.respondent],
            horizon=args.horizon,
            specs=SPECS,
            cached=cached,
        )
        elapsed = time.perf_counter() - started
        print(f"  {label} ({elapsed:.0f}s)")
        for name, result in output.results.items():
            summary = result.summary()
            summary["windows"] = result.n_windows
            market[f"{label}:{name}"] = summary
            print(
                f"    {name:<28} MASE {summary['mase']:.3f}  WQL {summary['wql']:.4f}  "
                f"sMAPE {summary['smape']:.2f}%  cov {summary['coverage_80']:.1%}"
            )

    market["_meta"] = {
        "origins": len(origins),
        "shared_hours": int(shared.sum()),
        "cadence_hours": 3,
        "arms": list(variants),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    _check_control(market)
    return 0


def _check_control(market: dict) -> None:
    """The calendar arm uses no temperature, so its three copies must agree exactly.

    They are scored in three separate passes over three temperature series. If those passes
    ever saw different windows, this is where it shows, and it costs one comparison rather
    than a retracted finding.
    """
    scores = {
        label: market[key]["smape"]
        for label in ("observed", "degraded", "forecast")
        for key in [f"{label}:chronos_bolt_base+calendar"]
        if key in market
    }
    if len(set(round(v, 9) for v in scores.values())) > 1:
        print("\nWARNING: the calendar control differs across arms, so they were not scored")
        print("on one window set. The comparison is invalid.")
        for label, value in scores.items():
            print(f"  {label}: sMAPE {value:.6f}")


if __name__ == "__main__":
    sys.exit(main())
