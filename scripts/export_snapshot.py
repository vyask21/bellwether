"""Export the committed Parquet snapshot the dashboard reads.

Usage: python scripts/export_snapshot.py CISO [--out snapshot]

One market per invocation, like every other long script here, so an interrupted run
resumes rather than restarts. Roughly 8 minutes per market: the base model forecasts once
over every origin, then the correctors refit at each of them.

## Why anything is committed at all

`.gitignore` says data artifacts are regenerable and never committed, and that rule still
holds for the DuckDB store. This is a deliberate exception with a narrow reason: a hosted
dashboard cannot reach DuckDB, and without stored forecasts it cannot draw a single
forecast. Every chart would be a bar chart over summary metrics, which is exactly the
material the results documents already carry better.

The snapshot is kept small enough to read at a glance in a diff, and regenerable from this
script, so it is a cache rather than a source of truth. If it disagrees with the DuckDB
store, the store wins and this file is stale.

## The window

The full demand history, with no weather constraint, which is the window the holiday arms
were measured on: 702 usable origins, 642 scored after warmup. Temperature is carried
alongside as a column and is **NaN for roughly half the grid**, because NCEI's archive ends
eleven months before EIA's data does. That is a property of the sources rather than a gap
to fill, and a chart drawing temperature must drop those hours rather than interpolate.

## Attribution

Observations are EIA and NOAA content; the quantiles are this project's output and are not
EIA content. `manifest.json` carries both notices separately, and they are never blended.
See docs/EIA_COMPLIANCE.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bellwether.attribution import (
    DERIVED_DISCLAIMER,
    eia_acknowledgment,
    noaa_acknowledgment,
)
from bellwether.eval.ablation import (
    HOLIDAY_ARM,
    HOLIDAY_CLASS_ARM,
    SCALE_ARM,
    cache_base_forecasts,
    run_corrector_ablation,
    usable_origins,
)
from bellwether.eval.metrics import DEFAULT_QUANTILES, _quantile_index
from bellwether.eval.operator import BA_TIMEZONES
from bellwether.storage.db import connect
from bellwether.storage.queries import load_market_temperature, load_series

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

HORIZON = 24

# The three levels every finding in this project is stated in terms of: the 80% band and
# its median. Storing all nine would triple the file to serve charts nothing draws.
EXPORTED_QUANTILES = (0.1, 0.5, 0.9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("respondent")
    parser.add_argument("--series-type", default="D")
    parser.add_argument("--out", default="snapshot")
    args = parser.parse_args()

    with connect(read_only=True) as conn:
        series = load_series(conn, args.respondent, args.series_type)
        temperature = load_market_temperature(conn, args.respondent, series.timestamps)

    from bellwether.forecast.chronos import ChronosBolt

    base = ChronosBolt()
    # No temperature: the weather arms are excluded, so the weather window does not bind
    # and this matches the window the holiday findings were measured on.
    origins = usable_origins(series.values, None, HORIZON, initial_train_size=672)
    print(f"{series.series_id}: {len(origins)} usable origins on the full demand history")

    cached = cache_base_forecasts(base, series.values, origins, HORIZON)
    output = run_corrector_ablation(
        base,
        series.values,
        series.timestamps,
        None,
        series_id=series.series_id,
        timezone=BA_TIMEZONES[args.respondent],
        specs=(),  # base, +scale, +scale+holiday, +scale+holidayclass
        cached=cached,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = _observations(series, temperature)
    observations.to_parquet(
        out_dir / f"demand_{args.respondent}.parquet", index=False, compression="zstd"
    )

    forecasts = _forecasts(series, output, base.name)
    forecasts.to_parquet(
        out_dir / f"forecasts_{args.respondent}.parquet", index=False, compression="zstd"
    )

    _write_manifest(out_dir, args.respondent, series, output, observations, forecasts)

    print(f"  {len(observations):,} observed hours, {observations.demand_mw.isna().sum()} missing")
    print(f"  {int(observations.temperature_c.notna().sum()):,} hours with temperature")
    print(f"  {len(forecasts):,} forecast rows over {forecasts.arm.nunique()} arms")
    for path in sorted(out_dir.glob(f"*_{args.respondent}.parquet")):
        print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB")


def _observations(series, temperature: np.ndarray) -> pd.DataFrame:
    """Hourly demand and market temperature on one grid.

    Both are kept as float32. Demand runs to five significant figures at most and
    temperature to three, so float64 would double the file to store noise.
    """
    return pd.DataFrame(
        {
            "period": pd.DatetimeIndex(series.timestamps),
            "demand_mw": series.values.astype(np.float32),
            "temperature_c": temperature.astype(np.float32),
        }
    )


def _forecasts(series, output, base_name: str) -> pd.DataFrame:
    """One row per arm, origin and horizon step, carrying the 80% band and the median.

    Long rather than wide, because the dashboard filters by arm far more often than it
    compares arms side by side, and a long frame makes that a mask instead of a reshape.
    """
    arms = {
        "base": base_name,
        "scale": f"{base_name}+{SCALE_ARM}",
        "holiday": f"{base_name}+{HOLIDAY_ARM}",
        "holidayclass": f"{base_name}+{HOLIDAY_CLASS_ARM}",
    }
    columns = {
        f"q{int(level * 100):02d}": _quantile_index(DEFAULT_QUANTILES, level)
        for level in EXPORTED_QUANTILES
    }

    origins = np.array(output.scored_origins)
    index = np.concatenate([np.arange(o, o + HORIZON) for o in origins])
    frames = []
    for label, arm in arms.items():
        stacked = np.concatenate(output.forecasts[arm])
        frame = pd.DataFrame(
            {
                "arm": label,
                "origin": np.repeat(origins, HORIZON).astype(np.int32),
                "step": np.tile(np.arange(1, HORIZON + 1), len(origins)).astype(np.int8),
                "period": pd.DatetimeIndex(series.timestamps[index]),
            }
        )
        for name, position in columns.items():
            frame[name] = stacked[:, position].astype(np.float32)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["arm"] = combined["arm"].astype("category")
    return combined


def _write_manifest(
    out_dir: Path, respondent: str, series, output, observations, forecasts
) -> None:
    """Record what this snapshot is, per market, alongside both agencies' notices.

    Merged rather than overwritten, so exporting one market does not erase the record of
    the other two.
    """
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.setdefault("markets", {})

    retrieved = datetime.now(UTC)
    manifest["generated"] = retrieved.date().isoformat()
    manifest["quantile_levels"] = list(EXPORTED_QUANTILES)
    manifest["horizon_hours"] = HORIZON
    manifest["window"] = "full demand history, no weather constraint"
    manifest["regenerate_with"] = "python scripts/export_snapshot.py <respondent>"
    # Kept apart deliberately: an EIA acknowledgment must never land on a NOAA value or on
    # a forecast, which is model output rather than agency content.
    manifest["attribution"] = {
        "observations_eia": eia_acknowledgment(retrieved),
        "observations_noaa": noaa_acknowledgment(retrieved),
        "forecasts": DERIVED_DISCLAIMER,
    }
    manifest["markets"][respondent] = {
        "series_id": series.series_id,
        "timezone": BA_TIMEZONES[respondent],
        "first_period": str(observations.period.min()),
        "last_period": str(observations.period.max()),
        "observed_hours": int(len(observations)),
        "hours_with_temperature": int(observations.temperature_c.notna().sum()),
        "scored_origins": len(output.scored_origins),
        "forecast_rows": int(len(forecasts)),
        "arms": sorted(forecasts.arm.unique().tolist()),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
