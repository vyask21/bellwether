"""Reading the committed results and snapshot.

Everything here is a file read. There is no DuckDB, no EIA call, and no model on the
serving path: a hosted Space cannot reach any of them, and a dashboard that needs a
network call to draw a chart it already computed is a dashboard that breaks in public.

The snapshot is a **cache, not a source of truth**. If it disagrees with the DuckDB store,
the store is right and the snapshot is stale. `manifest.json` carries the date it was
generated so a viewer can tell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

MARKETS = ("CISO", "ERCO", "PACE")
BASE_ARM = "chronos_bolt_base"

# Display names. The arm identifiers are precise and unreadable, and a reader who has to
# decode "chronos_bolt_base+scale+holidayclass" is not reading the finding.
ARM_LABELS = {
    "chronos_bolt_base": "Chronos-Bolt",
    "chronos_bolt_base+scale": "+ scale",
    "chronos_bolt_base+calendar": "+ calendar",
    "chronos_bolt_base+weather": "+ weather",
    "chronos_bolt_base+volatility": "+ volatility",
    "chronos_bolt_base+weather+volatility": "+ weather + volatility",
    "chronos_bolt_base+scale+holiday": "+ holiday (pooled)",
    "chronos_bolt_base+scale+holidayclass": "+ holiday (by class)",
    "seasonal_naive_daily": "Seasonal naive, daily",
    "seasonal_naive_weekly": "Seasonal naive, weekly",
    "operator_day_ahead": "Operator day-ahead",
}


def root() -> Path:
    """The repo root, or the Space root, whichever this is running from."""
    here = Path(__file__).resolve().parent
    return here.parent if (here.parent / "docs").is_dir() else here


@st.cache_data(show_spinner=False)
def results(name: str) -> dict:
    path = root() / "docs" / name
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def manifest() -> dict:
    path = root() / "snapshot" / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def observations(market: str) -> pd.DataFrame:
    path = root() / "snapshot" / f"demand_{market}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["period", "demand_mw", "temperature_c"])
    return pd.read_parquet(path)


# The snapshot stores short arm labels rather than the full identifiers, because the label
# repeats on all 61,632 rows of every market. Naming them here keeps the two files from
# drifting apart silently, which is exactly how a chart ends up drawing nothing.
SNAPSHOT_ARMS = ("base", "scale", "holiday", "holidayclass")


@st.cache_data(show_spinner=False)
def forecasts(market: str, arm: str) -> pd.DataFrame:
    """One arm's stored forecasts. `arm` is a snapshot label, not a full identifier."""
    if arm not in SNAPSHOT_ARMS:
        raise ValueError(f"Unknown snapshot arm {arm!r}, expected one of {SNAPSHOT_ARMS}")
    path = root() / "snapshot" / f"forecasts_{market}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["arm", "origin", "step", "period", "q10", "q50", "q90"])
    frame = pd.read_parquet(path)
    return frame[frame["arm"] == arm].copy()


@st.cache_data(show_spinner=False)
def snapshot_available() -> bool:
    return any((root() / "snapshot").glob("forecasts_*.parquet"))


@st.cache_data(show_spinner=False)
def mean_demand() -> dict[str, float]:
    """Mean demand per market, the denominator every cross-market chart normalises by.

    Falls back to the recorded backtest MAE and MASE if no snapshot is present, so the
    comparative charts still draw on a fresh clone before the export has been run.
    """
    means = {}
    for market in MARKETS:
        frame = observations(market)
        if not frame.empty:
            means[market] = float(frame["demand_mw"].mean())
    return means


def coverage_width_frame() -> pd.DataFrame:
    """Coverage and normalised interval width per market and arm, from the weather run.

    Width is divided by mean demand because ERCO's band is 7,000 MW and PACE's is 200, and
    an unnormalised chart would report only that Texas is larger than Utah.
    """
    ablation, means = results("weather_ablation.json"), mean_demand()
    rows = []
    for series_id, arms in ablation.items():
        market = series_id.split(":")[0]
        for arm, metrics in arms.items():
            if arm.startswith("_") or "coverage_80" not in metrics:
                continue
            width = metrics.get("width_80")
            if width is None or market not in means:
                continue
            rows.append(
                {
                    "market": market,
                    "arm": ARM_LABELS.get(arm, arm),
                    "coverage": metrics["coverage_80"] * 100,
                    "width_pct": width / means[market] * 100,
                }
            )
    return pd.DataFrame(rows)


def profile_frame(field: str, axis: str, arm: str = f"{BASE_ARM}+scale") -> pd.DataFrame:
    """One breach-analysis profile across all markets, normalised where it is an error.

    Defaults to the `+scale` arm deliberately. Every distributional claim measured on a
    residual corrector is a claim about that corrector, and publishing one as a property of
    Chronos-Bolt is a mistake this project has already made once.
    """
    breaches, means = results("breach_analysis.json"), mean_demand()
    rows = []
    for series_id, arms in breaches.items():
        market = series_id.split(":")[0]
        entry = arms.get(arm) or arms.get(BASE_ARM)
        if not entry or axis not in entry:
            continue
        for bucket in entry[axis]:
            value = bucket[field]
            if field == "mae" and market in means:
                value = value / means[market] * 100
            elif field == "coverage":
                value = value * 100
            rows.append({"market": market, "bucket": bucket[_key(axis)], "value": value})
    return pd.DataFrame(rows)


def _key(axis: str) -> str:
    return {"by_local_hour": "hour", "by_month": "month", "by_horizon_step": "step"}[axis]
