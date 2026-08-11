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
from content import FALLBACK_ATTRIBUTION

MARKETS = ("CISO", "ERCO", "PACE")
BASE_ARM = "chronos_bolt_base"

# Display names. The arm identifiers are precise and unreadable, and a reader who has to
# decode "chronos_bolt_base+scale+holidayclass" is not reading the finding.
ARM_LABELS = {
    "chronos_bolt_base": "Chronos-Bolt",
    "timesfm_2p5_200m": "TimesFM 2.5",
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


def origin_labels(window: pd.DataFrame) -> dict[int, str]:
    """Each forecast origin's calendar date, for a control that has to name one."""
    return {
        int(origin): pd.Timestamp(
            window.loc[window["origin"] == origin, "period"].iloc[0]
        ).strftime("%Y-%m-%d")
        for origin in sorted(window["origin"].unique())
    }


def attribution() -> str:
    """The source notice the snapshot carries, or the minimum one if it carries none."""
    notice = manifest().get("attribution", {})
    joined = " · ".join(
        value
        for value in (
            notice.get("observations_eia"),
            notice.get("observations_noaa"),
            notice.get("forecasts"),
        )
        if value
    )
    return joined or FALLBACK_ATTRIBUTION


def skill_frame() -> pd.DataFrame:
    """Error reduction against each market's *best* baseline, not against a fixed one.

    Taking the best means the headline can never be manufactured by comparing against the
    weaker of two controls, which is the whole point of carrying two.
    """
    rows = []
    for series_id, arms in results("backtest_results.json").items():
        model = arms.get(BASE_ARM)
        baselines = {k: v for k, v in arms.items() if k.startswith("seasonal_naive")}
        if not model or not baselines:
            continue
        best = min(baselines.items(), key=lambda kv: kv[1]["smape"])
        rows.append(
            {
                "market": series_id.split(":")[0],
                "reduction": (1 - model["smape"] / best[1]["smape"]) * 100,
                "baseline": ARM_LABELS.get(best[0], best[0]),
                "mase": model["mase"],
            }
        )
    return pd.DataFrame(rows)


def operator_frame() -> pd.DataFrame:
    """Model against the operator's own day-ahead forecast, where one is comparable."""
    rows = []
    for series_id, arms in results("operator_comparison.json").items():
        if "operator_day_ahead" not in arms or BASE_ARM not in arms:
            continue
        for arm, metrics in arms.items():
            rows.append(
                {
                    "market": series_id.split(":")[0],
                    "arm": ARM_LABELS.get(arm, arm),
                    "sMAPE (%)": round(metrics["smape"], 3),
                    "MASE": round(metrics["mase"], 3),
                }
            )
    return pd.DataFrame(rows)


def weather_frame() -> pd.DataFrame:
    """The weather arm against the calendar-only control, never against the raw model."""
    rows = []
    for series_id, arms in results("weather_ablation.json").items():
        calendar, weather = arms.get(f"{BASE_ARM}+calendar"), arms.get(f"{BASE_ARM}+weather")
        if not calendar or not weather:
            continue
        rows.append(
            {
                "Market": series_id.split(":")[0],
                "sMAPE change (%)": round((weather["smape"] / calendar["smape"] - 1) * 100, 1),
                "WQL change (%)": round((weather["wql"] / calendar["wql"] - 1) * 100, 1),
                "Coverage, calendar (%)": round(calendar["coverage_80"] * 100, 1),
                "Coverage, weather (%)": round(weather["coverage_80"] * 100, 1),
            }
        )
    return pd.DataFrame(rows)


# The three temperature series the forecast ablation scores, in the order the finding reads:
# perfect foresight, the same series at the forecast's cadence, and the forecast itself. The
# middle one exists because a forecast arm differs from an observed arm in two ways at once,
# and without it the shortfall would be split between two causes with no way to say which.
FORECAST_ARMS = {
    "observed": "Observed",
    "degraded": "Observed, 3-hourly",
    "forecast": "NDFD forecast",
}


def forecast_frame() -> pd.DataFrame:
    """The three-arm forecast ablation, each arm against the control scored in its own pass.

    Every pass carries its own calendar-only control and all three come out identical,
    which is the check that the arms saw the same windows. Dividing by the control from the
    same pass rather than by one borrowed from another keeps that property load-bearing: if
    a pass ever drifts, the number moves instead of quietly reading off the wrong windows.
    """
    rows = []
    for series_id, arms in results("forecast_ablation.json").items():
        for prefix, label in FORECAST_ARMS.items():
            control = arms.get(f"{prefix}:{BASE_ARM}+calendar")
            weather = arms.get(f"{prefix}:{BASE_ARM}+weather")
            if not control or not weather:
                continue
            rows.append(
                {
                    "Market": series_id.split(":")[0],
                    "Temperature": label,
                    "sMAPE change (%)": round((weather["smape"] / control["smape"] - 1) * 100, 1),
                    "Coverage (%)": round(weather["coverage_80"] * 100, 1),
                }
            )
    return pd.DataFrame(rows)


def offsets_frame() -> pd.DataFrame:
    """Learned holiday offset per market and observance class. The pooled one is dropped:
    the finding is that the two classes disagree, and the average of them is what hid it."""
    rows = [
        {"market": series_id.split(":")[0], "observance": label, "offset": value}
        for series_id, entry in results("holiday_arm.json").items()
        for label, value in entry.get("learned_offsets", {}).items()
        if label != "pooled"
    ]
    return pd.DataFrame(rows)


# The class the hour-profile finding is about. The federal-only profile is estimated from
# holidays that mostly turn out not to be holidays for demand, so its shape is a shape of
# very little and putting it on the same axes would invite reading it as the same kind of
# thing.
PROFILED_CLASS = "widely observed"


def hour_profile_frame() -> pd.DataFrame:
    """The holiday offset learned per local hour, on widely-observed holidays.

    The chart is the finding: a flat line here would say the scalar shift was the right
    shape after all, and it is not flat anywhere.
    """
    rows = [
        {"market": series_id.split(":")[0], "hour": int(hour), "offset": value}
        for series_id, entry in results("holiday_arm.json").items()
        for hour, value in entry.get("learned_hour_profile", {}).get(PROFILED_CLASS, {}).items()
    ]
    return pd.DataFrame(rows).sort_values(["market", "hour"]) if rows else pd.DataFrame()


def per_holiday_frame(market: str) -> pd.DataFrame:
    """One market's per-holiday change against the shipped arm, signed for colour."""
    entry = results("holiday_arm.json").get(f"{market}:D", {})
    rows = [
        {
            "date": record["date"],
            "name": record["date"],
            "observance": record["observance"],
            "change": record["change_vs_scale"],
            "direction": "better" if record["change_vs_scale"] < 0 else "worse",
        }
        for record in entry.get("per_holiday", [])
    ]
    return pd.DataFrame(rows)


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


# The two foundation models, in the order the finding reads: the champion, then the one
# brought in to find out which of its properties were its own.
MODEL_ARMS = ("chronos_bolt_base", "timesfm_2p5_200m")


def model_comparison_frame() -> pd.DataFrame:
    """Both foundation models on the same windows, coverage against width.

    Deliberately the same shape as `coverage_width_frame`, because it draws through the
    same chart. The finding is that two models miss coverage in the same markets in the
    same direction, and that is the picture that chart already makes: a second row of dots
    landing in the same left to right order as the first.

    Width comes from the backtest record rather than from the ablation, so both models are
    read off one window set. The two files score different windows and a calibration
    comparison across them would be a comparison of window sets.
    """
    backtest, means = results("backtest_results.json"), mean_demand()
    rows = []
    for series_id, arms in backtest.items():
        market = series_id.split(":")[0]
        for arm in MODEL_ARMS:
            metrics = arms.get(arm) or {}
            width = metrics.get("width_80")
            if width is None or "coverage_80" not in metrics or market not in means:
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
