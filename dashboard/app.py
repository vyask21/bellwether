"""Bellwether: what a foundation model does and does not do to electricity demand forecasting.

A walkthrough rather than an explorer. It reads top to bottom and argues a case, because
the interesting thing about this project is not any single metric but the sequence of
claims that were made, tested, and in two cases withdrawn.

Everything here is read from committed files. No EIA call, no DuckDB, no model on the
serving path.
"""

from __future__ import annotations

import loaders as data
import pandas as pd
import streamlit as st
import viz

st.set_page_config(
    page_title="Bellwether: forecasting electricity demand",
    page_icon="⚡",
    layout="wide",
)
viz.enable_theme()


def table_view(frame: pd.DataFrame, label: str = "Table view") -> None:
    """Every chart ships one.

    Not decoration. The palette's aqua slot sits at 2.74:1 against this surface, under the
    3:1 bar, and the documented relief for that is visible labels or a reachable table. It
    also means no value in this app is only obtainable by hovering.
    """
    with st.expander(label):
        st.dataframe(frame, width="stretch", hide_index=True)


st.title("Bellwether")
st.markdown(
    "**Does a time-series foundation model beat the people who run the grid?** "
    "Three US balancing authorities, two years of hourly demand, everything measured "
    "against controls rather than against nothing."
)

if not data.snapshot_available():
    st.warning(
        "No snapshot found. Charts drawn from stored forecasts will be empty until "
        "`python scripts/export_snapshot.py <market>` has been run for each market.",
        icon="⚠️",
    )

backtest = data.results("backtest_results.json")
operator = data.results("operator_comparison.json")
holiday = data.results("holiday_arm.json")
info = data.manifest()

st.caption(
    f"Snapshot generated {info.get('generated', 'not yet')}. "
    "Three balancing authorities out of 83, chosen to contrast rather than to sample."
)

# One filter row, above everything it scopes. The market-specific sections below read it;
# the cross-market comparisons deliberately ignore it and show all three.
market = st.radio(
    "Market for the single-market sections",
    options=list(data.MARKETS),
    horizontal=True,
    captions=["California ISO", "ERCOT, Texas", "PacifiCorp East, Utah"],
)

st.divider()

# ----------------------------------------------------------------------------------------
st.header("1. The model beats the statistical baselines everywhere")

skill_rows = []
for series_id, arms in backtest.items():
    name = series_id.split(":")[0]
    model = arms.get("chronos_bolt_base")
    baselines = {k: v for k, v in arms.items() if k.startswith("seasonal_naive")}
    if not model or not baselines:
        continue
    best = min(baselines.items(), key=lambda kv: kv[1]["smape"])
    skill_rows.append(
        {
            "market": name,
            "reduction": (1 - model["smape"] / best[1]["smape"]) * 100,
            "baseline": data.ARM_LABELS.get(best[0], best[0]),
            "mase": model["mase"],
        }
    )
skill = pd.DataFrame(skill_rows)

left, right = st.columns([3, 2])
with left:
    if not skill.empty:
        st.altair_chart(viz.skill_bars(skill), width="stretch")
with right:
    st.markdown(
        "Zero-shot, with no weather input and no training on these series. "
        "MASE below 1.0 means it beat the naive baseline on its own scale.\n\n"
        "**The daily seasonal lag beats the weekly one in every market**, which contradicts "
        "the textbook expectation for electricity load. Weather persistence is the likely "
        "reason: yesterday resembles today because yesterday's weather does."
    )
if not skill.empty:
    table_view(skill.round(3), "Table view: skill against baselines")

st.divider()

# ----------------------------------------------------------------------------------------
st.header("2. Against the operator's own forecast, it splits")

op_rows = []
for series_id, arms in operator.items():
    name = series_id.split(":")[0]
    if "operator_day_ahead" not in arms or "chronos_bolt_base" not in arms:
        continue
    for arm, metrics in arms.items():
        op_rows.append(
            {
                "market": name,
                "arm": data.ARM_LABELS.get(arm, arm),
                "sMAPE (%)": round(metrics["smape"], 3),
                "MASE": round(metrics["mase"], 3),
            }
        )
if op_rows:
    st.dataframe(pd.DataFrame(op_rows), width="stretch", hide_index=True)

st.markdown(
    "**ERCOT beats us by 31%. PacifiCorp East does not beat us, or even seasonal-naive.** "
    "That reads as forecasting investment rather than method: ERCOT runs a large, "
    "weather-driven market and forecasts it accordingly.\n\n"
    "CISO is absent on purpose. Its `DF` series diverges from its own `D` series at midday "
    "in a way nobody here has explained, and an operator baseline nobody understands is "
    "worse than no operator baseline."
)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(f"3. A forecast, drawn. {market}")

observed = data.observations(market)
window = data.forecasts(market, "scale")

if not window.empty:
    origins = sorted(window["origin"].unique())
    labels = {
        int(o): pd.Timestamp(window.loc[window["origin"] == o, "period"].iloc[0]).strftime(
            "%Y-%m-%d"
        )
        for o in origins
    }
    chosen = st.select_slider(
        "Forecast origin",
        options=origins,
        value=origins[len(origins) // 2],
        format_func=lambda o: labels[int(o)],
    )
    slice_ = window[window["origin"] == chosen]
    st.altair_chart(
        viz.forecast_window(
            observed, slice_, f"{market}: 24 hours ahead from {labels[int(chosen)]}"
        ),
        width="stretch",
    )
    st.caption(
        "Blue is what happened. Orange is the forecast median, and the shaded band is the "
        "80% interval, which should contain the actual on 4 days in 5."
    )
    table_view(
        slice_[["period", "q10", "q50", "q90"]].round({"q10": 0, "q50": 0, "q90": 0}),
        "Table view: this forecast window",
    )
else:
    st.info("Run the snapshot export to draw forecasts.")

st.divider()

# ----------------------------------------------------------------------------------------
st.header("4. Never report coverage without width")

cw = data.coverage_width_frame()
if not cw.empty:
    st.altair_chart(viz.coverage_and_width(cw), width="stretch")
    table_view(cw.round(2), "Table view: coverage and width")

st.markdown(
    "A model can buy coverage by widening its interval and sharpen its way back out again, "
    "and the two are indistinguishable in a coverage column. This is not hypothetical: it "
    "is what separated the calendar arm from the weather arm below, and reading coverage "
    "alone reversed the conclusion.\n\n"
    "**The base model's defect is a level error, not a conditioning one.** Chronos-Bolt "
    "conditions its intervals well, widening them 43 to 83% between its narrowest and "
    "widest month. It simply runs a few points under nominal. Scaling its own spread about "
    "its own median fixes the level and keeps the conditioning; rebuilding the distribution "
    "from residual quantiles fixes the average by flattening the seasons, which is worse "
    "for anyone acting on it."
)

st.divider()

# ----------------------------------------------------------------------------------------
st.header("5. Where the forecast actually fails")

tab_hour, tab_month = st.tabs(["By hour of day", "By month"])
with tab_hour:
    frame = data.profile_frame("mae", "by_local_hour")
    if not frame.empty:
        st.altair_chart(
            viz.profile(frame, "bucket", "Local hour", "value", "MAE (% of mean demand)"),
            width="stretch",
        )
        table_view(frame.round(3), "Table view: error by local hour")
    st.markdown(
        "Peak error lands at 17:00 local in ERCO and 15:00 in CISO, the evening ramp in "
        "both cases. Aggregate metrics hide this completely.\n\n"
        "**A trap worth naming.** Hour of day and horizon step are the same variable in a "
        "single backtest run, because origins advance by exactly the horizon, so every hour "
        "is always forecast at the same lead time. The first version of this analysis "
        "reported a diurnal profile that was line for line a horizon-step profile, looked "
        "entirely reasonable, and named the wrong hours. These numbers pool four staggered "
        "origin sets so the two cross."
    )
with tab_month:
    frame = data.profile_frame("coverage", "by_month")
    if not frame.empty:
        st.altair_chart(
            viz.profile(frame, "bucket", "Month", "value", "80% interval coverage (%)"),
            width="stretch",
        )
        table_view(frame.round(3), "Table view: coverage by month")
    st.markdown(
        "**Three markets, three different bad seasons.** ERCO is worst in January and "
        "November, CISO in March and May, PACE in June and November. No single calendar "
        "rule helps all three: widening winter would target ERCO's worst season and PACE's "
        "second best."
    )

st.divider()

# ----------------------------------------------------------------------------------------
st.header("6. Temperature buys accuracy and does not fix calibration")

st.markdown(
    "Against a **calendar-only control**, not against the raw model. That control is what "
    "makes the number mean anything: rebuilding a predictive distribution from residual "
    "quantiles is itself a recalibration, so a weather corrector scored only against the "
    "base model collects credit for work that has nothing to do with weather."
)

ablation = data.results("weather_ablation.json")
wx_rows = []
for series_id, arms in ablation.items():
    name = series_id.split(":")[0]
    cal, wx = arms.get(f"{data.BASE_ARM}+calendar"), arms.get(f"{data.BASE_ARM}+weather")
    if not cal or not wx:
        continue
    wx_rows.append(
        {
            "Market": name,
            "sMAPE change (%)": round((wx["smape"] / cal["smape"] - 1) * 100, 1),
            "WQL change (%)": round((wx["wql"] / cal["wql"] - 1) * 100, 1),
            "Coverage, calendar (%)": round(cal["coverage_80"] * 100, 1),
            "Coverage, weather (%)": round(wx["coverage_80"] * 100, 1),
        }
    )
if wx_rows:
    st.dataframe(pd.DataFrame(wx_rows), width="stretch", hide_index=True)

st.markdown(
    "**ERCOT is the most temperature-coupled market of the three by a wide margin**, and "
    "that was measured before any model was fitted: summer correlation 0.925, winter "
    "-0.694, against CISO's 0.559 and *positive* 0.254. It predicted the ordering above "
    "correctly.\n\n"
    "California's demand barely tracks its own weather and its winter correlation has the "
    "wrong sign. Gas heating plus behind-the-meter solar, which is the duck curve restated."
)

st.divider()

# ----------------------------------------------------------------------------------------
st.header("7. Holidays: the effect is real and it is confined to six days")

offsets = []
for series_id, entry in holiday.items():
    name = series_id.split(":")[0]
    for label, value in entry.get("learned_offsets", {}).items():
        if label == "pooled":
            continue
        offsets.append({"market": name, "observance": label, "offset": value})
if offsets:
    st.altair_chart(viz.learned_offsets(pd.DataFrame(offsets)), width="stretch")
    table_view(pd.DataFrame(offsets).round(0), "Table view: learned offsets")

st.markdown(
    "One offset applied to all eleven federal holidays improved 28 of 33 widely-observed "
    "holidays and only 10 of 27 federal-only ones. Below a coin flip on the second group is "
    "the signature of a correction being applied where nothing needs correcting.\n\n"
    "Splitting the offset by whether private employers actually close confirms it. **In "
    "ERCO and PACE the federal-only offset does not shrink, it changes sign.** Demand on "
    "Veterans Day in Texas is not depressed at all, and the pooled version had been pushing "
    "those days down because Christmas dragged the shared estimate with it."
)

entry = holiday.get(f"{market}:D", {})
rows = []
for record in entry.get("per_holiday", []):
    rows.append(
        {
            "date": record["date"],
            "name": record["date"],
            "observance": record["observance"],
            "change": record["change_vs_scale"],
            "direction": "better" if record["change_vs_scale"] < 0 else "worse",
        }
    )
if rows:
    st.altair_chart(viz.paired_holidays(pd.DataFrame(rows)), width="stretch")
    table_view(pd.DataFrame(rows).round(0), f"Table view: {market} per-holiday change")

st.markdown(
    "**It was measured twice and shipped neither time.** Against no calendar at all only "
    "CISO shows a market-level win, worth 17.6% of its holiday error, and corrected for "
    "three markets that does not clear the bar. The honest reason is not caution: it is "
    "that a single scalar shift over 24 hours is the wrong shape. Load barely moves "
    "overnight and falls hard through the working day, so one number over-corrects the "
    "small hours to reach the large ones."
)

st.divider()

# ----------------------------------------------------------------------------------------
st.header("8. Two published claims that turned out to be wrong")

st.markdown(
    "Both were caught by re-measurement rather than by review, and both are kept here "
    "because a results page that reports only what survived is not reporting.\n\n"
    '**"Chronos-Bolt\'s interval is unconditional."** It is not. That measurement was taken '
    "on a residual corrector, which rebuilds the predictive distribution from scratch, so "
    "every distributional property it showed belonged to the corrector. The base model "
    "conditions its intervals well. The rule that came out of it: profile the base model, "
    "not only the arm you happen to be using.\n\n"
    '**"The duck curve shows up as a breach pattern in CISO."** Below-bound breaches at '
    "the 10:00 to 11:00 solar ramp were an artifact of the same corrector. Peak *error* by "
    "hour survived re-measurement; worst *coverage* by hour did not.\n\n"
    "A third prediction was pre-registered and falsified cleanly: a volatility-conditioned "
    "interval was supposed to fix the seasonal miscoverage. It did not, and its "
    "discriminating check passed, so the target turned out to be an artifact of the method "
    "rather than a property of the grid."
)

st.divider()

# ----------------------------------------------------------------------------------------
st.header("Method and sources")

with st.expander("Scope, and what these numbers are not"):
    st.markdown(
        "- **Three balancing authorities out of 83**, chosen to contrast rather than to "
        "sample. No state is a unit in this data.\n"
        "- **Weather is observed, not forecast.** Every weather number here was measured "
        "with perfect foresight and is therefore a ceiling, not an operational result.\n"
        "- NCEI's archive ends about eleven months before EIA's data does, so weather work "
        "covers roughly half the demand grid. That limit was accepted rather than patched "
        "with a third-party source.\n"
        "- Rolling-origin backtest, 24 hour horizon, scored on MASE, WQL, sMAPE, and 80% "
        "interval coverage with width."
    )

notice = info.get("attribution", {})
st.caption(
    " · ".join(
        v
        for v in (
            notice.get("observations_eia"),
            notice.get("observations_noaa"),
            notice.get("forecasts"),
        )
        if v
    )
    or "Source: U.S. Energy Information Administration. Forecasts are produced by this "
    "project, not by EIA, and are not authoritative."
)
