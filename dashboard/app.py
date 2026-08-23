"""Bellwether: what a foundation model does and does not do to electricity demand forecasting.

A walkthrough rather than an explorer. It reads top to bottom and argues a case, because
the interesting thing about this project is not any single metric but the sequence of
claims that were made, tested, and in two cases withdrawn.

Everything here is read from committed files. No EIA call, no DuckDB, no model on the
serving path.

**This is the development renderer, not the published one.** The Space is static HTML built
by `scripts/build_static_space.py`, because Hugging Face charges for the container a
Streamlit Space now needs. Both read their prose from `content.py` and their charts from
`viz.py`, so what you see here is what ships. Run it with `streamlit run dashboard/app.py`.
"""

from __future__ import annotations

import content as text
import loaders as data
import pandas as pd
import streamlit as st
import viz

st.set_page_config(page_title=text.PAGE_TITLE, page_icon="⚡", layout="wide")
viz.enable_theme()


def table_view(frame: pd.DataFrame, label: str = text.TABLE_VIEW) -> None:
    """Every chart ships one.

    Not decoration. The palette's aqua slot sits at 2.74:1 against this surface, under the
    3:1 bar, and the documented relief for that is visible labels or a reachable table. It
    also means no value in this app is only obtainable by hovering.
    """
    with st.expander(label):
        st.dataframe(frame, width="stretch", hide_index=True)


def market_control(slot: str) -> str:
    """The market selector, rendered beside each section it scopes rather than once at the top.

    Two copies and one source of truth: `st.session_state["market"]` holds the choice, and
    each copy mirrors it in before instantiating, so moving either moves both.

    It used to render once above section 1. That was a bug rather than a layout: the first
    thing beneath it is a cross-market chart that deliberately shows all three markets, so
    the control changed nothing a reader could see and read as broken.
    """
    key = f"market_{slot}"
    st.session_state[key] = st.session_state["market"]

    def _adopt() -> None:
        st.session_state["market"] = st.session_state[key]

    st.radio(
        text.MARKET_PROMPT,
        options=list(data.MARKETS),
        horizontal=True,
        captions=[text.MARKET_CAPTIONS[m] for m in data.MARKETS],
        key=key,
        on_change=_adopt,
    )
    return st.session_state["market"]


st.title(text.TITLE)
st.markdown(text.LEDE)

if not data.snapshot_available():
    st.warning(text.NO_SNAPSHOT, icon="⚠️")

backtest = data.results("backtest_results.json")
operator = data.results("operator_comparison.json")
holiday = data.results("holiday_arm.json")
info = data.manifest()

st.caption(text.SNAPSHOT_CAPTION.format(generated=info.get("generated", "not yet")))

# The choice lives here rather than in a widget, because two widgets now read it. Only
# sections 3 and 7 are market-scoped; every other section is a cross-market comparison and
# ignores this entirely.
st.session_state.setdefault("market", list(data.MARKETS)[0])

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S1_HEADING)

skill = data.skill_frame()

left, right = st.columns([3, 2])
with left:
    if not skill.empty:
        st.altair_chart(viz.skill_bars(skill), width="stretch")
with right:
    st.markdown(text.S1_PROSE)
if not skill.empty:
    table_view(skill.round(3), text.S1_TABLE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S2_HEADING)

operator_frame = data.operator_frame()
if not operator_frame.empty:
    st.dataframe(operator_frame, width="stretch", hide_index=True)

st.markdown(text.S2_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S3_HEADING.format(market=st.session_state["market"]))

market = market_control("s3")

observed = data.observations(market)
window = data.forecasts(market, "scale")

if not window.empty:
    origins = sorted(window["origin"].unique())
    labels = data.origin_labels(window)
    chosen = st.select_slider(
        text.S3_SLIDER,
        options=origins,
        value=origins[len(origins) // 2],
        format_func=lambda o: labels[int(o)],
    )
    slice_ = window[window["origin"] == chosen]
    st.altair_chart(
        viz.forecast_window(
            observed,
            slice_,
            text.S3_TITLE.format(market=market, date=labels[int(chosen)]),
        ),
        width="stretch",
    )
    st.caption(text.S3_CAPTION)
    table_view(
        slice_[["period", "q10", "q50", "q90"]].round({"q10": 0, "q50": 0, "q90": 0}),
        text.S3_TABLE,
    )
else:
    st.info(text.S3_EMPTY)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S4_HEADING)

cw = data.coverage_width_frame()
if not cw.empty:
    st.altair_chart(viz.coverage_and_width(cw), width="stretch")
    table_view(cw.round(2), text.S4_TABLE)

st.markdown(text.S4_PROSE)

st.markdown(text.S4_MODELS_PROSE)

models = data.model_comparison_frame()
if not models.empty:
    st.altair_chart(viz.coverage_and_width(models), width="stretch")
    table_view(models.round(2), text.S4_MODELS_TABLE)

st.markdown(text.S4_MODELS_CLOSING)

st.markdown(text.S4_SMALL_LEDE)

retention = data.retention_frame()
if not retention.empty:
    st.dataframe(retention, width="stretch", hide_index=True)

st.markdown(text.S4_SMALL_PROSE)

st.markdown(text.S4_CONTEXT_LEDE)

context = data.context_frame()
if not context.empty:
    st.dataframe(context, width="stretch", hide_index=True)

st.markdown(text.S4_CONTEXT_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S5_HEADING)

tab_hour, tab_month = st.tabs([text.S5_HOUR_TAB, text.S5_MONTH_TAB])
with tab_hour:
    frame = data.profile_frame("mae", "by_local_hour")
    if not frame.empty:
        x_title, y_title = text.S5_HOUR_AXES
        st.altair_chart(viz.profile(frame, "bucket", x_title, "value", y_title), width="stretch")
        table_view(frame.round(3), text.S5_HOUR_TABLE)
    st.markdown(text.S5_HOUR_PROSE)
with tab_month:
    frame = data.profile_frame("coverage", "by_month")
    if not frame.empty:
        x_title, y_title = text.S5_MONTH_AXES
        st.altair_chart(viz.profile(frame, "bucket", x_title, "value", y_title), width="stretch")
        table_view(frame.round(3), text.S5_MONTH_TABLE)
    st.markdown(text.S5_MONTH_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S6_HEADING)

st.markdown(text.S6_LEDE)

diagnosis = data.diagnosis_frame()
if not diagnosis.empty:
    st.altair_chart(viz.attribution_bars(diagnosis), width="stretch")
    table_view(diagnosis.round(1), text.S6_TABLE)

market = market_control("s6")

episodes = data.episode_frame(market)
if not episodes.empty:
    st.dataframe(episodes, width="stretch", hide_index=True)

st.markdown(text.S6_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S7_HEADING)

st.markdown(text.S7_LEDE)

weather = data.weather_frame()
if not weather.empty:
    st.dataframe(weather, width="stretch", hide_index=True)

st.markdown(text.S7_PROSE)

st.markdown(text.S7_FORECAST_LEDE)

forecast_arms = data.forecast_frame()
if not forecast_arms.empty:
    st.dataframe(forecast_arms, width="stretch", hide_index=True)

st.markdown(text.S7_FORECAST_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S8_HEADING)

offsets = data.offsets_frame()
if not offsets.empty:
    st.altair_chart(viz.learned_offsets(offsets), width="stretch")
    table_view(offsets.round(0), text.S8_OFFSETS_TABLE)

st.markdown(text.S8_PROSE)

market = market_control("s8")

per_holiday = data.per_holiday_frame(market)
if not per_holiday.empty:
    st.altair_chart(viz.paired_holidays(per_holiday), width="stretch")
    table_view(per_holiday.round(0), text.S8_HOLIDAY_TABLE.format(market=market))

st.markdown(text.S8_CLOSING)

shape = data.hour_profile_frame()
if not shape.empty:
    x_title, y_title = text.S8_SHAPE_AXES
    st.altair_chart(viz.profile(shape, "hour", x_title, "offset", y_title), width="stretch")
    table_view(shape.round(0), text.S8_SHAPE_TABLE)

st.markdown(text.S8_SHAPE_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S9_HEADING)

st.markdown(text.S9_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.METHOD_HEADING)

with st.expander(text.METHOD_SUMMARY):
    st.markdown(text.METHOD_PROSE)

st.caption(data.attribution())
