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


st.title(text.TITLE)
st.markdown(text.LEDE)

if not data.snapshot_available():
    st.warning(text.NO_SNAPSHOT, icon="⚠️")

backtest = data.results("backtest_results.json")
operator = data.results("operator_comparison.json")
holiday = data.results("holiday_arm.json")
info = data.manifest()

st.caption(text.SNAPSHOT_CAPTION.format(generated=info.get("generated", "not yet")))

# One filter row, above everything it scopes. The market-specific sections below read it;
# the cross-market comparisons deliberately ignore it and show all three.
market = st.radio(
    text.MARKET_PROMPT,
    options=list(data.MARKETS),
    horizontal=True,
    captions=[text.MARKET_CAPTIONS[m] for m in data.MARKETS],
)

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
st.header(text.S3_HEADING.format(market=market))

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

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S5_HEADING)

tab_hour, tab_month = st.tabs([text.S5_HOUR_TAB, text.S5_MONTH_TAB])
with tab_hour:
    frame = data.profile_frame("mae", "by_local_hour")
    if not frame.empty:
        x_title, y_title = text.S5_HOUR_AXES
        st.altair_chart(
            viz.profile(frame, "bucket", x_title, "value", y_title), width="stretch"
        )
        table_view(frame.round(3), text.S5_HOUR_TABLE)
    st.markdown(text.S5_HOUR_PROSE)
with tab_month:
    frame = data.profile_frame("coverage", "by_month")
    if not frame.empty:
        x_title, y_title = text.S5_MONTH_AXES
        st.altair_chart(
            viz.profile(frame, "bucket", x_title, "value", y_title), width="stretch"
        )
        table_view(frame.round(3), text.S5_MONTH_TABLE)
    st.markdown(text.S5_MONTH_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S6_HEADING)

st.markdown(text.S6_LEDE)

weather = data.weather_frame()
if not weather.empty:
    st.dataframe(weather, width="stretch", hide_index=True)

st.markdown(text.S6_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S7_HEADING)

offsets = data.offsets_frame()
if not offsets.empty:
    st.altair_chart(viz.learned_offsets(offsets), width="stretch")
    table_view(offsets.round(0), text.S7_OFFSETS_TABLE)

st.markdown(text.S7_PROSE)

per_holiday = data.per_holiday_frame(market)
if not per_holiday.empty:
    st.altair_chart(viz.paired_holidays(per_holiday), width="stretch")
    table_view(per_holiday.round(0), text.S7_HOLIDAY_TABLE.format(market=market))

st.markdown(text.S7_CLOSING)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.S8_HEADING)

st.markdown(text.S8_PROSE)

st.divider()

# ----------------------------------------------------------------------------------------
st.header(text.METHOD_HEADING)

with st.expander(text.METHOD_SUMMARY):
    st.markdown(text.METHOD_PROSE)

st.caption(data.attribution())
