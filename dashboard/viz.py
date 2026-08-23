"""Palette and chart builders for the dashboard.

## The palette is validated, not chosen by eye

Every categorical set below was run through the reference validator before use, on the
light surface the app pins:

* 3 series, all pairs: worst CVD dE 9.2, worst normal-vision dE 24.0.
* 2 series, all pairs: worst CVD dE 24.7.
* Diverging poles: worst CVD dE 21.6.

The one warning is aqua at 2.74:1 against the surface, below the 3:1 bar. The documented
relief for that is visible labels or a table view, so **every chart in this app ships a
table view beneath it**. That is an obligation carried by the palette, not a nicety.

## Two rules that shaped these charts

No dual axes anywhere. Coverage and interval width are the two halves of one finding and
the temptation to overlay them on one plot is exactly the mistake the finding warns about,
so they are drawn as two charts sharing a category axis.

Cross-market quantities are normalised by each market's mean demand. ERCO's interval is
7,000 MW wide and PACE's is 200, and plotting both raw would say only that Texas is
bigger than Utah, which nobody needed a chart to learn.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed slot order. Colour follows the entity: a market keeps its hue in every chart, and
# filtering one out never repaints the others.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKET_COLOURS = {"CISO": SERIES[0], "ERCO": SERIES[1], "PACE": SERIES[2]}

# Polarity, not identity: blue and red read as opposite with a neutral between them.
BETTER, WORSE, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def enable_theme() -> None:
    """Register and enable the theme, across the Altair 5 and 6 APIs.

    `alt.themes.register` was deprecated in Altair 5.5 in favour of `alt.theme.register`.
    Both are handled because the Space pins a range rather than an exact version, and a
    deprecation warning printed into a dashboard is a defect a reader can see.
    """
    if hasattr(alt, "theme"):
        alt.theme.register("bellwether", enable=True)(theme)
    else:  # Altair < 5.5
        alt.themes.register("bellwether", theme)
        alt.themes.enable("bellwether")


def theme() -> dict:
    """Recessive chrome, hairline grid, no dashes."""
    return {
        "config": {
            "background": SURFACE,
            "font": FONT,
            "view": {"stroke": None, "continuousWidth": 700, "continuousHeight": 300},
            "axis": {
                "labelColor": INK_MUTED,
                "titleColor": INK_SECONDARY,
                "titleFontWeight": 500,
                "gridColor": GRID,
                "gridWidth": 1,
                "domainColor": AXIS,
                "tickColor": AXIS,
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titlePadding": 10,
            },
            "legend": {
                "labelColor": INK_SECONDARY,
                "titleColor": INK_SECONDARY,
                "labelFontSize": 12,
                "titleFontSize": 12,
                "symbolStrokeWidth": 2,
                "orient": "top",
                "direction": "horizontal",
                "titleLimit": 0,
            },
            "title": {"color": INK_PRIMARY, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
            "range": {"category": SERIES},
        }
    }


def _market_scale(markets) -> alt.Scale:
    """Pin each market to its slot so a subset never triggers a repaint."""
    ordered = [m for m in ("CISO", "ERCO", "PACE") if m in set(markets)]
    return alt.Scale(domain=ordered, range=[MARKET_COLOURS[m] for m in ordered])


# Hours of observed demand drawn before the forecast starts, so the reader can see what the
# model was extrapolating from rather than only where it landed.
LEAD_IN_HOURS = 24


def forecast_window(
    observed: pd.DataFrame, forecast: pd.DataFrame, title: str, lead_in: int = LEAD_IN_HOURS
) -> alt.Chart:
    """One 24 hour forecast: the 80% band, its median, and what actually happened.

    The band is the point of the chart, so it is drawn first and palest, with the two lines
    over it. Actual takes slot 1 because it is the reference every reader looks for.

    **`observed` is clipped here rather than by the caller.** Altair embeds its data in the
    chart specification, so handing this the full two-year series inlines 17,521 rows of
    JSON into a page that draws 48 of them. Doing the slice inside means no caller can get
    it wrong, which is the same reason `load_market_temperature` takes a grid instead of
    deriving one.
    """
    start = forecast["period"].min() - pd.Timedelta(hours=lead_in)
    observed = observed[observed["period"].between(start, forecast["period"].max())]

    band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.18, color=SERIES[1])
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("q10:Q", title="Demand (MW)", scale=alt.Scale(zero=False)),
            y2="q90:Q",
        )
    )
    median = (
        alt.Chart(forecast)
        .mark_line(strokeWidth=2, color=SERIES[1])
        .encode(x="period:T", y="q50:Q")
    )
    actual = (
        alt.Chart(observed)
        .mark_line(strokeWidth=2, color=SERIES[0])
        .encode(x="period:T", y="demand_mw:Q")
    )
    # Crosshair and tooltip over the whole plot rather than the marks, so the hit target is
    # the column of pixels above any hour instead of a 2px line.
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["period"], empty=False)
    rule = (
        alt.Chart(observed)
        .mark_rule(color=INK_MUTED, strokeWidth=1)
        .encode(
            x="period:T",
            opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
            tooltip=[
                alt.Tooltip("period:T", title="Hour", format="%b %d %H:%M"),
                alt.Tooltip("demand_mw:Q", title="Actual MW", format=",.0f"),
            ],
        )
        .add_params(hover)
    )
    return (band + median + actual + rule).properties(title=title, height=320)


def forecast_explorer(
    observed_url: str, forecast_url: str, initial: int, lead_in: int = LEAD_IN_HOURS
) -> alt.LayerChart:
    """`forecast_window` with the origin chosen by a signal instead of by Python.

    The static page has no server to re-slice a frame on every drag, so the slice moves
    into the chart: both datasets ship whole and a filter picks the window. This is only
    affordable because `origin` **is** the observed series' row index, so the lead-in is
    arithmetic rather than a date join. `build_static_space.py` asserts that alignment per
    market rather than trusting it, since a silent drift here would draw a confident chart
    of the wrong two days.

    Kept beside `forecast_window` on purpose. They are the same picture built twice, and
    two functions a screen apart drift visibly where two files do not.
    """
    origin = alt.param(name="originIdx", value=int(initial))
    forecast = alt.UrlData(
        url=forecast_url,
        format=alt.CsvDataFormat(
            parse={
                "period": "date",
                "origin": "number",
                "q10": "number",
                "q50": "number",
                "q90": "number",
            }
        ),
    )
    observed = alt.UrlData(
        url=observed_url,
        format=alt.CsvDataFormat(parse={"period": "date", "idx": "number", "demand_mw": "number"}),
    )
    in_window = f"datum.idx >= originIdx - {lead_in} && datum.idx < originIdx + 24"

    band = (
        alt.Chart(forecast)
        .transform_filter("datum.origin === originIdx")
        .mark_area(opacity=0.18, color=SERIES[1])
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("q10:Q", title="Demand (MW)", scale=alt.Scale(zero=False)),
            y2="q90:Q",
        )
    )
    median = (
        alt.Chart(forecast)
        .transform_filter("datum.origin === originIdx")
        .mark_line(strokeWidth=2, color=SERIES[1])
        .encode(x="period:T", y="q50:Q")
    )
    actual = (
        alt.Chart(observed)
        .transform_filter(in_window)
        .mark_line(strokeWidth=2, color=SERIES[0])
        .encode(x="period:T", y="demand_mw:Q")
    )
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["period"], empty=False)
    rule = (
        alt.Chart(observed)
        .transform_filter(in_window)
        .mark_rule(color=INK_MUTED, strokeWidth=1)
        .encode(
            x="period:T",
            opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
            tooltip=[
                alt.Tooltip("period:T", title="Hour", format="%b %d %H:%M"),
                alt.Tooltip("demand_mw:Q", title="Actual MW", format=",.0f"),
            ],
        )
        .add_params(hover)
    )
    return (band + median + actual + rule).properties(height=320).add_params(origin)


def coverage_and_width(frame: pd.DataFrame) -> alt.VConcatChart:
    """Coverage and interval width as two plots, never one.

    A model can buy coverage by widening and sharpen its way back out, and the two look
    identical in a coverage column. Putting them on one dual axis would invent a
    relationship between two scales that have none, so they share a category axis instead
    and are read together by position.
    """
    scale = _market_scale(frame["market"])
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", title=None, sort=None),
        color=alt.Color("market:N", title="Market", scale=scale),
        yOffset=alt.YOffset("market:N"),
    )
    nominal = (
        alt.Chart(pd.DataFrame({"v": [80.0]}))
        .mark_rule(color=INK_MUTED, strokeWidth=1)
        .encode(x="v:Q")
    )
    coverage = (
        base.mark_circle(size=110, stroke=SURFACE, strokeWidth=2)
        .encode(
            x=alt.X(
                "coverage:Q",
                title="80% interval coverage (%). The rule is nominal.",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("market:N"),
                alt.Tooltip("arm:N", title="Arm"),
                alt.Tooltip("coverage:Q", format=".1f", title="Coverage %"),
            ],
        )
        .properties(height=alt.Step(16))
    )
    width = (
        base.mark_circle(size=110, stroke=SURFACE, strokeWidth=2)
        .encode(
            x=alt.X("width_pct:Q", title="Mean interval width (% of mean demand)"),
            tooltip=[
                alt.Tooltip("market:N"),
                alt.Tooltip("arm:N", title="Arm"),
                alt.Tooltip("width_pct:Q", format=".1f", title="Width %"),
            ],
        )
        .properties(height=alt.Step(16))
    )
    return alt.vconcat(coverage + nominal, width, spacing=28).resolve_scale(color="shared")


def profile(frame: pd.DataFrame, x: str, x_title: str, y: str, y_title: str) -> alt.Chart:
    """Error or coverage across a cyclical axis, one line per market."""
    scale = _market_scale(frame["market"])
    hover = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)
    line = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True))
        .encode(
            x=alt.X(f"{x}:O", title=x_title),
            y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color("market:N", title="Market", scale=scale),
            tooltip=[
                alt.Tooltip("market:N"),
                alt.Tooltip(f"{x}:O", title=x_title),
                alt.Tooltip(f"{y}:Q", format=".2f", title=y_title),
            ],
        )
    )
    return (line.add_params(hover)).properties(height=300)


def paired_holidays(frame: pd.DataFrame) -> alt.Chart:
    """Per-holiday change in error, faceted by observance, coloured by sign.

    Two channels doing two jobs. Observance is identity and carries the finding, so it is
    position (the facet). Better or worse is polarity, so it is the diverging pair. Neither
    is asked to do the other's work.
    """
    return (
        alt.Chart(frame)
        .mark_bar(height=9, cornerRadius=2)
        .encode(
            x=alt.X("change:Q", title="Change in holiday MAE (MW). Left is better."),
            y=alt.Y("date:N", title=None, sort=alt.SortField("date")),
            color=alt.Color(
                "direction:N",
                title=None,
                scale=alt.Scale(domain=["better", "worse"], range=[BETTER, WORSE]),
            ),
            tooltip=[
                alt.Tooltip("date:N", title="Holiday"),
                alt.Tooltip("name:N", title="Name"),
                alt.Tooltip("observance:N", title="Observance"),
                alt.Tooltip("change:Q", format="+,.0f", title="Change MW"),
            ],
        )
        .properties(height=alt.Step(13), width=300)
        .facet(column=alt.Column("observance:N", title=None))
    )


def learned_offsets(frame: pd.DataFrame) -> alt.Chart:
    """The holiday offset each market learned per observance class.

    The finding is a sign change, so a zero rule is the most important mark here and is
    drawn heavier than the grid.
    """
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=2, height=18)
        .encode(
            x=alt.X("offset:Q", title="Learned holiday offset (MW)"),
            y=alt.Y("market:N", title=None),
            yOffset=alt.YOffset("observance:N", sort=["widely observed", "federal only"]),
            color=alt.Color(
                "observance:N",
                title=None,
                sort=["widely observed", "federal only"],
                scale=alt.Scale(range=[SERIES[0], SERIES[1]]),
            ),
            tooltip=[
                alt.Tooltip("market:N"),
                alt.Tooltip("observance:N", title="Observance"),
                alt.Tooltip("offset:Q", format="+,.0f", title="Offset MW"),
            ],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"v": [0.0]}))
        .mark_rule(color=INK_SECONDARY, strokeWidth=1.5)
        .encode(x="v:Q")
    )
    return (bars + zero).properties(height=200)


def skill_bars(frame: pd.DataFrame) -> alt.Chart:
    """Error against the best baseline, one bar per market. One series, one colour."""
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=2, height=26, color=SERIES[0])
        .encode(
            x=alt.X("reduction:Q", title="sMAPE reduction against the best baseline (%)"),
            y=alt.Y("market:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("market:N"),
                alt.Tooltip("reduction:Q", format=".1f", title="Reduction %"),
                alt.Tooltip("baseline:N", title="Baseline beaten"),
            ],
        )
    )
    # Selective direct labels: one per bar, three bars. A number on every point is chaos,
    # but three endpoints are the whole chart.
    labels = bars.mark_text(align="left", dx=6, color=INK_SECONDARY, fontSize=11).encode(
        text=alt.Text("reduction:Q", format=".1f")
    )
    return (bars + labels).properties(height=150)


def attribution_bars(frame: pd.DataFrame) -> alt.Chart:
    """Share of a market's largest anomalies that a cause was found for.

    Deliberately the same form as `skill_bars`: one series over three markets. A second
    chart shape here would imply a second kind of quantity, and a colour per cause would
    add a categorical scale the rest of the page does not use.
    """
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=2, height=26, color=SERIES[0])
        .encode(
            x=alt.X(
                "share:Q",
                title="Largest anomalies given a candidate cause (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("market:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("market:N"),
                alt.Tooltip("attributed:Q", title="Attributed"),
                alt.Tooltip("episodes:Q", title="Episodes examined"),
                alt.Tooltip("leading:N", title="Leading cause"),
            ],
        )
    )
    labels = bars.mark_text(align="left", dx=6, color=INK_SECONDARY, fontSize=11).encode(
        text=alt.Text("share:Q", format=".0f")
    )
    return (bars + labels).properties(height=150)
