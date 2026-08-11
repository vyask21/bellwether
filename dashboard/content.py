"""The walkthrough's prose, written once and rendered twice.

The Space is static HTML and the local development loop is `streamlit run dashboard/app.py`,
which means two renderers for one argument. Prose duplicated across them would drift, and
the failure mode is specific: a finding gets re-measured, one copy is updated, and the
published page keeps asserting the version that was withdrawn. So every sentence lives
here, and neither renderer owns any of it.

Markdown, not HTML, because Streamlit renders it directly and the static builder runs it
through a CommonMark parser. Layout stays with each renderer: the same paragraph sits in a
column next to a chart in one and under it in the other, and forcing those to agree would
buy nothing.
"""

from __future__ import annotations

TITLE = "Bellwether"
PAGE_TITLE = "Bellwether: forecasting electricity demand"

LEDE = (
    "**Does a time-series foundation model beat the people who run the grid?** "
    "Three US balancing authorities, two years of hourly demand, everything measured "
    "against controls rather than against nothing."
)

SNAPSHOT_CAPTION = (
    "Snapshot generated {generated}. "
    "Three balancing authorities out of 83, chosen to contrast rather than to sample."
)

NO_SNAPSHOT = (
    "No snapshot found. Charts drawn from stored forecasts will be empty until "
    "`python scripts/export_snapshot.py <market>` has been run for each market."
)

MARKET_PROMPT = "Market for the single-market sections"
MARKET_CAPTIONS = {
    "CISO": "California ISO",
    "ERCO": "ERCOT, Texas",
    "PACE": "PacifiCorp East, Utah",
}

# ----------------------------------------------------------------------------------------
S1_HEADING = "1. The model beats the statistical baselines everywhere"
S1_PROSE = (
    "Zero-shot, with no weather input and no training on these series. "
    "MASE below 1.0 means it beat the naive baseline on its own scale.\n\n"
    "**The daily seasonal lag beats the weekly one in every market**, which contradicts "
    "the textbook expectation for electricity load. Weather persistence is the likely "
    "reason: yesterday resembles today because yesterday's weather does."
)
S1_TABLE = "Table view: skill against baselines"

# ----------------------------------------------------------------------------------------
S2_HEADING = "2. Against the operator's own forecast, it splits"
S2_PROSE = (
    "**ERCOT beats us by 31%. PacifiCorp East does not beat us, or even seasonal-naive.** "
    "That reads as forecasting investment rather than method: ERCOT runs a large, "
    "weather-driven market and forecasts it accordingly.\n\n"
    "CISO is absent on purpose. Its `DF` series diverges from its own `D` series at midday "
    "in a way nobody here has explained, and an operator baseline nobody understands is "
    "worse than no operator baseline."
)

# ----------------------------------------------------------------------------------------
S3_HEADING = "3. A forecast, drawn. {market}"
S3_TITLE = "{market}: 24 hours ahead from {date}"
S3_CAPTION = (
    "Blue is what happened. Orange is the forecast median, and the shaded band is the "
    "80% interval, which should contain the actual on 4 days in 5."
)
S3_SLIDER = "Forecast origin"
S3_TABLE = "Table view: this forecast window"
S3_EMPTY = "Run the snapshot export to draw forecasts."

# ----------------------------------------------------------------------------------------
S4_HEADING = "4. Never report coverage without width"
S4_PROSE = (
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
S4_TABLE = "Table view: coverage and width"

# ----------------------------------------------------------------------------------------
S5_HEADING = "5. Where the forecast actually fails"
S5_HOUR_TAB = "By hour of day"
S5_MONTH_TAB = "By month"
S5_HOUR_PROSE = (
    "Peak error lands at 17:00 local in ERCO and 15:00 in CISO, the evening ramp in "
    "both cases. Aggregate metrics hide this completely.\n\n"
    "**A trap worth naming.** Hour of day and horizon step are the same variable in a "
    "single backtest run, because origins advance by exactly the horizon, so every hour "
    "is always forecast at the same lead time. The first version of this analysis "
    "reported a diurnal profile that was line for line a horizon-step profile, looked "
    "entirely reasonable, and named the wrong hours. These numbers pool four staggered "
    "origin sets so the two cross."
)
S5_MONTH_PROSE = (
    "**Three markets, three different bad seasons.** ERCO is worst in January and "
    "November, CISO in March and May, PACE in June and November. No single calendar "
    "rule helps all three: widening winter would target ERCO's worst season and PACE's "
    "second best."
)
S5_HOUR_TABLE = "Table view: error by local hour"
S5_MONTH_TABLE = "Table view: coverage by month"
S5_HOUR_AXES = ("Local hour", "MAE (% of mean demand)")
S5_MONTH_AXES = ("Month", "80% interval coverage (%)")

# ----------------------------------------------------------------------------------------
S6_HEADING = "6. Temperature buys accuracy and does not fix calibration"
S6_LEDE = (
    "Against a **calendar-only control**, not against the raw model. That control is what "
    "makes the number mean anything: rebuilding a predictive distribution from residual "
    "quantiles is itself a recalibration, so a weather corrector scored only against the "
    "base model collects credit for work that has nothing to do with weather."
)
S6_PROSE = (
    "**ERCOT is the most temperature-coupled market of the three by a wide margin**, and "
    "that was measured before any model was fitted: summer correlation 0.925, winter "
    "-0.694, against CISO's 0.559 and *positive* 0.254. It predicted the ordering above "
    "correctly.\n\n"
    "California's demand barely tracks its own weather and its winter correlation has the "
    "wrong sign. Gas heating plus behind-the-meter solar, which is the duck curve restated."
)
S6_FORECAST_LEDE = (
    "**Everything above was measured with observed temperature**, which hands the corrector "
    "perfect knowledge of tomorrow and makes every weather number a ceiling. The table below "
    "replaces it with the forecast a forecaster would actually have had: NOAA's archived "
    "NDFD grids, restricted per window to the freshest run published at or before that "
    "window opened.\n\n"
    "Archived forecast temperature is three-hourly and everything else here is hourly, so a "
    "straight swap would differ from the arm above in **two** ways at once, forecast error "
    "and resolution, and attribute the difference to whichever the reader already believed. "
    "The middle arm is the observed series put through the coarseness and none of the error.\n\n"
    "All three are scored on one set of 362 origins, bounded by the archive rather than by "
    "the forecast, and each carries its own calendar-only control which comes out identical "
    "in all three passes. That is the check that they saw the same windows. It is also why "
    "the observed arm below reads a few tenths off the table above, which scores more "
    "windows: the comparison that matters is within a table, never across the two."
)
S6_FORECAST_PROSE = (
    "**Four fifths of the ceiling survives contact with a real forecast.** On ERCO, the only "
    "market with a large weather effect, perfect foresight cuts sMAPE 12.3% against the "
    "calendar control and a forecast available in advance cuts it 9.8%. PACE behaves the "
    "same way on a smaller effect, 3.5% falling to 2.9%. CISO is not evidence either way: "
    "its whole weather effect is 1.6% and the three arms sit inside 0.01 points of each "
    "other, so the forecast arm coming out nominally ahead of the observed one is noise.\n\n"
    "**Resolution costs nothing.** The three-hourly arm matches the hourly one in all three "
    "markets to within 0.003 points of sMAPE, and sometimes on the better side. Demand "
    "responds to temperature slowly enough that sampling it eight times a day loses nothing "
    "a corrector can use, so the entire shortfall is forecast error. The concern that "
    "motivated the arm was unfounded, which is only knowable because it was measured.\n\n"
    "Where the forecast does cost is calibration, which is the same shape as the rest of "
    "this section: ERCO's coverage falls from 77.2% on observed temperature to 76.0% on "
    "forecast, against a nominal 80%. The honest summary is not that weather helps only "
    "with perfect foresight, but that weather helps and most of the help is available in "
    "advance."
)

# ----------------------------------------------------------------------------------------
S7_HEADING = "7. Holidays: the effect is real and it is confined to six days"
S7_PROSE = (
    "One offset applied to all eleven federal holidays improved 28 of 33 widely-observed "
    "holidays and only 10 of 27 federal-only ones. Below a coin flip on the second group is "
    "the signature of a correction being applied where nothing needs correcting.\n\n"
    "Splitting the offset by whether private employers actually close confirms it. **In "
    "ERCO and PACE the federal-only offset does not shrink, it changes sign.** Demand on "
    "Veterans Day in Texas is not depressed at all, and the pooled version had been pushing "
    "those days down because Christmas dragged the shared estimate with it."
)
S7_CLOSING = (
    "**It was measured twice and shipped neither time.** Against no calendar at all only "
    "CISO shows a market-level win, worth 17.6% of its holiday error, and corrected for "
    "three markets that does not clear the bar. The honest reason is not caution: it is "
    "that a single scalar shift over 24 hours is the wrong shape. Load barely moves "
    "overnight and falls hard through the working day, so one number over-corrects the "
    "small hours to reach the large ones."
)
S7_OFFSETS_TABLE = "Table view: learned offsets"
S7_HOLIDAY_TABLE = "Table view: {market} per-holiday change"

# ----------------------------------------------------------------------------------------
S8_HEADING = "8. Two published claims that turned out to be wrong"
S8_PROSE = (
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

# ----------------------------------------------------------------------------------------
METHOD_HEADING = "Method and sources"
METHOD_SUMMARY = "Scope, and what these numbers are not"
METHOD_PROSE = (
    "- **Three balancing authorities out of 83**, chosen to contrast rather than to "
    "sample. No state is a unit in this data.\n"
    "- **The weather ceiling is measured, and so is the distance to it.** The headline "
    "weather numbers use observed temperature, which is perfect foresight. Section 6 also "
    "scores the same correction against NOAA's archived forecasts, restricted to runs "
    "published before each window opened, and four fifths of the gain survives.\n"
    "- NCEI's archive ends about eleven months before EIA's data does, so weather work "
    "covers roughly half the demand grid. That limit was accepted rather than patched "
    "with a third-party source.\n"
    "- Rolling-origin backtest, 24 hour horizon, scored on MASE, WQL, sMAPE, and 80% "
    "interval coverage with width."
)

FALLBACK_ATTRIBUTION = (
    "Source: U.S. Energy Information Administration. Forecasts are produced by this "
    "project, not by EIA, and are not authoritative."
)

TABLE_VIEW = "Table view"
