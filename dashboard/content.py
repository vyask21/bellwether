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
PAGE_TITLE = "Bellwether: diagnosing electricity demand anomalies"

LEDE = (
    "**When grid demand departs from what was expected, what caused it?** "
    "Bellwether forecasts demand with an uncertainty band, flags the hours that fall "
    "outside it, and attributes each of those episodes to a cause computed from stored "
    "history. Three US balancing authorities, two years of hourly data. The forecast is "
    "the instrument rather than the argument: a band you cannot trust cannot tell you an "
    "hour is unusual, so sections 1 to 5 establish that it can, and section 6 is what the "
    "project is for."
)

SNAPSHOT_CAPTION = (
    "Snapshot generated {generated}. "
    "Three balancing authorities out of 83, chosen to contrast rather than to sample."
)

NO_SNAPSHOT = (
    "No snapshot found. Charts drawn from stored forecasts will be empty until "
    "`python scripts/export_snapshot.py <market>` has been run for each market."
)

MARKET_PROMPT = "Market for this section"
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
S2_HEADING = "2. A reference point: the operator's own day-ahead forecast"
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
S4_MODELS_PROSE = (
    "**A second model, and the defect stays where it was.** Every distributional claim on "
    "this page rested on one checkpoint, and a level error a few points under nominal is "
    "exactly the kind of thing that could belong to the model or to the grid. TimesFM 2.5 "
    "(200M parameters, trained by different people on a different corpus) was run against "
    "Chronos-Bolt (205M) on the same 702 windows, the same 24 hour horizon, and the same "
    "2,048 hours of context. Matching the context matters: TimesFM accepts far more, and "
    "letting it read years where the other reads days would compare two amounts of evidence "
    "and report the difference as a difference of method.\n\n"
    "Chronos-Bolt wins on accuracy in all three markets, by 4.0% of sMAPE on PACE, 8.3% on "
    "ERCO and 10.7% on CISO, with WQL and MASE agreeing everywhere. **That is the smaller "
    "half of the result.** The larger half is that both models land under nominal on ERCO "
    "and PACE and both land closest on CISO. The ordering of the three markets by "
    "calibration difficulty survives a change of checkpoint, which is what a property of "
    "the grid looks like and not what a property of a model looks like.\n\n"
    "Two further arms are on the chart below: Chronos-Bolt at 48M parameters against its own "
    "205M sibling, and TimesFM re-run on its own context ceiling rather than the matched "
    "one. Both are drawn here because they were scored on the same windows and are read "
    "with the same rule, and each is the subject of a passage underneath. **The matched "
    "context turns out not to be what lost**, which is the last passage in this section."
)
S4_MODELS_CLOSING = (
    "**And this section's own rule earns its keep on California.** Read from the coverage "
    "column alone, TimesFM's 80.2% beats Chronos-Bolt's 79.6% and CISO is the one market "
    "where the challenger calibrates better. Read with the width beside it, CISO is the "
    "market where it pays most for the appearance: that half point costs a band 12.9% "
    "wider. On ERCO no such care is needed, where it covers 2.5 points less on a band 6.4% "
    "wider, which is worse in both directions at once."
)
S4_MODELS_TABLE = "Table view: the three checkpoints"

S4_SMALL_LEDE = (
    "**Four fifths of the parameters buy a tenth of the margin.** The small checkpoint "
    "carries 48M against base's 205M and read the same 702 windows with the same 2,048 "
    "hours of context. Measured as the share of base's gain over the daily seasonal naive "
    "that it keeps, rather than as a ratio of the metrics themselves, it holds 90.8% on "
    "ERCO, 93.0% on CISO and 95.3% on PACE, with WQL agreeing to within a point everywhere. "
    "It also beats TimesFM 2.5 on MASE, WQL and sMAPE in all three markets on a quarter of "
    "the parameters, which removes the equal-budget condition the comparison above was "
    "careful to impose and leaves that ordering standing anyway."
)
S4_SMALL_PROSE = (
    "**On the interval it buys nothing, and this section's rule catches it a third time.** "
    "CISO's 81.4% is the first above-nominal coverage figure anywhere on this page, and it "
    "costs a band 10.1% wider. ERCO and PACE are flat on coverage, 77.3% against 77.4% and "
    "79.1% against 79.1%, on bands 4.8% and 2.7% wider: less sharp everywhere and no better "
    "conditioned. The market ordering survives the third checkpoint as well, ERCO worst and "
    "CISO best, so the level error above has now held across two training corpora, two "
    "architectures and a 4x parameter range.\n\n"
    "**None of this is a reason to change the shipped model.** The small checkpoint ran "
    "3.1x faster in the session that measured it, and nothing here is compute bound: a "
    "day-ahead forecast has hours to make a prediction that takes under a second. It would "
    "matter somewhere that is bound, which the weekly refresh may turn out to be. That "
    "speed figure is a within-session ratio on purpose, because accuracy in this project "
    "has reproduced exactly three times and runtime has never reproduced once."
)

S4_CONTEXT_LEDE = (
    "**The context was not what lost.** The comparison above matched TimesFM to Chronos's "
    "2,048 hours and said plainly that this left a question open: a model held to a rival's "
    "limit might be losing on evidence rather than on method. So it was re-run on its own "
    "ceiling, **16,256 hours**, 7.9 times the matched context and more history than "
    "Chronos-Bolt is able to read at all, over the same 702 windows."
)
S4_CONTEXT_PROSE = (
    "**The extra history is worth something real, small, and different in every market.** "
    "MASE improves 4.3% on ERCO, 2.2% on CISO and 1.1% on PACE, with WQL and sMAPE moving "
    "the same way and about as far inside each market. Three metrics agreeing within a "
    "market is what makes it an effect; the spread between markets is what stops it being "
    "one number, and nothing measured here predicts which end a market lands on.\n\n"
    "**And this section's rule finally catches a model in its favour.** On ERCO coverage "
    "rises 2.5 points onto a band that is 0.6% **narrower**, the only place on this page "
    "where coverage has not been paid for with width. That is what running short of "
    "evidence looks like: given more, the model became both more accurate and more certain. "
    "On CISO and PACE the bands widen instead, 1.8% and 1.9%, for 1.7 and 1.2 points of "
    "coverage, which is the ordinary purchase every other arm here has made; on CISO it "
    "buys a figure that now sits 1.9 points above nominal rather than closer to it. So the "
    "narrowing is a rescue of one badly under-covering market, not a property of longer "
    "context.\n\n"
    "**It still does not win.** Against Chronos-Bolt base it closes between a quarter and a "
    "half of the accuracy gap and stops, in every market. On ERCO it arrives at exactly "
    "base's coverage, 77.4% against 77.4%, and needs a band 5.8% wider to get there. It "
    "does not beat the 48M checkpoint either. Eight times the evidence, read by a model "
    "four times that size, lands behind both: the ordering in this section is a property of "
    "the models and not of what they were allowed to read."
)


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
S6_HEADING = "6. What caused it: anomalies attributed to stored evidence"
S6_LEDE = (
    "This is the part the rest of the project exists to support. Section 5 says where the "
    "forecast fails; this says **why**, episode by episode, with numbers a reader can check."
)
S6_PROSE = (
    "Each of the ten largest episodes per market is put to three screens computed in "
    "Python from stored data: an unusual temperature against the preceding fortnight, a "
    "US federal holiday inside the window, and a data-quality check for values no grid "
    "could physically produce. **A cause was found for 28 of the 30 episodes.** "
    "Temperature leads with 23, holidays account for 4, and one episode is not a grid "
    "event at all.\n\n"
    "**Nothing here is generated.** Every quantity is measured, and the written brief for "
    "each episode is checked token by token against the evidence it was given: a number "
    "that does not trace to a measurement means the brief is rejected rather than shown. "
    "All 30 briefs pass that check. An explanation that invents a figure is worse than no "
    "explanation, because it reads exactly like one that did not.\n\n"
    "**The most severe episode in the analysis is not a grid event.** It is an EIA value "
    "of 11,819 MW sitting between two hours near 29,900. A grid does not shed and recover "
    "60% of its load in two hours, so the screen marks it an artifact and disqualifies it "
    "from being explained. Without that screen this page would carry a confident account "
    "of a blackout that never happened.\n\n"
    "**What this does not claim.** Thirty episodes is a small sample, the three screens "
    "are the ones the detector's own output justified rather than a complete taxonomy, "
    "and strength orders candidates rather than estimating a probability. Two PACE "
    "episodes have no candidate at all and are reported as such."
)
S6_TABLE = "Table view: attribution rate by market"
S6_EPISODE_TABLE = "Table view: {market} largest anomalies and their causes"

# ----------------------------------------------------------------------------------------
S7_HEADING = "7. Temperature buys accuracy and does not fix calibration"
S7_LEDE = (
    "Against a **calendar-only control**, not against the raw model. That control is what "
    "makes the number mean anything: rebuilding a predictive distribution from residual "
    "quantiles is itself a recalibration, so a weather corrector scored only against the "
    "base model collects credit for work that has nothing to do with weather."
)
S7_PROSE = (
    "**ERCOT is the most temperature-coupled market of the three by a wide margin**, and "
    "that was measured before any model was fitted: summer correlation 0.925, winter "
    "-0.694, against CISO's 0.559 and *positive* 0.254. It predicted the ordering above "
    "correctly.\n\n"
    "California's demand barely tracks its own weather and its winter correlation has the "
    "wrong sign. Gas heating plus behind-the-meter solar, which is the duck curve restated."
)
S7_FORECAST_LEDE = (
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
S7_FORECAST_PROSE = (
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
S8_HEADING = "8. Holidays: the effect is real and it is confined to six days"
S8_PROSE = (
    "One offset applied to all eleven federal holidays improved 28 of 33 widely-observed "
    "holidays and only 10 of 27 federal-only ones. Below a coin flip on the second group is "
    "the signature of a correction being applied where nothing needs correcting.\n\n"
    "Splitting the offset by whether private employers actually close confirms it. **In "
    "ERCO and PACE the federal-only offset does not shrink, it changes sign.** Demand on "
    "Veterans Day in Texas is not depressed at all, and the pooled version had been pushing "
    "those days down because Christmas dragged the shared estimate with it."
)
S8_CLOSING = (
    "**It was measured twice and shipped neither time**, and the stated reason was that a "
    "single scalar shift over 24 hours is the wrong shape: load barely moves overnight and "
    "falls hard through the working day, so one number over-corrects the small hours to "
    "reach the large ones. That was a diagnosis rather than a measurement, so it was tested."
)
S8_SHAPE_PROSE = (
    "**The diagnosis was right, and it was not what was blocking anything.** A third arm "
    "learns an offset per hour rather than per day, and the shape it finds is larger than "
    "the scalar implied. On a widely-observed holiday ERCO is 132 MW below normal overnight "
    "and 2,315 below at 08:00, a 17.6x swing that no single number can express; CISO runs "
    "714 against 2,080. The arm beats the one it replaces in **all three** markets, which is "
    "more than that arm could say for itself, and improves 27 of 33 widely-observed "
    "holidays.\n\n"
    "**It still does not ship, and the third measurement is the one that explains why.** "
    "Against no calendar at all, CISO wins clearly, at 26.4% of its holiday error. ERCO and "
    "PACE improve on 12 of 20 holidays each, which is a coin flip. The giveaway is that on "
    "ERCO the *entire* calendar family fails the same way: 11 of 20 for the flat arm, 11 for "
    "the split one, 12 for this one. What is missing there is not a better corrector but a "
    "holiday effect worth correcting, and no amount of shaping manufactures one.\n\n"
    "Three arms, three declines, and the reason changed. The first two were declined because "
    "they might have been the wrong correction. This one is the right correction and is "
    "declined because two of three markets do not need it, which leaves exactly one live "
    "option: a holiday arm for California alone."
)
S8_SHAPE_TABLE = "Table view: the learned hour profile"
S8_OFFSETS_TABLE = "Table view: learned offsets"
S8_SHAPE_AXES = ("Local hour", "Holiday offset (MW), widely observed holidays")
S8_HOLIDAY_TABLE = "Table view: {market} per-holiday change"

# ----------------------------------------------------------------------------------------
S9_HEADING = "9. Two published claims that turned out to be wrong"
S9_PROSE = (
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
    "weather numbers use observed temperature, which is perfect foresight. Section 7 also "
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
