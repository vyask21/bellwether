# Bellwether

When electricity demand departs from what was expected, what caused it?

Bellwether answers that. It forecasts demand for US grid operators with an uncertainty
band, flags the hours outside it, and attributes each episode to a cause computed from
stored history: unusual weather, a public holiday, or a data fault.

Every number in an explanation traces to a measurement. One citing anything else is
rejected rather than published.

## What it finds

Across the 30 largest demand anomalies in three grids, a cause was found for 28.

Temperature accounts for 23, public holidays for 4, and one is not a grid event: a
reported 11,819 MW between two hours near 29,900. No grid sheds and recovers 60 percent of
its load in two hours, so it is flagged as a fault rather than written up as a blackout.

All 30 briefs pass the citation check. Two anomalies have no candidate cause and are
reported as such.

## Metrics

The band must be trustworthy before an hour outside it means anything, so it is judged
three ways, reported together.

MASE is point accuracy against a seasonal naive forecast; below 1.0 beats it. WQL scores
the full predicted range. Coverage and width are always a pair, since any forecast can hit
a claimed 80 percent interval by predicting a very wide range.

Over 702 windows per market it cut error 43, 29 and 26 percent against the best baseline.
Its 80 percent bands held 77 to 80 percent.

## Data sources

Hourly demand, day-ahead forecasts, net generation and interchange come from the U.S.
Energy Information Administration. Hourly observed temperature comes from NOAA's Integrated
Surface Database, across 14 weather stations. Archived temperature forecasts come from
NOAA's National Digital Forecast Database.

An EIA key is free and issued immediately at
https://www.eia.gov/opendata/register.php

NOAA data requires no key.

## Attribution and terms

Source: U.S. Energy Information Administration, https://www.eia.gov/opendata/. Used under
the EIA API Terms of Service and Copyrights and Reuse policy.

Source: NOAA National Centers for Environmental Information, Integrated Surface Database.

Both publish in the public domain. Acknowledgments carry a retrieval date.

This project is not affiliated with or endorsed by EIA. It is
not affiliated with or endorsed by NOAA. Forecasts here are this
project's and are not authoritative.

## Layout

```
src/bellwether/
  ingest/      EIA and NOAA API clients
  storage/     Schema, loading, export
  forecast/    Baselines and foundation models
  eval/        Anomaly detection, metrics, backtest
  explain/     Evidence and written briefs
dashboard/     Findings walkthrough
scripts/       Analysis runs and export
docs/          Results and compliance note
tests/         508 tests, Python 3.11 and 3.12
```

Evidence is computed in Python, so no explanation can introduce a number the data lacks.

## License

MIT
