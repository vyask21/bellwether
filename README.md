# Bellwether

Probabilistic electricity demand forecasting for the US power grid.

Bellwether reads public grid and weather data, forecasts demand 24 hours ahead with
uncertainty intervals, detects when demand lands outside those intervals, and explains each
miss from measured evidence. It covers three regional grid operators over two years of
hourly data.

Every number in a written explanation traces back to a measurement. An explanation citing
anything else is rejected rather than published.

## Metrics

Forecasts are judged three ways, and all three are reported together.

MASE measures point accuracy against a seasonal naive forecast on the same data. Below 1.0
means the model beat that baseline.

WQL scores the full range of predicted outcomes rather than a single best guess.

Coverage and width are always reported as a pair. Any forecast can hit its claimed 80
percent interval by predicting a very wide range, so neither number means much alone.

## Results

Over 702 rolling test windows per operator, the model cut forecast error against the best
statistical baseline by 43 percent in California, 29 percent in Utah and 26 percent in
Texas.

Interval accuracy is weaker and is reported as such. Intervals claiming 80 percent of
demand held 77 to 80 percent, so they are overconfident.

Against each operator's own day-ahead forecast the result splits: Texas forecasts its
demand better than this project does, Utah does not, and California is excluded.

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
  eval/        Metrics, backtest, ablations
  explain/     Evidence and written briefs
dashboard/     Findings walkthrough
scripts/       Analysis runs and export
docs/          Results and source notes
tests/         504 tests, Python 3.11 and 3.12
```

Baselines and foundation models share one interface, so the harness cannot tell them
apart.

## License

MIT
