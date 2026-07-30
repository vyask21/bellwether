# Bellwether

Probabilistic load forecasting for the US electricity grid, with an evidence-grounded
explanation layer.

Bellwether ingests public grid and weather data, forecasts electricity demand with
uncertainty intervals, detects when reality falls outside those intervals, and explains
the breach against retrieved evidence.

Design constraint: the language model never produces a number. Forecasts come from
time-series models and are scored as forecasts. The LLM only retrieves, attributes, and
narrates. Every numeric claim in a generated brief must trace back to a forecasting-layer
value.

## Status

| Phase | Status |
|---|---|
| EIA ingestion into DuckDB | done, not yet run on live data |
| Rolling-origin backtest (MASE, WQL, coverage) | done |
| Seasonal-naive baselines | done |
| Chronos-Bolt and TimesFM comparison | todo |
| Weather features (NOAA/NWS) | todo |
| Nuclear outage ingestion | todo |
| Anomaly detection and brief generation | todo |
| Scheduled refresh and dashboard | todo |

## Metrics

Every model emits quantiles and is scored on three axes:

* **MASE**: point accuracy scaled by seasonal-naive error on the same training window.
  Scale-free, so different-sized markets are comparable. Above 1.0 means the model lost to
  the naive baseline.
* **WQL**: weighted quantile loss, a discrete CRPS approximation. Scores the whole
  predictive distribution.
* **80% interval coverage**: what fraction of actuals landed inside the claimed 80% band.
  Under-coverage means overconfidence.

EIA also publishes each balancing authority's own day-ahead forecast (`DF`), which serves
as an operator baseline alongside the statistical ones.

## Setup

Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate        # source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env          # then fill in EIA_API_KEY
```

Free EIA key: https://www.eia.gov/opendata/register.php

## Usage

```bash
bellwether ingest --respondent CISO --days 730
bellwether status
bellwether backtest --respondent CISO --horizon 24
```

`ingest` is idempotent. EIA restates recent values, so re-running an overlapping window
converges on the latest published number instead of duplicating rows.

## Data sources

| Source | Used for |
|---|---|
| [EIA v2 API](https://www.eia.gov/opendata/) | Hourly demand, day-ahead forecast, net generation, interchange |
| [NOAA/NWS API](https://www.weather.gov/documentation/services-web-api) | Temperature forecasts, weather alerts (planned) |
| EIA nuclear outages | Generator outage evidence for briefs (planned) |

Scoped to CISO (California ISO) and ERCO (ERCOT) rather than all ~60 balancing
authorities.

### Attribution

Data courtesy of the U.S. Energy Information Administration
([eia.gov/opendata](https://www.eia.gov/opendata/)), public domain, used under their
[API Terms of Service](https://www.eia.gov/opendata/terms-of-service.php).

The client throttles to 0.25s between requests and sends an identifying `User-Agent`, per
the ToS prohibition on excessive automated request loops. A two-year backfill is about 4
requests per series, so the throttle costs nothing and bounds the blast radius of a
pagination bug.

The API key goes in an `X-Api-Key` header rather than an `api_key=` query parameter. Both
work; the header keeps the key out of request URLs, which are what tends to reach logs.

## Storage

DuckDB, in-process. At this volume (about 10^5 rows) no database is a bottleneck, so the
choice is friction and fit: columnar engine suited to the window-function-heavy backtest,
installs with pip, identical on Windows, Linux, and CI, keeps the repo clone-and-run.

DuckDB is single-writer. The ingest job writes and exports a Parquet snapshot; concurrent
readers use the snapshot.

## Development

```bash
pytest
ruff check .
ruff format .
```

## Layout

```
src/bellwether/
  config.py             Settings from .env
  ingest/eia.py         Paginating EIA v2 client
  storage/db.py         Schema, idempotent upsert, Parquet snapshots
  storage/queries.py    Gap-aware series loading
  forecast/base.py      Forecaster protocol
  forecast/baseline.py  Seasonal-naive with empirical residual quantiles
  eval/metrics.py       MASE, WQL, pinball loss, coverage
  eval/backtest.py      Rolling-origin evaluation
```

Baselines and foundation models share one `Forecaster` protocol, so the backtest harness
cannot tell them apart and no model gets an easier evaluation path.

## License

MIT
