# Bellwether

Probabilistic load forecasting for the US electricity grid, with an evidence-grounded
explanation layer.

Bellwether ingests live public grid and weather data, produces **probabilistic** forecasts
of electricity demand, detects when reality falls outside the forecast interval, and
explains those breaches against retrieved evidence — outage notices, weather alerts,
market events.

**The design constraint that shapes everything:** the language model never invents a
number. Forecasts come from time-series models and are scored as forecasts. The LLM only
retrieves, attributes, and narrates, and its output is evaluated separately. Any numeric
claim in a generated brief must trace back to a value the forecasting layer produced.

## Status

Early. Ingestion, storage, statistical baselines, and the backtest harness are in place.
Foundation models and the explanation layer are not yet built.

| Phase | Status |
|---|---|
| EIA ingestion → DuckDB | ✅ built, not yet run against live data |
| Rolling-origin backtest harness (MASE, WQL, coverage) | ✅ |
| Seasonal-naive baselines (daily, weekly) | ✅ |
| Chronos-Bolt / TimesFM champion–challenger | ⬜ |
| Weather features (NOAA/NWS) | ⬜ |
| Anomaly detection on interval breaches | ⬜ |
| Evidence retrieval + brief generation | ⬜ |
| Scheduled refresh + published dashboard | ⬜ |

## Why probabilistic, and why these metrics

A point forecast that says "demand will be 31.2 GW" is not actionable on its own — an
operator needs to know how wrong it could plausibly be. So every model here emits
quantiles, and is scored on three axes:

- **MASE** — point accuracy, scaled by the error of a seasonal-naive forecast on the same
  training window. Scale-free, so a 30 GW market and a 3 GW market are comparable, and
  MASE > 1 means the model lost to the naive baseline.
- **WQL** (weighted quantile loss, a discrete CRPS approximation) — scores the whole
  predictive distribution, not just its middle.
- **80% interval coverage** — of the hours the model claimed an 80% interval for, how many
  actually landed inside it? Systematic under-coverage means the model is overconfident,
  which is the failure that burns someone acting on it.

The baselines exist to be beaten, and the comparison is reported either way.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"

cp .env.example .env          # then fill in EIA_API_KEY
```

An EIA API key is free and issued instantly:
<https://www.eia.gov/opendata/register.php>

## Usage

```bash
bellwether ingest --respondent CISO --days 730   # backfill two years of hourly demand
bellwether status                                # row counts, span, and gap report
bellwether backtest --respondent CISO --horizon 24
```

`ingest` is idempotent — re-running overlapping windows converges on the latest published
values rather than duplicating rows, which matters because EIA restates recent data.

## Data sources

All free, all public, all US.

| Source | Used for |
|---|---|
| [EIA v2 API](https://www.eia.gov/opendata/) | Hourly demand, day-ahead forecast, net generation, interchange per balancing authority |
| [NOAA / NWS API](https://www.weather.gov/documentation/services-web-api) | Temperature forecasts and active weather alerts *(planned)* |
| CAISO OASIS / ERCOT | Prices and outage notices for the evidence layer *(planned)* |

Scope is deliberately narrow — CISO (California ISO) and ERCO (ERCOT/Texas) rather than
all ~60 balancing authorities. They are the most volatile and best documented, and depth
beats breadth for what this project is demonstrating.

### Attribution and acceptable use

Data courtesy of the **U.S. Energy Information Administration** ([eia.gov/opendata](https://www.eia.gov/opendata/)).
As a federal statistical agency's output it is public domain, but use is subject to EIA's
[API Terms of Service](https://www.eia.gov/opendata/terms-of-service.php), which prohibit
excessive automated request loops. The client therefore throttles to a minimum interval
between requests and identifies itself with a `User-Agent`. A two-year hourly backfill is
only about four paginated requests per series, so this costs nothing in practice — it
exists so a pagination bug cannot turn into a hammering loop against a public service.

The API key is sent as an `X-Api-Key` header rather than an `api_key=` query parameter.
Both are accepted by EIA; the header keeps the key out of request URLs, which are the
thing that most often ends up in logs, traces, and error reports.

## Storage

DuckDB, in-process. At this volume (hourly × 2 years × a handful of series ≈ 10⁵ rows) no
database is a bottleneck, so the choice is about friction and fit: DuckDB's columnar engine
suits the window-function-heavy backtest workload, installs with `pip`, behaves identically
on Windows/Linux/CI, and keeps the repo clone-and-run.

Its one real constraint is single-writer concurrency. The ingest job writes and then
exports a Parquet snapshot; anything reading concurrently reads the snapshot.

## Development

```bash
pytest              # 30 tests
ruff check .
ruff format .
```

## Layout

```
src/bellwether/
  config.py          Settings from .env
  ingest/eia.py      Paginating EIA v2 client
  storage/db.py      DuckDB schema, idempotent upsert, Parquet snapshots
  storage/queries.py Gap-aware series loading
  forecast/base.py   The Forecaster protocol every model implements
  forecast/baseline.py  Seasonal-naive with empirical residual quantiles
  eval/metrics.py    MASE, WQL, pinball loss, interval coverage
  eval/backtest.py   Rolling-origin evaluation
```

Statistical baselines and foundation models sit behind one `Forecaster` protocol, so the
backtest harness cannot tell them apart — no model gets an accidental advantage from a
different evaluation path.

## License

MIT
