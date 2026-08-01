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
| EIA ingestion into DuckDB | done, 3 markets, 2 years hourly |
| Rolling-origin backtest (MASE, WQL, coverage) | done |
| Seasonal-naive baselines | done, see [results](docs/RESULTS.md) |
| Chronos-Bolt vs baselines | done, [wins on all metrics](docs/RESULTS.md) |
| TimesFM comparison | todo |
| Operator baseline (EIA `DF` series) | done, [splits by market](docs/RESULTS.md) |
| NOAA weather ingestion | done, 14 stations, hourly temperature |
| Weather ablation vs a calendar control | done, [prediction half failed](docs/RESULTS.md) |
| Breach detection and error decomposition | done, [miscoverage is seasonal](docs/RESULTS.md) |
| Nuclear outage and energy disruption ingestion | todo |
| Brief generation from retrieved evidence | todo |
| Scheduled refresh and dashboard | todo |

A companion project is sketched but not started: an MCP server exposing EIA data to LLM
agents, reusing this project's compliant API client. See
[docs/COMPANION_MCP_SERVER.md](docs/COMPANION_MCP_SERVER.md).

Current status, decisions, and next steps: [docs/STATE.md](docs/STATE.md).

## Metrics

Every model emits quantiles and is scored on three axes:

* **MASE**: point accuracy scaled by seasonal-naive error on the same training window.
  Scale-free, so different-sized markets are comparable. Above 1.0 means the model lost to
  the naive baseline.
* **WQL**: weighted quantile loss, a discrete CRPS approximation. Scores the whole
  predictive distribution.
* **80% interval coverage and width**: what fraction of actuals landed inside the claimed
  80% band, and how wide that band was. Reported as a pair, because neither means much
  alone: a model can buy coverage by widening and sharpen its way back out of it, and both
  look the same in a coverage column.

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
bellwether ingest-weather --respondent CISO --days 730
bellwether status
bellwether backtest --respondent CISO --horizon 24

python scripts/run_weather_ablation.py ERCO          # weather vs a calendar control
python scripts/analyze_breaches.py ERCO --stagger 4  # where the forecast fails
```

Both ingest commands are idempotent. EIA restates recent values and NCEI revises its
archive as late reports and quality control land, so re-running an overlapping window
converges on the latest published number instead of duplicating rows.

`--stagger` is not optional detail. Origins advance by exactly the horizon, so within one
run every local hour is always forecast at the same lead time, which makes hour of day and
horizon step the same variable under two names. Pooling offset origin sets is what
separates them, and skipping it produces a diurnal profile that is really a horizon
profile and names the wrong hours.

## Data sources

| Source | Used for |
|---|---|
| [EIA v2 API](https://www.eia.gov/opendata/) | Hourly demand, day-ahead forecast, net generation, interchange |
| [NOAA NCEI Integrated Surface Database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Hourly observed temperature, 14 stations |
| [NOAA/NWS API](https://www.weather.gov/documentation/services-web-api) | Temperature forecasts, weather alerts (planned) |
| EIA nuclear outages, energy disruptions | Evidence for briefs (planned) |

Weather comes from two NOAA surfaces because they answer different questions. NCEI
archives observations permanently after quality control, which is what a backtest needs
and what is ingested today. NWS carries the live forecast, which is what a running
day-ahead forecast will need, and retains about a week of observations, so it cannot
substitute. NCEI's quality-control pass is also why it lags: its archive currently ends in
August 2025 while EIA demand runs to the present, so weather experiments are scoped to the
overlap.

Scoped to three balancing authorities out of the 83 the API lists: CISO (California ISO),
ERCO (ERCOT), and PACE (PacifiCorp East). No state is a unit in this data. Also scoped to
a handful of EIA routes rather than the full catalog, since most EIA data is monthly or
annual and carries no signal at hourly resolution. What was evaluated and rejected, and
why, is in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

### Attribution and terms

Source: U.S. Energy Information Administration (EIA),
[eia.gov/opendata](https://www.eia.gov/opendata/). Used under the
[API Terms of Service](https://www.eia.gov/opendata/terms-of-service.php) and the
[Copyrights and Reuse policy](https://www.eia.gov/about/copyrights_reuse.php).

EIA data is public domain, so derived artifacts and snapshots can be committed here. Its
reuse policy asks acknowledgments to carry a date; APIv2 exposes no publication date, so
acknowledgments carry the retrieval date and say so. EIA's logo and Energy Ant servicemark
are trademarked and appear nowhere in this project, and site imagery may be privately
licensed, so no eia.gov images are reproduced.

This project is not affiliated with or endorsed by EIA. Forecasts and derived values are
produced by this project, not by EIA, are stored separately from EIA observations, and are
not authoritative.

Every API-related decision here maps to a TOS clause. See
[docs/EIA_COMPLIANCE.md](docs/EIA_COMPLIANCE.md).

EIA's [FAQ](https://www.eia.gov/opendata/faqs.php) gives a burst ceiling under 5
requests/second and a sustained ceiling under 9,000/hour, noting that real limits vary by
key usage, series demand, and IP, and that some routes are stricter. The client paces at
1.0s between requests, EIA's own conservative suggestion, and sends an identifying
`User-Agent`. A two-year backfill is about 4 requests per series, so pacing an order of
magnitude below the ceiling costs seconds and removes any chance of a key ban.

Retries cover 429 and 5xx with exponential backoff. A 403 fails immediately, since a bad
key never recovers.

If this ever scales past a couple of balancing authorities, the
[bulk download facility](https://www.eia.gov/opendata/bulkfiles.php) is the right tool for
large historical pulls rather than sequential API calls.

The API key goes in an `X-Api-Key` header rather than an `api_key=` query parameter. Both
work; the header keeps the key out of request URLs, which are what tends to reach logs.

Source: NOAA National Centers for Environmental Information, Integrated Surface Database
(ISD). NOAA data is a US Government work and so public domain, which makes the obligation
citation rather than permission. NCEI is named alongside its dataset because NOAA runs many
archives and "NOAA" alone does not identify which one a number came from. Acknowledgments
carry a retrieval date, which matters more here than for EIA: NCEI revises the archive as
late reports arrive and quality control runs, so two retrievals of the same hour can
differ.

This project is not affiliated with or endorsed by NOAA.

NCEI's Access Data Service needs no key and publishes no rate limit, so the 1.0s pacing is
a courtesy rather than a ceiling being avoided. Retries cover 429 and 5xx; a 400 fails
immediately, since a malformed request never becomes valid. Readings NOAA flagged suspect
or erroneous are stored with their quality code and screened when read, so the stored table
stays a faithful copy of the archive and the screening policy stays visible where it is
applied.

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
  ingest/noaa.py        NCEI hourly weather client, station registry and weights
  storage/db.py         Schema, idempotent upsert, Parquet snapshots
  storage/queries.py    Gap-aware series loading, weather gridding and weighting
  forecast/base.py      Forecaster protocol
  forecast/baseline.py  Seasonal-naive with empirical residual quantiles
  forecast/residual.py  Weather-conditioned residual quantile correction
  eval/metrics.py       MASE, WQL, pinball loss, coverage, sharpness
  eval/backtest.py      Rolling-origin evaluation
  eval/ablation.py      Weather ablation against a calendar-only control
  eval/breaches.py      Interval breaches as episodes, error by hour and season
```

Baselines and foundation models share one `Forecaster` protocol, so the backtest harness
cannot tell them apart and no model gets an easier evaluation path.

## License

MIT
