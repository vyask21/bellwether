# Bellwether

Probabilistic load forecasting for the US electricity grid, with an evidence-grounded
explanation layer.

Bellwether ingests public grid and weather data, forecasts electricity demand with
uncertainty intervals, detects when reality falls outside those intervals, and explains
the breach against retrieved evidence.

Design constraint: **no number in a brief is ever generated.** Forecasts come from
time-series models and are scored as forecasts; evidence is computed in Python; briefs are
assembled from that evidence and then mechanically checked, so every numeric claim traces
back to a specific measurement or the brief is rejected.

The shipped explanation layer is deterministic and needs no API key. A language-model path
exists behind the same verifier for whoever wants better prose, but it has never been run
and `anthropic` is not a dependency.

## Status

| Phase | Status |
|---|---|
| EIA ingestion into DuckDB | done, 3 markets, 2 years hourly |
| Rolling-origin backtest (MASE, WQL, coverage) | done |
| Seasonal-naive baselines | done, see [results](docs/RESULTS.md) |
| Chronos-Bolt vs baselines | done, [wins on all metrics](docs/RESULTS.md) |
| TimesFM comparison | done, [Chronos wins on accuracy, both miss coverage the same way](docs/RESULTS.md) |
| Chronos-Bolt small vs base | done, [nine tenths of the gain for a third of the compute](docs/RESULTS.md) |
| Operator baseline (EIA `DF` series) | done, [splits by market](docs/RESULTS.md) |
| NOAA weather ingestion | done, 14 stations, hourly temperature |
| Corrector ablation: weather and volatility | done, [both predictions failed usefully](docs/RESULTS.md) |
| Breach detection and error decomposition | done, [where the forecast fails](docs/RESULTS.md) |
| Holiday corrector, pooled and split by observance | done, [measured twice, shipped neither](docs/RESULTS.md) |
| NDFD forecast temperature ingestion | done, 731 days backfilled, [validated against the observations](docs/DATA_SOURCES.md) |
| Forecast vs observed temperature, three arms | done, [four fifths of the weather gain survives](docs/RESULTS.md) |
| Nuclear outage and energy disruption ingestion | todo |
| Brief generation, with citation verification | done, deterministic, no API key |
| Findings walkthrough | done, reads committed files only |
| Deploy to Hugging Face Spaces | done, static site, no server on the serving path |
| Scheduled refresh | built, weekly, against a 2.0 MB committed Parquet store; never yet fired |

A companion project is sketched and parked: an MCP server exposing EIA data to LLM agents,
reusing this project's compliant API client. Nothing here depends on it. See
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
python scripts/run_holiday_arm.py CISO               # holiday shift, pooled and by class

pip install -e ".[ndfd]"                             # eccodes, a GRIB2 decoder
python scripts/ingest_ndfd.py --start 2024-07-31 --end 2026-07-31 --skip-stored

python scripts/sync_store.py dump                    # mirror the source tables to store/
python scripts/sync_store.py restore                 # rebuild a database from that mirror

python scripts/export_snapshot.py CISO               # ~8 min, writes snapshot/
pip install -e ".[dashboard]" && streamlit run dashboard/app.py   # local renderer

pip install -e ".[deploy]" && hf auth login          # once per machine
python scripts/build_static_space.py                 # the published page, into .space/
python scripts/deploy_space.py                       # dry run; --push to publish
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
| [NOAA NDFD archive](https://registry.opendata.aws/noaa-ndfd/) (`s3://noaa-ndfd-pds/wmo/`) | Archived forecast temperature, 3-hourly, for scoring a weather corrector on what was actually available |
| [NOAA/NWS API](https://www.weather.gov/documentation/services-web-api) | Live forecasts and weather alerts (planned) |
| EIA nuclear outages, energy disruptions | Evidence for briefs (planned) |

Weather comes from three NOAA surfaces because they answer different questions. NCEI
archives observations permanently after quality control, which is what a backtest needs.
The NDFD archive keeps the forecasts as they were published, which is what scoring a
weather corrector honestly needs: each window sees only the run issued before it opened.
NWS carries the live forecast a running system would call and retains about a week of
observations, so it can substitute for neither. NCEI's quality-control pass is also why it
lags: its archive currently ends in August 2025 while EIA demand runs to the present, so
weather experiments are scoped to the overlap.

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

The database file itself is not committed, but its three source tables are, as
`store/*.parquet` — 2.0 MB against a 35 MB database. That exists for the weekly refresh,
which runs on a machine that has never seen this repository before: it rebuilds a database
from the store in seconds rather than re-ingesting two years from two agencies. Model
output is not in the store; it is reproduced by re-running the harness. Both exports sort
by primary key, so a file changes only when the data does.

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
  storage/db.py         Schema, idempotent upsert, sorted Parquet export and restore
  storage/queries.py    Gap-aware series loading, weather gridding and weighting
  forecast/base.py      Forecaster protocol
  forecast/baseline.py  Seasonal-naive with empirical residual quantiles
  forecast/residual.py  Weather-conditioned residual quantile correction
  explain/evidence.py   Candidate explanations computed from stored data
  explain/brief.py      Brief generation, and the check that it cited nothing else
  eval/metrics.py       MASE, WQL, pinball loss, coverage, sharpness
  eval/backtest.py      Rolling-origin evaluation
  eval/ablation.py      Weather ablation against a calendar-only control
  eval/breaches.py      Interval breaches as episodes, error by hour and season

dashboard/
  app.py                The findings walkthrough, top to bottom
  loaders.py            Committed Parquet and JSON, nothing else
  viz.py                Validated palette and the chart builders

snapshot/               Committed on purpose, ~1 MB per market. See scripts/export_snapshot.py
store/                  The source tables as Parquet, 2.0 MB. See scripts/sync_store.py
.github/workflows/
  ci.yml                Lint, format gate, tests on 3.11 and 3.12
  refresh.yml           Weekly: restore, ingest, dump, re-export, commit
```

Baselines and foundation models share one `Forecaster` protocol, so the backtest harness
cannot tell them apart and no model gets an easier evaluation path.

## License

MIT
