# EIA API Terms of Service compliance

Canonical text: https://www.eia.gov/opendata/terms-of-service.php
Reviewed against: 2026-07-30.

The TOS reserves EIA's right to revise these terms unilaterally, and continued use
constitutes acceptance. Re-read the canonical page before any change to how this project
calls the API, and update the review date above.

Every clause below maps to a specific decision in this repository. If a future change
conflicts with any row, the TOS wins.

## Use

> You may use the EIA API to develop a service to search, display, analyze, retrieve, view
> and otherwise "get" information from EIA data.

Bellwether retrieves and analyzes. It never writes to EIA. The client issues `GET` only
(`ingest/eia.py`).

## Attribution

> You should use the "EIA" or the "U.S. Energy Information Administration" names in order
> to identify the source of API content. You may not use the EIA names, or the like to
> imply endorsement or approval of any product, service, or entity.

Attribution text lives in one place, `bellwether/attribution.py`, so every surface that
displays EIA content emits the same string. It is printed by the CLI and belongs on any
future dashboard.

Wording is deliberately neutral ("Source: U.S. Energy Information Administration"). Phrases
like "courtesy of", "powered by", or "in partnership with" are not used, since they can
read as endorsement.

## EIA logo

> The EIA logo is trademarked and may not be used without written permission.

No EIA logo appears anywhere in this repository, and none may be added to the README,
docs, or any dashboard.

## Modification or false representation

> You may not modify or falsely represent content accessed through the API and still claim
> the source is the EIA.

This is the clause with the widest reach in a forecasting project, because most of what
Bellwether produces is derived data that is emphatically not EIA content.

Enforcement:

* Raw API observations are stored in the `observations` table, unmodified. Values are
  written exactly as returned.
* Nulls stay null. EIA reports genuine gaps as null, and zero-filling them would be a
  modification presented as EIA data. Backtest windows containing gaps are skipped rather
  than imputed.
* Model output is stored in a separate `forecasts` table with a `model_name` column. It is
  never mixed into `observations` and never attributed to EIA.
* Any displayed value carries the attribution appropriate to its origin: EIA attribution
  for observations, model attribution for forecasts. The two are never blended into one
  figure presented as EIA data.
* Generated briefs may cite EIA observations as EIA data. They may not present a forecast,
  an interpolation, or any LLM-produced number as an EIA figure.

## Right to limit

> Your use of the API may be subject to certain limitations on access, calls, or use. If
> the EIA reasonably believes that you have attempted to exceed or circumvent these limits,
> your ability to use the API may be temporarily or permanently blocked.

Published guidance (https://www.eia.gov/opendata/faqs.php) is a burst rate under 5
requests/second and a sustained rate under 9,000/hour, with actual limits varying by key,
series demand, and IP.

Enforcement:

* Requests are paced at a minimum 1.0s interval, EIA's own suggested value, roughly a fifth
  of the published burst ceiling.
* Requests are sequential. No concurrency, no worker pool, no parallel ingest.
* One API key. No key rotation, no multiple keys, no IP rotation. Any of these would be an
  attempt to circumvent limits.
* A 429 is met with exponential backoff, not immediate retry. Backing off is the compliant
  response to throttling.
* Scheduled refresh runs at most hourly, pulling a short recent window, which is a handful
  of requests per run.

For large historical pulls, EIA directs users to the bulk download facility rather than
sequential API calls. If this project ever expands past a couple of balancing authorities,
use that instead of raising request volume.

## As is, as available

> The API is provided "as is" and on an "as-available" basis. EIA makes no warranty that
> the API will be error free or that access thereto will be continuous or uninterrupted.

Ingestion must tolerate the API being unavailable or returning partial data. Failures are
logged and the run exits without corrupting stored state; the next run resumes, since
upserts are idempotent. Data staleness is surfaced by `bellwether status` rather than
hidden.

## Limitations on liability

> In no event will the EIA be liable ... for interruption of use or loss or corruption of
> data.

Nothing produced here may be presented as authoritative or operational. Forecasts are
model output on public data, published as a portfolio project. Any user-facing surface
carries a disclaimer to that effect.

## General representations

> Your use of the API will be in strict accordance with the EIA privacy policy, this
> Agreement, and all applicable laws and regulations.

The API key is loaded from `.env`, which is gitignored, and is sent as a request header so
it does not appear in URLs that reach logs. No key is committed. No EIA content is
redistributed in a way that misrepresents its source.

## Changes

> The EIA reserves the right, at its sole discretion, to modify or replace this Agreement.

Hence the review date at the top of this file. Treat it as expiring.
