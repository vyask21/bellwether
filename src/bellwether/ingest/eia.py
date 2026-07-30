"""Client for the EIA v2 API — hourly electricity demand, generation, and interchange.

Docs: https://www.eia.gov/opendata/documentation.php
The `electricity/rto/region-data` route reports UTC-stamped hourly values per balancing
authority. The API caps a single response at 5,000 rows, so every pull paginates.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from bellwether.config import settings

log = logging.getLogger(__name__)

BASE_URL = "https://api.eia.gov/v2"
REGION_DATA_ROUTE = "electricity/rto/region-data/data/"

# EIA caps a single response at 5,000 rows regardless of a larger `length`.
MAX_PAGE_SIZE = 5000

# EIA's Terms of Service prohibit "excessive automated server request loops". A two-year
# hourly backfill is ~4 pages per series, so throttling costs us almost nothing and keeps
# a bug that loops on pagination from hammering a public service.
MIN_REQUEST_INTERVAL_SECONDS = 0.25

# Series types published per balancing authority on the region-data route.
SERIES_TYPES = {
    "D": "demand",
    "DF": "day_ahead_demand_forecast",
    "NG": "net_generation",
    "TI": "total_interchange",
}

# Balancing authorities we track. Scoped deliberately small — depth over breadth.
BALANCING_AUTHORITIES = {
    "CISO": "California ISO",
    "ERCO": "ERCOT (Texas)",
}


@dataclass(frozen=True, slots=True)
class ObservationRow:
    """One hourly observation for a (balancing authority, series type) pair."""

    period: datetime
    respondent: str
    series_type: str
    value: float | None
    value_units: str


class EIAClient:
    """Paginating, retrying client for the EIA v2 API.

    The API key is passed as a header rather than a query parameter so it never lands in
    logs or exception messages that echo the request URL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
    ) -> None:
        self._api_key = api_key or settings.require_eia_key()
        self._min_request_interval = min_request_interval
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={
                "X-Api-Key": self._api_key,
                # Identify the client so EIA can see who is calling, per their ToS.
                "User-Agent": "bellwether/0.1 (https://github.com/vyask21/bellwether)",
            },
        )

    def __enter__(self) -> EIAClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, route: str, params: list[tuple[str, str]]) -> dict:
        self._throttle()
        response = self._client.get(route, params=params)
        response.raise_for_status()
        return response.json()

    def _throttle(self) -> None:
        """Space requests out to stay well inside EIA's acceptable-use terms."""
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
        self._last_request_at = time.monotonic()

    def fetch_region_data(
        self,
        respondent: str,
        series_type: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[ObservationRow]:
        """Yield hourly observations for one BA and series type over [start, end).

        Pages until the API's reported total is exhausted. Timestamps are UTC.
        """
        if series_type not in SERIES_TYPES:
            raise ValueError(f"Unknown series type {series_type!r}; expected {set(SERIES_TYPES)}")

        offset = 0
        total: int | None = None

        while total is None or offset < total:
            # EIA uses repeated bracketed keys, so params must be a list of pairs.
            params: list[tuple[str, str]] = [
                ("frequency", "hourly"),
                ("data[0]", "value"),
                ("facets[respondent][]", respondent),
                ("facets[type][]", series_type),
                ("start", _to_eia_hour(start)),
                ("end", _to_eia_hour(end)),
                ("sort[0][column]", "period"),
                ("sort[0][direction]", "asc"),
                ("offset", str(offset)),
                ("length", str(MAX_PAGE_SIZE)),
            ]
            payload = self._get(REGION_DATA_ROUTE, params)
            body = payload.get("response", {})

            if total is None:
                total = int(body.get("total", 0))
                log.info(
                    "EIA %s/%s: %s rows over %s..%s",
                    respondent,
                    series_type,
                    total,
                    start.date(),
                    end.date(),
                )

            rows = body.get("data", [])
            if not rows:
                break

            for row in rows:
                yield _parse_row(row)

            offset += len(rows)


def _to_eia_hour(moment: datetime) -> str:
    """EIA hourly routes expect `YYYY-MM-DDTHH` in UTC."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H")


def _parse_row(row: dict) -> ObservationRow:
    raw_value = row.get("value")
    return ObservationRow(
        period=datetime.strptime(row["period"], "%Y-%m-%dT%H").replace(tzinfo=UTC),
        respondent=row["respondent"],
        series_type=row["type"],
        # EIA reports genuine gaps as null; keep them null rather than coercing to zero,
        # which would silently corrupt any demand aggregate computed downstream.
        value=float(raw_value) if raw_value is not None else None,
        value_units=row.get("value-units", ""),
    )
