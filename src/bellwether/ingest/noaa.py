"""Client for NOAA NCEI's Integrated Surface Database: hourly surface weather.

Docs: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
Format: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf

Three NOAA surfaces publish these observations and they are not interchangeable:

* **NWS** (`api.weather.gov`) is the live operational feed. It retains roughly a week, so
  it cannot support a backtest.
* **NCEI global-hourly** is the permanent archive of the same observations, after
  quality control and cross-source de-duplication. That QC pass is why it lags, and it is
  what this module reads.
* NCEI's own file archive under `/data/global-hourly/access/` carries the same content as
  whole-year CSVs. The service used here subsets by date, which keeps a resumable backfill
  cheap.

The archive lag is real and load-bearing: as of retrieval it ends in **August 2025** while
EIA demand runs to the present day. Weather experiments are therefore scoped to the
overlap rather than the full demand history. See docs/DATA_SOURCES.md.

No API key is required, and NCEI publishes no rate limit for this service. Requests are
paced anyway, on the same reasoning as the EIA client: a public service should not have to
absorb an unpaced backfill.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

BASE_URL = "https://www.ncei.noaa.gov/access/services"
DATA_ROUTE = "data/v1"
DATASET = "global-hourly"

# NCEI documents no published limit for this service, so this is a courtesy pace rather
# than a ceiling we are avoiding. A full backfill is a few hundred requests.
MIN_REQUEST_INTERVAL_SECONDS = 1.0

# NCEI documents 400 (malformed) and 500 (service fault) for this route. 400 is permanent:
# retrying a malformed request just burns the backoff budget before the same error.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Requesting the whole window in one call works, but chunking keeps responses small and
# lets an interrupted backfill resume near where it stopped rather than from the start.
DEFAULT_CHUNK_DAYS = 90

# ISD stores air temperature in tenths of a degree Celsius, with 9999 as the missing
# sentinel. A raw "+0106,1" is 10.6 C with quality code 1.
TEMPERATURE_SCALE = 10.0
MISSING_TEMPERATURE_TENTHS = 9999

# Air-temperature quality codes meaning "suspect" or "erroneous", per the ISD format
# document. Everything else (0, 1, 4, 5, 9, A, C, I, M, P, R, U) is either a passed check
# or a documented validator edit. These are kept in the raw table exactly as NOAA
# published them and screened at read time, so the stored record stays faithful to the
# source and the screening policy stays visible where it is applied.
SUSPECT_QUALITY_CODES = frozenset({"2", "3", "6", "7"})

# Daily and monthly summary rows. Their timestamp is a bookkeeping stamp, not an
# observation time, and their fields carry period aggregates rather than point readings.
# Storing them in a table keyed by observation time would misrepresent both.
SUMMARY_REPORT_TYPES = frozenset({"SOD", "SOM"})


@dataclass(frozen=True, slots=True)
class Station:
    """A surface station standing in for part of a balancing authority's load.

    `population` is the metro population used to weight this station in the market
    average. It is a weight, not a claim about who serves those customers.
    """

    call_sign: str
    station_id: str  # ISD identifier: USAF number concatenated with WBAN number
    place: str
    population: int


@dataclass(frozen=True, slots=True)
class WeatherRow:
    """One temperature reading as NOAA published it."""

    observed_at: datetime
    station_id: str
    report_type: str
    temperature_c: float | None
    quality_code: str


# Stations per balancing authority, weighted by metro population.
#
# Two caveats that the weights do not capture, recorded because they bound what the
# feature can do rather than because they are fixable here:
#
# 1. Population is a proxy for load, not load itself. Permian Basin oil and gas load in
#    ERCOT and trona and gas processing load in Wyoming are large and are weighted here by
#    the small resident populations of Midland-Odessa and Casper. Both markets are
#    therefore under-weighted toward their industrial regions.
# 2. Stations are climate proxies, not service-territory assertions. CAISO does not serve
#    the City of Los Angeles (LADWP is a separate balancing authority) or Sacramento
#    (SMUD, within BANC), so those are excluded from the CISO weights, but the airport
#    standing in for a metro sits wherever the airport sits.
#
# Populations are 2020 Census metropolitan statistical area totals, rounded to the nearest
# thousand. Rounding does not move the weights meaningfully and the extra digits would
# imply a precision the metro-to-territory mapping does not have.
MARKET_STATIONS: dict[str, tuple[Station, ...]] = {
    "CISO": (
        # LA metro less the City of LA, which LADWP serves rather than CAISO.
        Station("KBUR", "72288023152", "Burbank, LA basin", 9_302_000),
        Station("KRIV", "72286023119", "March AFB, Inland Empire", 4_600_000),
        Station("KSFO", "72494023234", "San Francisco Bay", 4_749_000),
        Station("KSAN", "72290023188", "San Diego", 3_299_000),
        Station("KSJC", "72494523293", "San Jose", 2_000_000),
        Station("KFAT", "72389093193", "Fresno, Central Valley", 1_009_000),
    ),
    "ERCO": (
        Station("KDFW", "72259003927", "Dallas-Fort Worth", 7_637_000),
        Station("KIAH", "72243012960", "Houston", 7_122_000),
        Station("KSAT", "72253012921", "San Antonio", 2_558_000),
        Station("KAUS", "72254013904", "Austin", 2_283_000),
        Station("KMAF", "72265023023", "Midland-Odessa, Permian Basin", 340_000),
    ),
    "PACE": (
        # One station for the whole Wasatch Front: Salt Lake City, Provo, and Ogden sit in
        # a single valley system and share a climate far more than they differ.
        Station("KSLC", "72572024127", "Wasatch Front", 2_624_000),
        Station("KIDA", "72578524145", "Idaho Falls", 152_000),
        Station("KCPR", "72569024089", "Casper", 80_000),
    ),
}


def stations_for(respondent: str) -> tuple[Station, ...]:
    """Stations weighted into one balancing authority's temperature series."""
    try:
        return MARKET_STATIONS[respondent]
    except KeyError:
        raise ValueError(
            f"No weather stations configured for {respondent!r}; "
            f"expected one of {sorted(MARKET_STATIONS)}"
        ) from None


class NCEIClient:
    """Chunking, retrying client for NCEI's Access Data Service."""

    def __init__(
        self,
        timeout: float = 120.0,
        min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        chunk_days: int = DEFAULT_CHUNK_DAYS,
    ) -> None:
        self._min_request_interval = min_request_interval
        self._chunk_days = chunk_days
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={
                # NCEI asks callers to identify themselves. There is no key to hide here,
                # so this is the only thing tying a request back to this project.
                "User-Agent": "bellwether/0.1 (https://github.com/vyask21/bellwether)",
            },
        )

    def __enter__(self) -> NCEIClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception(lambda exc: _is_retryable(exc)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, params: list[tuple[str, str]]) -> list[dict]:
        self._throttle()
        response = self._client.get(DATA_ROUTE, params=params)
        response.raise_for_status()
        payload = response.json()
        # A window with no archived data returns an empty list rather than an error.
        return payload if isinstance(payload, list) else []

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
        self._last_request_at = time.monotonic()

    def fetch_temperatures(
        self,
        station_id: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[WeatherRow]:
        """Yield temperature readings for one station over [start, end).

        Summary rows are dropped. Readings NOAA flagged suspect are kept, with their
        quality code, and screened at read time.
        """
        for chunk_start, chunk_end in _chunks(start, end, self._chunk_days):
            params: list[tuple[str, str]] = [
                ("dataset", DATASET),
                ("stations", station_id),
                ("dataTypes", "TMP"),
                ("startDate", chunk_start.strftime("%Y-%m-%d")),
                ("endDate", chunk_end.strftime("%Y-%m-%d")),
                ("format", "json"),
            ]
            rows = self._get(params)
            log.info(
                "NCEI %s: %s rows over %s..%s",
                station_id,
                len(rows),
                chunk_start.date(),
                chunk_end.date(),
            )
            for row in rows:
                parsed = _parse_row(row)
                if parsed is not None:
                    yield parsed


def _chunks(start: datetime, end: datetime, chunk_days: int) -> Iterator[tuple[datetime, datetime]]:
    """Split [start, end) into consecutive windows of at most `chunk_days`."""
    if chunk_days < 1:
        raise ValueError(f"chunk_days must be at least 1, got {chunk_days}")
    span = timedelta(days=chunk_days)
    cursor = start
    while cursor < end:
        stop = min(cursor + span, end)
        yield cursor, stop
        cursor = stop


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _parse_row(row: dict) -> WeatherRow | None:
    """Turn one ISD record into a `WeatherRow`, or None if it is not an observation."""
    report_type = str(row.get("REPORT_TYPE", "")).strip()
    if report_type in SUMMARY_REPORT_TYPES:
        return None

    raw = str(row.get("TMP", ""))
    temperature_c, quality_code = _parse_temperature(raw)

    return WeatherRow(
        # ISD timestamps are UTC and carry no offset, so the zone is attached explicitly
        # rather than left to whatever the reading machine's locale would infer.
        observed_at=datetime.strptime(row["DATE"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC),
        station_id=str(row["STATION"]),
        report_type=report_type,
        temperature_c=temperature_c,
        quality_code=quality_code,
    )


def _parse_temperature(raw: str) -> tuple[float | None, str]:
    """Split an ISD `TMP` field into degrees Celsius and its quality code.

    Returns None for the missing sentinel. A malformed field is treated as missing rather
    than raising: one unparseable record should not abort a backfill, and a null is
    indistinguishable downstream from the gap it actually represents.
    """
    value, _, code = raw.partition(",")
    code = code.strip()
    try:
        tenths = int(value)
    except ValueError:
        return None, code
    if abs(tenths) == MISSING_TEMPERATURE_TENTHS:
        return None, code
    return tenths / TEMPERATURE_SCALE, code
