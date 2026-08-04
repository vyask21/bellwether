"""Client for NOAA's National Digital Forecast Database: *forecast* temperature.

Bucket: https://registry.opendata.aws/noaa-ndfd/ (`noaa-ndfd-pds`, anonymous)
Elements: https://vlab.noaa.gov/web/mdl/ndfd

Every weather number this project has published was measured with **observed** temperature,
which gives the model perfect knowledge of tomorrow's weather. That is a ceiling, not an
operational result. This module reads the forecast a forecaster would actually have had,
so the ablation can be repeated against it.

## Which NDFD surface, and why not the obvious one

| Surface | Retention | Verdict |
|---|---|---|
| `api.weather.gov`, the point-forecast API | Current forecast only | No history, no backtest. |
| `noaa-ndfd-pds/opnl/`, the operational grids | Overwritten in place | Same problem. |
| `noaa-ndfd-pds/wmo/`, the WMO-header archive | **2020-04-16 onward** | **Taken.** |

The archive covers the whole demand history, which is the first thing that has to be true
and is not true of NCEI's *observation* archive: that one ends eleven months short. So
forecast temperature can be evaluated over a longer window than observed temperature was.

## Which grid

`wmo/temp/` carries several sectors and resolutions. Only two cover CONUS:

* `YEUZ98`, 2.5 km, about **48 MB** per issuance.
* `YEUZ88`, 5 km Lambert 1073x689, about **5.5 MB** per issuance. **Taken.**

They carry the same forecast at different spatial resolution, and this project reads
fourteen point locations out of a 1.8 million point grid. Paying 8x the bytes to move a
grid cell from 5 km to 2.5 km around an airport is not a measurement improvement worth two
years of download. At one issuance per day the 5 km product is about **4 GB** for the full
history, against 33 GB for the fine one.

## The finding that shapes the experiment

**Archived CONUS temperature is 3-hourly, not hourly.** Both products carry twenty
messages at three-hour steps (`YEUZ88` at +3..+60, `YEUZ98` offset at +2..+59). Everything
else in this project is hourly: the demand grid, the observed temperature, and the 24 hour
forecast horizon.

That means a weather arm built on this source differs from the published one in *two* ways
at once, forecast error and temporal resolution, and an ablation that changes two things
measures neither. The control that separates them is to degrade the **observed** series the
same way, to 3-hourly and back, and score a third arm on it. If the degraded-observation
arm loses most of the weather gain, the limit is resolution rather than forecast skill, and
a finer product or an interpolation scheme is the answer rather than a conclusion about
weather. This is the same reasoning that put a calendar-only control next to the weather
arm, where reading coverage alone had reversed the conclusion.

## Decoding

GRIB2, which needs a decoder. NOAA recommends `grib2io`; it has no Windows wheel and fails
to build from source here. `eccodes` publishes a native Windows wheel that carries the
ECMWF library with it, and `codes_grib_find_nearest` does the projection arithmetic, so
neither the Lambert grid nor the 0-360 longitude convention has to be reimplemented. It is
an optional extra: nothing else in this project reads GRIB.
"""

from __future__ import annotations

import logging
import struct
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from bellwether.ingest.noaa import MARKET_STATIONS

log = logging.getLogger(__name__)

BUCKET_URL = "https://noaa-ndfd-pds.s3.amazonaws.com/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

ELEMENT = "temp"
# CONUS, 5 km Lambert, projections +3h to +60h. See the module docstring for why not the
# 2.5 km sibling. `KWBN` is the issuing office in every NDFD WMO filename.
CONUS_HEADER = "YEUZ88"
ORIGINATOR = "KWBN"

# The archive begins here. Requesting earlier returns an empty listing rather than an
# error, which would otherwise look exactly like a day the model did not run.
ARCHIVE_START = date(2020, 4, 16)

# Anonymous public bucket with no published limit. Paced on the same reasoning as the EIA
# and NCEI clients: a public service should not have to absorb an unpaced backfill.
MIN_REQUEST_INTERVAL_SECONDS = 1.0

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# GRIB2 encodes 2 metre temperature in Kelvin.
KELVIN_OFFSET = 273.15

# Projections to decode, in hours ahead. The file carries +3 to +60 and the backtest scores
# a 24 hour horizon, so two thirds of every file is for an experiment this project does not
# run. Decoding is the entire cost here, about 0.33s per station per projection against a
# 1.3s download, so this is the difference between a nine hour backfill and a nineteen hour
# one. Raise it and re-run the range if a longer horizon is ever scored; the archive is not
# going anywhere.
MAX_STEP_HOURS = 27

# GRIB2 section 0: the marker, then two reserved bytes, discipline, edition, and the total
# message length as a big-endian 8 byte integer. Sixteen bytes in all.
MESSAGE_MARKER = b"GRIB"
MESSAGE_HEADER_BYTES = 16

# Station coordinates, retrieved from NCEI's own station metadata for the same identifiers
# the observation ingest uses, so a station cannot mean one place in one table and a
# different place in another. Retrieved 2026-08-03 via the Access Data Service with
# `includeStationLocation=1`.
STATION_COORDINATES: dict[str, tuple[float, float]] = {
    "72288023152": (34.19966, -118.36543),  # KBUR Burbank
    "72286023119": (33.90000, -117.25000),  # KRIV March AFB
    "72494023234": (37.61962, -122.36562),  # KSFO San Francisco
    "72290023188": (32.73360, -117.18310),  # KSAN San Diego
    "72494523293": (37.35938, -121.92444),  # KSJC San Jose
    "72389093193": (36.77999, -119.72016),  # KFAT Fresno
    "72259003927": (32.89744, -97.02196),  # KDFW Dallas-Fort Worth
    "72243012960": (29.98438, -95.36072),  # KIAH Houston
    "72253012921": (29.54429, -98.48395),  # KSAT San Antonio
    "72254013904": (30.18311, -97.67989),  # KAUS Austin
    "72265023023": (31.94754, -102.20859),  # KMAF Midland-Odessa
    "72572024127": (40.77069, -111.96503),  # KSLC Wasatch Front
    "72578524145": (43.52044, -112.06753),  # KIDA Idaho Falls
    "72569024089": (42.89778, -106.47361),  # KCPR Casper
}


@dataclass(frozen=True, slots=True)
class Issuance:
    """One published NDFD run: when it was issued and where its grids live."""

    issued_at: datetime
    key: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ForecastRow:
    """One station's forecast temperature for one hour, as NOAA published it.

    `issued_at` is when the forecast was made and `valid_at` is the hour it describes.
    Keeping both is the entire point: a forecast is only honest evidence about a demand
    hour if it was issued before that hour, and a table that stored only `valid_at` could
    not prove it.
    """

    issued_at: datetime
    valid_at: datetime
    station_id: str
    temperature_c: float | None


def stations_with_coordinates() -> dict[str, tuple[float, float]]:
    """Every station the observation ingest uses, with its grid location.

    Raises if the two ever disagree. A station added to `MARKET_STATIONS` without a
    coordinate here would otherwise be silently absent from the forecast series while every
    count and coverage report still looked complete.
    """
    configured = {station.station_id for group in MARKET_STATIONS.values() for station in group}
    missing = configured - set(STATION_COORDINATES)
    if missing:
        raise ValueError(
            f"No NDFD coordinates for station(s) {sorted(missing)}. Add them from NCEI "
            "station metadata rather than from an airport lookup, so the identifier and "
            "the position come from the same source."
        )
    return {sid: STATION_COORDINATES[sid] for sid in sorted(configured)}


class NDFDClient:
    """Listing and fetching client for the public NDFD archive."""

    def __init__(
        self,
        timeout: float = 180.0,
        min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
    ) -> None:
        self._min_request_interval = min_request_interval
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            base_url=BUCKET_URL,
            timeout=timeout,
            headers={"User-Agent": "bellwether/0.1 (https://github.com/vyask21/bellwether)"},
        )

    def __enter__(self) -> NDFDClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def list_issuances(self, day: date, header: str = CONUS_HEADER) -> list[Issuance]:
        """Every issuance published on one UTC day, oldest first."""
        if day < ARCHIVE_START:
            raise ValueError(
                f"NDFD archive begins {ARCHIVE_START}, asked for {day}. Earlier days are "
                "not missing runs, they are outside the archive."
            )
        prefix = f"wmo/{ELEMENT}/{day:%Y/%m/%d}/"
        listing = self._get_xml({"list-type": "2", "prefix": prefix, "max-keys": "1000"})
        issuances = []
        for contents in listing.findall(f"{S3_NS}Contents"):
            key = contents.findtext(f"{S3_NS}Key", "")
            name = key.rsplit("/", 1)[-1]
            if not name.startswith(f"{header}_"):
                continue
            issued_at = _issued_at(name)
            if issued_at is None:
                continue
            issuances.append(Issuance(issued_at, key, int(contents.findtext(f"{S3_NS}Size", "0"))))
        return sorted(issuances, key=lambda issuance: issuance.issued_at)

    def fetch(self, key: str) -> bytes:
        """One GRIB2 file, whole. About 5.5 MB for the 5 km CONUS product."""
        return self._get_bytes(key)

    @retry(
        retry=retry_if_exception(lambda exc: _is_retryable(exc)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_xml(self, params: dict[str, str]) -> ET.Element:
        self._throttle()
        response = self._client.get("", params=params)
        response.raise_for_status()
        return ET.fromstring(response.text)

    @retry(
        retry=retry_if_exception(lambda exc: _is_retryable(exc)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_bytes(self, key: str) -> bytes:
        self._throttle()
        response = self._client.get(key)
        response.raise_for_status()
        return response.content

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
        self._last_request_at = time.monotonic()


def select_issuance(issuances: Sequence[Issuance], at_or_before: datetime) -> Issuance | None:
    """The freshest run a forecaster could have had at `at_or_before`.

    Later is not better here. Picking the newest run of the day would hand the model
    information published after the forecast origin it is being scored at, which is the
    same leak as scoring on observed temperature, only harder to see.
    """
    eligible = [issuance for issuance in issuances if issuance.issued_at <= at_or_before]
    return max(eligible, key=lambda issuance: issuance.issued_at) if eligible else None


def split_messages(grib: bytes) -> list[bytes]:
    """Split a multi-message GRIB2 buffer into its messages.

    GRIB2 is self-delimiting: every message opens with `GRIB` and carries its own total
    length as a big-endian 8 byte integer at offset 8. Scanning for the marker and stepping
    by the declared length is the whole format, at this level.
    """
    messages, cursor = [], 0
    while True:
        start = grib.find(MESSAGE_MARKER, cursor)
        if start < 0 or start + MESSAGE_HEADER_BYTES > len(grib):
            return messages
        (length,) = struct.unpack(">Q", grib[start + 8 : start + MESSAGE_HEADER_BYTES])
        if length <= 0 or start + length > len(grib):
            # A truncated tail is a partial download, not a message. Returning what parsed
            # would quietly publish a short day.
            raise ValueError(
                f"GRIB2 message at byte {start} declares {length} bytes but only "
                f"{len(grib) - start} remain; the download is truncated."
            )
        messages.append(grib[start : start + length])
        cursor = start + length


def extract_station_forecasts(
    grib: bytes,
    coordinates: dict[str, tuple[float, float]] | None = None,
    max_step_hours: int = MAX_STEP_HOURS,
) -> list[ForecastRow]:
    """Pull each station's nearest grid point out of every message in one GRIB2 file.

    Decoded from memory rather than from a temporary file. eccodes caches the underlying
    `FILE*` against the Python file object and never releases it, which on Windows leaves
    the file locked: the first temporary file could not be deleted and the run ended on it.
    Splitting the buffer and handing eccodes one message at a time avoids the file layer
    entirely, and produces identical values.
    """
    import eccodes  # Imported here: the decoder is an optional extra, not a core dependency.

    points = coordinates if coordinates is not None else stations_with_coordinates()
    rows: list[ForecastRow] = []
    for message in split_messages(grib):
        gid = eccodes.codes_new_from_message(message)
        try:
            if int(eccodes.codes_get(gid, "forecastTime")) > max_step_hours:
                continue
            rows.extend(_read_message(eccodes, gid, points))
        finally:
            eccodes.codes_release(gid)
    return rows


def _read_message(eccodes, gid, points: dict[str, tuple[float, float]]) -> Iterator[ForecastRow]:
    """One projection's value at each station.

    **Read through `codes_grib_find_nearest`, never by indexing `codes_get_values`.** The
    two disagree on this grid: the index the nearest search reports is geographically
    right, and the value at that index in the values array is not the value the search
    returns. Half the values array reads as the 9999 sentinel while the search returns a
    temperature for the same cell. The cause was not found and the shortcut was worth about
    twenty times the speed, so it is recorded here as refused rather than left to be
    rediscovered as a plausible optimisation.
    """
    issued_at = _message_issued_at(eccodes, gid)
    valid_at = issued_at + timedelta(hours=int(eccodes.codes_get(gid, "forecastTime")))
    missing = float(eccodes.codes_get(gid, "missingValue"))
    for station_id, (latitude, longitude) in points.items():
        # NDFD grids run 0-360. eccodes normalises, but converting here keeps the stored
        # coordinate in the convention NCEI published it in.
        nearest = eccodes.codes_grib_find_nearest(gid, latitude, longitude % 360)[0]
        value = float(nearest.value)
        absent = value == missing or value != value
        yield ForecastRow(
            issued_at=issued_at,
            valid_at=valid_at,
            station_id=station_id,
            temperature_c=None if absent else value - KELVIN_OFFSET,
        )


def _message_issued_at(eccodes, gid) -> datetime:
    """The run time, from GRIB's split date and time keys. `dataTime` is HHMM."""
    stamp = str(eccodes.codes_get(gid, "dataDate"))
    minutes = int(eccodes.codes_get(gid, "dataTime"))
    return datetime.strptime(stamp, "%Y%m%d").replace(
        hour=minutes // 100, minute=minutes % 100, tzinfo=UTC
    )


def _issued_at(filename: str) -> datetime | None:
    """Parse `YEUZ88_KWBN_202501151147` into its issuance timestamp.

    Returns None rather than raising on an unexpected name. The bucket is a public archive
    and one oddly named object should not abort a two-year backfill.
    """
    parts = filename.split("_")
    if len(parts) != 3 or parts[1] != ORIGINATOR or len(parts[2]) != 12:
        return None
    try:
        return datetime.strptime(parts[2], "%Y%m%d%H%M").replace(tzinfo=UTC)
    except ValueError:
        return None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False
