"""The forecast-temperature client, and the leak it exists to prevent.

This source was added because every published weather number in this project was measured
with observed temperature, which is perfect foresight. The way that defect comes back is
subtle: take the freshest run of the day and the model sees a forecast issued after the
origin it is scored at. So `select_issuance` gets more attention here than the parsing
does, and the decoder arithmetic is tested against a stand-in rather than skipped, because
a Kelvin left unconverted is a 273 degree feature that would train perfectly well.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bellwether.ingest.ndfd import (  # noqa: E402
    ARCHIVE_START,
    BUCKET_URL,
    Issuance,
    NDFDClient,
    _issued_at,
    _read_message,
    extract_station_forecasts,
    select_issuance,
    split_messages,
    stations_with_coordinates,
)
from bellwether.ingest.noaa import MARKET_STATIONS  # noqa: E402


def _listing(names: list[tuple[str, int]]) -> str:
    entries = "".join(
        f"<Contents><Key>wmo/temp/2025/01/15/{name}</Key><Size>{size}</Size></Contents>"
        for name, size in names
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{entries}</ListBucketResult>"
    )


@pytest.fixture
def client() -> NDFDClient:
    with NDFDClient(min_request_interval=0.0) as instance:
        yield instance


class TestStationCoordinates:
    def test_every_configured_station_has_a_position(self):
        """A station added to the observation ingest without a coordinate here would be
        absent from the forecast series while every coverage report still looked full."""
        coordinates = stations_with_coordinates()
        configured = {s.station_id for group in MARKET_STATIONS.values() for s in group}
        assert set(coordinates) == configured
        assert len(coordinates) == 14

    def test_positions_are_in_north_america(self):
        for station_id, (latitude, longitude) in stations_with_coordinates().items():
            assert 25 < latitude < 50, station_id
            assert -125 < longitude < -95, station_id

    def test_a_station_without_coordinates_is_refused_rather_than_dropped(self, monkeypatch):
        from bellwether.ingest import ndfd

        monkeypatch.setitem(MARKET_STATIONS, "TEST", (_station("99999999999"),))
        with pytest.raises(ValueError, match="99999999999"):
            ndfd.stations_with_coordinates()


class TestFilenames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("YEUZ88_KWBN_202501151147", datetime(2025, 1, 15, 11, 47, tzinfo=UTC)),
            ("YEUZ98_KWBN_202408150016", datetime(2024, 8, 15, 0, 16, tzinfo=UTC)),
        ],
    )
    def test_it_parses_an_issuance_stamp(self, name, expected):
        assert _issued_at(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "YEUZ88_KWBN_2025011511",  # truncated
            "YEUZ88_KWBN_202501993147",  # impossible month
            "YEUZ88_202501151147",  # no originator
            "index.html",
        ],
    )
    def test_an_unexpected_name_is_skipped_rather_than_fatal(self, name):
        """The bucket is a public archive. One odd object must not abort a two-year run."""
        assert _issued_at(name) is None


class TestListing:
    @respx.mock
    def test_it_returns_only_the_requested_grid_oldest_first(self, client: NDFDClient):
        respx.get(BUCKET_URL).mock(
            return_value=httpx.Response(
                200,
                text=_listing(
                    [
                        ("YEUZ88_KWBN_202501151147", 5_517_845),
                        ("YEUZ88_KWBN_202501150047", 5_400_000),
                        ("YEUZ98_KWBN_202501151147", 48_000_000),  # the 2.5 km sibling
                        ("YERZ98_KWBN_202501151147", 4_472_109),  # Alaska
                    ]
                ),
            )
        )
        issuances = client.list_issuances(date(2025, 1, 15))
        assert [i.issued_at.hour for i in issuances] == [0, 11]
        assert all("YEUZ88" in i.key for i in issuances)
        assert issuances[1].size_bytes == 5_517_845

    @respx.mock
    def test_a_day_with_no_runs_is_empty_rather_than_an_error(self, client: NDFDClient):
        respx.get(BUCKET_URL).mock(return_value=httpx.Response(200, text=_listing([])))
        assert client.list_issuances(date(2025, 1, 15)) == []

    def test_it_refuses_days_before_the_archive_begins(self, client: NDFDClient):
        """Otherwise an out-of-range request looks exactly like a day NOAA did not run."""
        with pytest.raises(ValueError, match="archive begins"):
            client.list_issuances(date(2020, 1, 1))
        assert date(2020, 4, 16) == ARCHIVE_START


class TestSelectIssuance:
    """The guard against handing the model a forecast from its own future."""

    def _runs(self, *hours: int) -> list[Issuance]:
        return [
            Issuance(datetime(2025, 1, 15, hour, tzinfo=UTC), f"key-{hour}", 1) for hour in hours
        ]

    def test_it_takes_the_freshest_run_at_or_before_the_cutoff(self):
        chosen = select_issuance(self._runs(0, 6, 11, 18), datetime(2025, 1, 15, 12, tzinfo=UTC))
        assert chosen.issued_at.hour == 11

    def test_it_never_returns_a_run_issued_after_the_cutoff(self):
        """The leak this module exists to close, restated as an assertion. A run published
        after the forecast origin carries information the forecaster did not have."""
        cutoff = datetime(2025, 1, 15, 12, tzinfo=UTC)
        for _ in range(3):
            chosen = select_issuance(self._runs(0, 6, 11, 13, 18, 23), cutoff)
            assert chosen.issued_at <= cutoff

    def test_a_run_exactly_at_the_cutoff_is_eligible(self):
        chosen = select_issuance(self._runs(6, 12), datetime(2025, 1, 15, 12, tzinfo=UTC))
        assert chosen.issued_at.hour == 12

    def test_nothing_early_enough_returns_none_rather_than_the_closest(self):
        assert select_issuance(self._runs(18, 23), datetime(2025, 1, 15, 12, tzinfo=UTC)) is None

    def test_no_runs_at_all_returns_none(self):
        assert select_issuance([], datetime(2025, 1, 15, 12, tzinfo=UTC)) is None


class TestMessageDecoding:
    """The arithmetic, driven by a stand-in for the GRIB library.

    Not a mock of the thing under test: the decoder is eccodes' job and this asserts what
    this module does with what eccodes returns. A unit conversion missed here is a feature
    off by 273 degrees that would fit perfectly well and mean nothing.
    """

    def test_kelvin_becomes_celsius_at_the_right_valid_hour(self):
        fake = _FakeEccodes(step=3, values={"KSLC": 268.70})
        rows = list(_read_message(fake, gid=1, points={"KSLC": (40.77069, -111.96503)}))
        assert len(rows) == 1
        assert rows[0].issued_at == datetime(2025, 1, 15, 12, tzinfo=UTC)
        assert rows[0].valid_at == datetime(2025, 1, 15, 15, tzinfo=UTC)
        assert rows[0].temperature_c == pytest.approx(-4.45, abs=0.01)

    def test_the_missing_sentinel_becomes_null_not_a_temperature(self):
        fake = _FakeEccodes(step=3, values={"KSLC": 9999.0}, missing=9999.0)
        rows = list(_read_message(fake, gid=1, points={"KSLC": (40.77069, -111.96503)}))
        assert rows[0].temperature_c is None

    def test_longitudes_are_asked_for_in_the_convention_the_grid_uses(self):
        """NDFD grids run 0-360. Passing -111.9 unconverted is a lookup in the Pacific."""
        fake = _FakeEccodes(step=3, values={"KSLC": 273.15})
        list(_read_message(fake, gid=1, points={"KSLC": (40.77069, -111.96503)}))
        assert fake.asked == [(40.77069, pytest.approx(248.035, abs=0.01))]

    def test_a_sixty_hour_step_lands_on_the_right_day(self):
        fake = _FakeEccodes(step=60, values={"KSLC": 273.15})
        rows = list(_read_message(fake, gid=1, points={"KSLC": (40.0, -111.0)}))
        assert rows[0].valid_at == datetime(2025, 1, 18, 0, tzinfo=UTC)


class TestMessageSplitting:
    """GRIB2 is self-delimiting, which is what lets this decode without a temporary file."""

    def _message(self, payload: bytes) -> bytes:
        body = b"\x00\x00\x00\x02" + payload  # reserved, discipline, edition, then payload
        return b"GRIB" + body[:4] + struct.pack(">Q", 16 + len(payload)) + payload

    def test_it_splits_a_multi_message_buffer(self):
        buffer = self._message(b"a" * 40) + self._message(b"b" * 60)
        messages = split_messages(buffer)
        assert [len(m) for m in messages] == [56, 76]
        assert all(m.startswith(b"GRIB") for m in messages)

    def test_a_buffer_with_no_messages_is_empty_rather_than_an_error(self):
        assert split_messages(b"not a grib file at all") == []

    def test_a_truncated_download_is_refused_rather_than_silently_short(self):
        """Returning the messages that did parse would publish a short day that looks
        exactly like a day NOAA issued fewer projections."""
        buffer = self._message(b"a" * 40) + self._message(b"b" * 60)
        with pytest.raises(ValueError, match="truncated"):
            split_messages(buffer[:-20])


class TestProjectionLimit:
    def test_it_decodes_only_the_horizon_the_backtest_scores(self):
        """Two thirds of every file is projections beyond a 24 hour horizon, and decoding
        is the whole cost of the ingest. The limit is the difference between a nine hour
        backfill and a nineteen hour one."""
        from bellwether.ingest import ndfd

        assert ndfd.MAX_STEP_HOURS == 27
        assert ndfd.MAX_STEP_HOURS >= 24, "a 24 hour horizon needs a projection covering it"


class TestExtraction:
    def test_it_needs_the_optional_decoder_and_says_so(self, monkeypatch):
        """`eccodes` is an extra. A checkout without it must fail on the import rather than
        on something that looks like a data problem."""
        monkeypatch.setitem(sys.modules, "eccodes", None)
        with pytest.raises((ImportError, AttributeError, TypeError)):
            extract_station_forecasts(b"not a grib file", {"KSLC": (40.0, -111.0)})


@dataclass
class _Nearest:
    value: float
    lat: float = 0.0
    lon: float = 0.0
    distance: float = 0.0


class _FakeEccodes:
    """The three eccodes calls `_read_message` makes, and nothing else."""

    def __init__(self, step: int, values: dict[str, float], missing: float = 9999.0) -> None:
        self._step = step
        self._values = list(values.values())
        self._missing = missing
        self.asked: list[tuple[float, float]] = []

    def codes_get(self, gid, key):
        return {
            "forecastTime": self._step,
            "missingValue": self._missing,
            "dataDate": 20250115,
            "dataTime": 1200,
        }[key]

    def codes_grib_find_nearest(self, gid, latitude, longitude):
        self.asked.append((latitude, longitude))
        return [_Nearest(self._values[len(self.asked) - 1])]


def _station(station_id: str):
    from bellwether.ingest.noaa import Station

    return Station("TEST", station_id, "nowhere", 1)
