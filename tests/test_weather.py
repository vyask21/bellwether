"""Gridding and weighting of NOAA readings onto the demand timeline.

Weather enters the project through the same door the alignment bug came in: two series
with different bounds, compared positionally. NCEI's archive ends well before EIA's data
does, so the bounds differ by construction rather than by accident, and every test here
exists because getting this wrong produces a plausible number rather than an error.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from bellwether.ingest.noaa import MARKET_STATIONS, Station, WeatherRow
from bellwether.storage.db import connect, upsert_weather_observations
from bellwether.storage.queries import (
    load_market_temperature,
    load_station_temperatures,
    weather_coverage_report,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
CISO_STATIONS = MARKET_STATIONS["CISO"]


def _reading(
    offset: timedelta,
    station_id: str = "72288023152",
    temperature: float | None = 10.0,
    quality: str = "1",
    report_type: str = "FM-15",
) -> WeatherRow:
    return WeatherRow(
        observed_at=START + offset,
        station_id=station_id,
        report_type=report_type,
        temperature_c=temperature,
        quality_code=quality,
    )


def _grid(hours: int, start: datetime = START) -> np.ndarray:
    return np.array(
        [np.datetime64(start.replace(tzinfo=None) + timedelta(hours=h), "ns") for h in range(hours)]
    )


def _db(tmp_path, rows: list[WeatherRow]):
    path = tmp_path / "weather.duckdb"
    with connect(path) as conn:
        upsert_weather_observations(conn, rows)
    return path


class TestHourlyGridding:
    def test_a_reading_is_credited_to_its_nearest_hour(self, tmp_path):
        """00:53 describes 01:00, not 00:00.

        Truncating instead of rounding would shift the entire temperature series an hour
        late against demand, which is exactly the kind of offset that still produces a
        result rather than an error.
        """
        path = _db(tmp_path, [_reading(timedelta(minutes=53), temperature=12.5)])

        with connect(path, read_only=True) as conn:
            readings = load_station_temperatures(conn, ["72288023152"])

        hours = list(readings["72288023152"])
        assert hours == [np.datetime64(datetime(2025, 1, 1, 1), "ns")]

    def test_a_reading_on_the_hour_stays_on_that_hour(self, tmp_path):
        path = _db(tmp_path, [_reading(timedelta(0), temperature=12.5)])

        with connect(path, read_only=True) as conn:
            readings = load_station_temperatures(conn, ["72288023152"])

        assert list(readings["72288023152"]) == [np.datetime64(datetime(2025, 1, 1, 0), "ns")]

    def test_the_reading_closest_to_the_hour_wins(self, tmp_path):
        """Special reports fire off-schedule, so an hour can hold several readings."""
        path = _db(
            tmp_path,
            [
                # 00:53 is 7 minutes from 01:00; 01:20 is 20 minutes from it.
                _reading(timedelta(minutes=53), temperature=10.0),
                _reading(timedelta(minutes=80), temperature=20.0, report_type="FM-16"),
            ],
        )

        with connect(path, read_only=True) as conn:
            readings = load_station_temperatures(conn, ["72288023152"])

        assert readings["72288023152"][np.datetime64(datetime(2025, 1, 1, 1), "ns")] == 10.0

    def test_suspect_readings_are_screened_out(self, tmp_path):
        """Stored faithfully, dropped on read. Code 7 is 'erroneous' in the ISD table."""
        path = _db(
            tmp_path,
            [
                _reading(timedelta(minutes=53), temperature=999.0, quality="7"),
                _reading(timedelta(hours=1, minutes=53), temperature=11.0, quality="1"),
            ],
        )

        with connect(path, read_only=True) as conn:
            readings = load_station_temperatures(conn, ["72288023152"])

        assert list(readings["72288023152"].values()) == [11.0]

    def test_missing_temperatures_are_skipped(self, tmp_path):
        path = _db(tmp_path, [_reading(timedelta(minutes=53), temperature=None, quality="9")])

        with connect(path, read_only=True) as conn:
            readings = load_station_temperatures(conn, ["72288023152"])

        assert readings["72288023152"] == {}

    def test_a_station_with_no_rows_returns_an_empty_mapping(self, tmp_path):
        """Absent, not missing from the result: callers index by station id."""
        path = _db(tmp_path, [_reading(timedelta(minutes=53))])

        with connect(path, read_only=True) as conn:
            readings = load_station_temperatures(conn, ["72288023152", "72290023188"])

        assert readings["72290023188"] == {}

    def test_requires_at_least_one_station(self, tmp_path):
        path = _db(tmp_path, [_reading(timedelta(minutes=53))])

        with connect(path, read_only=True) as conn, pytest.raises(ValueError, match="at least one"):
            load_station_temperatures(conn, [])


class TestMarketTemperature:
    def test_weights_are_population_proportional(self, tmp_path):
        """Two stations, known weights, so the answer is checkable by hand."""
        big, small = CISO_STATIONS[0], CISO_STATIONS[5]
        path = _db(
            tmp_path,
            [
                _reading(timedelta(0), station_id=big.station_id, temperature=10.0),
                _reading(timedelta(0), station_id=small.station_id, temperature=20.0),
            ],
        )

        with connect(path, read_only=True) as conn:
            values = load_market_temperature(conn, "CISO", _grid(1))

        expected = (10.0 * big.population + 20.0 * small.population) / (
            big.population + small.population
        )
        assert values[0] == pytest.approx(expected)

    def test_weights_renormalise_when_a_station_is_missing(self, tmp_path):
        """A station outage must move the average, not dilute it toward zero.

        Dividing by the full weight total instead of the reporting total would drag the
        market temperature toward 0 C in proportion to the missing city's size, which in
        winter is a plausible-looking number.
        """
        big, small = CISO_STATIONS[0], CISO_STATIONS[5]
        path = _db(
            tmp_path,
            [
                _reading(timedelta(0), station_id=big.station_id, temperature=10.0),
                _reading(timedelta(0), station_id=small.station_id, temperature=20.0),
                # Second hour: only the small station reports.
                _reading(timedelta(hours=1), station_id=small.station_id, temperature=20.0),
            ],
        )

        with connect(path, read_only=True) as conn:
            values = load_market_temperature(conn, "CISO", _grid(2))

        assert values[1] == pytest.approx(20.0)

    def test_an_hour_with_no_station_reporting_is_nan(self, tmp_path):
        """A gap must stay a gap: the backtest skips windows containing one."""
        path = _db(tmp_path, [_reading(timedelta(0), station_id=CISO_STATIONS[0].station_id)])

        with connect(path, read_only=True) as conn:
            values = load_market_temperature(conn, "CISO", _grid(3))

        assert np.isfinite(values[0])
        assert not np.isfinite(values[1])
        assert not np.isfinite(values[2])

    def test_the_result_matches_the_supplied_grid_length(self, tmp_path):
        path = _db(tmp_path, [_reading(timedelta(0), station_id=CISO_STATIONS[0].station_id)])

        with connect(path, read_only=True) as conn:
            assert load_market_temperature(conn, "CISO", _grid(48)).size == 48

    def test_hours_past_the_archive_end_are_nan_not_carried_forward(self, tmp_path):
        """NCEI's archive ends months before EIA's data does.

        Holding the last known temperature across that gap would hand the model a
        confident wrong covariate for every hour after the archive stops.
        """
        station = CISO_STATIONS[0]
        path = _db(
            tmp_path,
            [
                _reading(timedelta(hours=h), station_id=station.station_id, temperature=15.0)
                for h in range(3)
            ],
        )

        with connect(path, read_only=True) as conn:
            values = load_market_temperature(conn, "CISO", _grid(6))

        assert np.all(np.isfinite(values[:3]))
        assert np.all(~np.isfinite(values[3:]))

    def test_a_grid_starting_before_any_reading_is_not_shifted(self, tmp_path):
        """The weather series is positioned by timestamp, never by offset from its own start.

        This is the alignment trap in its weather form: weather and demand have different
        bounds by construction, so a grid built from weather's own bounds would slide the
        whole series against demand while keeping the same length.
        """
        station = CISO_STATIONS[0]
        path = _db(
            tmp_path,
            [
                _reading(
                    timedelta(hours=24 + h), station_id=station.station_id, temperature=15.0 + h
                )
                for h in range(3)
            ],
        )

        with connect(path, read_only=True) as conn:
            values = load_market_temperature(conn, "CISO", _grid(30))

        assert np.all(~np.isfinite(values[:24]))
        assert values[24] == pytest.approx(15.0)
        assert values[26] == pytest.approx(17.0)

    def test_unknown_market_is_rejected(self, tmp_path):
        path = _db(tmp_path, [_reading(timedelta(0))])

        with connect(path, read_only=True) as conn, pytest.raises(ValueError, match="configured"):
            load_market_temperature(conn, "NOT_A_MARKET", _grid(1))


class TestWeatherCoverageReport:
    def test_counts_hours_and_flags_suspect_readings(self, tmp_path):
        path = _db(
            tmp_path,
            [
                _reading(timedelta(minutes=53), temperature=10.0),
                _reading(timedelta(hours=1, minutes=53), temperature=11.0, quality="7"),
                _reading(timedelta(hours=2, minutes=53), temperature=None, quality="9"),
            ],
        )

        with connect(path, read_only=True) as conn:
            report = weather_coverage_report(conn)

        assert len(report) == 1
        assert report[0]["hours"] == 3
        assert report[0]["suspect_values"] == 1
        assert report[0]["missing_values"] == 1


class TestStationRegistry:
    def test_market_weights_are_dominated_by_no_single_tiny_station(self):
        """A sanity floor on the registry: no market should hinge on a rounding error."""
        for market, stations in MARKET_STATIONS.items():
            total = sum(s.population for s in stations)
            assert total > 0, market
            assert max(s.population for s in stations) / total <= 0.95, (
                f"{market} is effectively a single-station market"
            )

    def test_a_station_is_a_frozen_record(self):
        station = Station("KXXX", "12345678901", "Somewhere", 1000)
        with pytest.raises(AttributeError):
            station.population = 2000  # type: ignore[misc]
