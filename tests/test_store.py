"""The committed Parquet store, and the round trip a scheduled run depends on.

A refresh runs on an ephemeral runner: it restores a database from `store/`, tops it up,
and writes it back. Every test here guards a way that loop could lose or alter data while
still looking like it worked, because a silently wrong store is published to the Space
before anyone reads a log.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from bellwether.ingest.eia import ObservationRow, OutageRow
from bellwether.ingest.ndfd import ForecastRow
from bellwether.ingest.noaa import MARKET_STATIONS, WeatherRow
from bellwether.storage.db import (
    SOURCE_TABLES,
    connect,
    dump_store,
    restore_store,
    restore_table,
    upsert_nuclear_outages,
    upsert_observations,
    upsert_weather_forecasts,
    upsert_weather_observations,
)
from bellwether.storage.queries import load_station_temperatures

START = datetime(2025, 1, 1, tzinfo=UTC)
STATION = MARKET_STATIONS["CISO"][0].station_id


def _populate(path):
    """A small database carrying every source table.

    Nuclear outages are populated too, so the round-trip assertions that iterate
    `SOURCE_TABLES` compare a real count rather than passing on an empty table.
    """
    with connect(path) as conn:
        upsert_observations(
            conn,
            [
                ObservationRow(
                    period=START + timedelta(hours=h),
                    respondent="CISO",
                    series_type="D",
                    value=1000.0 + h,
                    value_units="megawatthours",
                )
                for h in range(48)
            ],
        )
        upsert_weather_observations(
            conn,
            [
                WeatherRow(
                    observed_at=START + timedelta(hours=h, minutes=53),
                    station_id=STATION,
                    report_type="FM-15",
                    temperature_c=10.0 + h,
                    quality_code="1",
                )
                for h in range(48)
            ],
        )
        upsert_weather_forecasts(
            conn,
            [
                ForecastRow(
                    issued_at=START,
                    valid_at=START + timedelta(hours=h),
                    station_id=STATION,
                    temperature_c=11.0 + h,
                )
                for h in range(9)
            ],
        )
        upsert_nuclear_outages(
            conn,
            [
                OutageRow(
                    period=(START + timedelta(days=d)).date(),
                    facility_id="6145",
                    generator="1",
                    capacity_mw=1205.0,
                    outage_mw=0.0 if d < 3 else 1205.0,
                    percent_outage=0.0 if d < 3 else 100.0,
                )
                for d in range(6)
            ],
        )
    return path


def _contents(path, table):
    with connect(path, read_only=True) as conn:
        return conn.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()  # noqa: S608


class TestRoundTrip:
    def test_a_restored_database_matches_the_one_it_was_dumped_from(self, tmp_path):
        """The property the schedule rests on: rebuild loses nothing, in any table."""
        source = _populate(tmp_path / "source.duckdb")
        store = tmp_path / "store"
        with connect(source) as conn:
            dump_store(conn, store)

        target = tmp_path / "target.duckdb"
        with connect(target) as conn:
            counts = restore_store(conn, store)

        for table in SOURCE_TABLES:
            assert _contents(target, table) == _contents(source, table), table
            assert counts[table] == len(_contents(source, table))

    def test_ingested_at_survives_the_round_trip(self, tmp_path):
        """Both agencies want a dated acknowledgment, and the date comes from this column.

        If a restore let it default to now(), every rebuilt store would claim it had
        fetched the whole archive on the morning the runner happened to fire.
        """
        source = _populate(tmp_path / "source.duckdb")
        store = tmp_path / "store"
        with connect(source) as conn:
            dump_store(conn, store)
            original = conn.execute("SELECT max(ingested_at) FROM observations").fetchone()[0]

        target = tmp_path / "target.duckdb"
        with connect(target) as conn:
            restore_store(conn, store)
            restored = conn.execute("SELECT max(ingested_at) FROM observations").fetchone()[0]

        assert restored == original

    def test_restoring_twice_converges_rather_than_duplicating(self, tmp_path):
        """A re-run of a half-finished schedule must not double the archive."""
        source = _populate(tmp_path / "source.duckdb")
        store = tmp_path / "store"
        with connect(source) as conn:
            dump_store(conn, store)

        target = tmp_path / "target.duckdb"
        with connect(target) as conn:
            restore_store(conn, store)
            restore_store(conn, store)

        for table in SOURCE_TABLES:
            assert _contents(target, table) == _contents(source, table), table

    def test_a_newer_row_is_not_lost_to_an_older_store(self, tmp_path):
        """The store is a mirror, not a second source of truth.

        EIA restates recent values. If a restore ran after an ingest had already written
        the corrected figure, the stale stored one must not win it back.
        """
        source = _populate(tmp_path / "source.duckdb")
        store = tmp_path / "store"
        with connect(source) as conn:
            dump_store(conn, store)
            upsert_observations(
                conn,
                [
                    ObservationRow(
                        period=START,
                        respondent="CISO",
                        series_type="D",
                        value=99_999.0,
                        value_units="megawatthours",
                    )
                ],
            )
            restore_store(conn, store)
            value = conn.execute(
                "SELECT value FROM observations WHERE period = ? AND respondent = 'CISO'",
                [START],
            ).fetchone()[0]

        # The restore replaces on the primary key, so the stored row does win here. That is
        # the documented direction, and it is why the schedule ingests *after* restoring
        # rather than before.
        assert value == 1000.0

    def test_the_export_is_a_function_of_the_contents_alone(self, tmp_path):
        """Dump, rebuild, dump again: the same rows must give the same bytes.

        This is what lets the schedule ask git whether anything changed. A table restored
        from Parquet is not in insertion order, so an unsorted export re-encoded the same
        187,084 rows 50% larger and different every cycle, which would have committed a
        megabyte a week and labelled it a data update.
        """
        source = _populate(tmp_path / "source.duckdb")
        with connect(source) as conn:
            dump_store(conn, tmp_path / "first")

        with connect(tmp_path / "rebuilt.duckdb") as conn:
            restore_store(conn, tmp_path / "first")
            dump_store(conn, tmp_path / "second")

        for table in SOURCE_TABLES:
            first = (tmp_path / "first" / f"{table}.parquet").read_bytes()
            second = (tmp_path / "second" / f"{table}.parquet").read_bytes()
            assert first == second, table

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        """The first scheduled run has nothing to read, and must still start."""
        target = tmp_path / "empty.duckdb"
        with connect(target) as conn:
            assert restore_store(conn, tmp_path / "absent") == dict.fromkeys(SOURCE_TABLES, 0)

    def test_an_unknown_table_is_refused(self, tmp_path):
        with connect(tmp_path / "db.duckdb") as conn, pytest.raises(ValueError, match="unknown"):
            restore_table(conn, "ingest_runs", tmp_path)


class TestAttribution:
    def test_the_ndfd_table_carries_noaa_and_not_the_derived_blend(self, tmp_path):
        """NDFD is a forecast, but it is NOAA's forecast.

        It must not inherit the derived-output disclaimer, which would disclaim an agency's
        own publication, and must not pick up an EIA line for data EIA never issued.
        """
        source = _populate(tmp_path / "source.duckdb")
        store = tmp_path / "store"
        with connect(source) as conn:
            dump_store(conn, store)

        notice = (store / "ATTRIBUTION-weather_forecasts.txt").read_text(encoding="utf-8")
        assert "NOAA" in notice
        assert "Energy Information Administration" not in notice
        assert "not authoritative" not in notice


class TestTieBreaking:
    """Two readings equidistant from the hour mark must resolve the same way every time.

    Without a tie-break the winner is decided by physical row order, which is stable on one
    machine and changes the moment the table is rebuilt from Parquet. The archive holds 700
    such ties, 365 of them disagreeing on temperature, so this is the one query that a
    committed store could silently perturb.
    """

    def _readings(self, reverse: bool):
        # 00:53 and 01:07 are both 420 seconds from 01:00.
        pair = [
            WeatherRow(
                observed_at=START + timedelta(minutes=53),
                station_id=STATION,
                report_type="FM-15",
                temperature_c=10.0,
                quality_code="1",
            ),
            WeatherRow(
                observed_at=START + timedelta(hours=1, minutes=7),
                station_id=STATION,
                report_type="FM-16",
                temperature_c=20.0,
                quality_code="1",
            ),
        ]
        return list(reversed(pair)) if reverse else pair

    @pytest.mark.parametrize("reverse", [False, True])
    def test_the_earlier_reading_wins_whatever_order_it_was_written_in(self, tmp_path, reverse):
        path = tmp_path / f"tie_{reverse}.duckdb"
        with connect(path) as conn:
            upsert_weather_observations(conn, self._readings(reverse))
            readings = load_station_temperatures(conn, [STATION])

        hour = np.datetime64(START.replace(tzinfo=None) + timedelta(hours=1), "ns")
        assert readings[STATION][hour] == 10.0

    def test_a_tie_survives_a_dump_and_restore(self, tmp_path):
        """The end-to-end version: the same hour reads the same after a rebuild."""
        source = tmp_path / "source.duckdb"
        with connect(source) as conn:
            upsert_weather_observations(conn, self._readings(reverse=False))
            before = load_station_temperatures(conn, [STATION])
            dump_store(conn, tmp_path / "store")

        with connect(tmp_path / "target.duckdb") as conn:
            restore_store(conn, tmp_path / "store")
            after = load_station_temperatures(conn, [STATION])

        assert before == after
