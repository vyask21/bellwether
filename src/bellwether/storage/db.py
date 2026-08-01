"""DuckDB connection management and schema.

DuckDB is single-writer: the ingest job writes, and anything that reads concurrently
(dashboards, notebooks) should open read-only or read an exported Parquet snapshot.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import duckdb

from bellwether.attribution import (
    DERIVED_DISCLAIMER,
    NOAA_NOT_AFFILIATED,
    NOT_AFFILIATED,
    eia_acknowledgment,
    noaa_acknowledgment,
)
from bellwether.config import SNAPSHOT_DIR, settings
from bellwether.ingest.eia import ObservationRow
from bellwether.ingest.noaa import WeatherRow

SCHEMA = """
-- Raw EIA content, stored exactly as returned. The API Terms of Service forbid modifying
-- content and still claiming EIA as the source, so nothing derived is ever written here:
-- no imputation, no gap filling, no rescaling. Model output goes in `forecasts`.
CREATE TABLE IF NOT EXISTS observations (
    period       TIMESTAMPTZ NOT NULL,
    respondent   VARCHAR     NOT NULL,
    series_type  VARCHAR     NOT NULL,
    value        DOUBLE,
    value_units  VARCHAR,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (period, respondent, series_type)
);

-- Raw NOAA content, stored as returned. Separate from `observations` because it is a
-- different agency under a different policy, and because conflating the two would make
-- the EIA attribution attach to values EIA never published.
--
-- Values NOAA flagged suspect are stored with their quality code rather than dropped, so
-- the table stays faithful to the source. Screening happens at read time.
CREATE TABLE IF NOT EXISTS weather_observations (
    observed_at   TIMESTAMPTZ NOT NULL,
    station_id    VARCHAR     NOT NULL,
    report_type   VARCHAR     NOT NULL,
    temperature_c DOUBLE,
    quality_code  VARCHAR,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (observed_at, station_id, report_type)
);

-- Derived data. Not EIA content and never attributed to EIA. Kept in its own table so a
-- forecast can never be read back as an observation.
CREATE TABLE IF NOT EXISTS forecasts (
    origin       TIMESTAMPTZ NOT NULL,  -- when the forecast was made
    period       TIMESTAMPTZ NOT NULL,  -- the hour being forecast
    respondent   VARCHAR     NOT NULL,
    series_type  VARCHAR     NOT NULL,
    model_name   VARCHAR     NOT NULL,
    quantile     DOUBLE      NOT NULL,
    value        DOUBLE      NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (origin, period, respondent, series_type, model_name, quantile)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id       BIGINT PRIMARY KEY,
    source       VARCHAR     NOT NULL,
    respondent   VARCHAR,
    series_type  VARCHAR,
    window_start TIMESTAMPTZ,
    window_end   TIMESTAMPTZ,
    rows_written BIGINT,
    started_at   TIMESTAMPTZ NOT NULL,
    finished_at  TIMESTAMPTZ
);

CREATE SEQUENCE IF NOT EXISTS ingest_run_id_seq START 1;
"""


@contextmanager
def connect(
    path: Path | None = None, read_only: bool = False
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a DuckDB connection, creating the schema on first write."""
    db_path = path or settings.duckdb_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path), read_only=read_only)
    try:
        if not read_only:
            conn.execute(SCHEMA)
        yield conn
    finally:
        conn.close()


def upsert_observations(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[ObservationRow],
    batch_size: int = 10_000,
) -> int:
    """Insert observations idempotently, replacing any existing row for the same key.

    Backfills and live pulls overlap by design, and EIA restates recent values, so a
    re-run must converge on the latest published number rather than duplicate or skip.
    """
    written = 0
    batch: list[tuple] = []

    def flush(records: list[tuple]) -> None:
        if not records:
            return
        conn.executemany(
            """
            INSERT OR REPLACE INTO observations
                (period, respondent, series_type, value, value_units)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )

    for row in rows:
        batch.append((row.period, row.respondent, row.series_type, row.value, row.value_units))
        if len(batch) >= batch_size:
            flush(batch)
            written += len(batch)
            batch = []

    flush(batch)
    written += len(batch)
    return written


def upsert_weather_observations(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[WeatherRow],
    batch_size: int = 10_000,
) -> int:
    """Insert weather readings idempotently, replacing any existing row for the same key.

    NCEI revises the archive as late reports arrive and quality control runs, so a re-run
    must converge on the current published value rather than duplicate it.
    """
    written = 0
    batch: list[tuple] = []

    def flush(records: list[tuple]) -> None:
        if not records:
            return
        conn.executemany(
            """
            INSERT OR REPLACE INTO weather_observations
                (observed_at, station_id, report_type, temperature_c, quality_code)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )

    for row in rows:
        batch.append(
            (
                row.observed_at,
                row.station_id,
                row.report_type,
                row.temperature_c,
                row.quality_code,
            )
        )
        if len(batch) >= batch_size:
            flush(batch)
            written += len(batch)
            batch = []

    flush(batch)
    written += len(batch)
    return written


EXPORTABLE_TABLES = frozenset({"observations", "weather_observations", "forecasts"})


def export_snapshot(
    conn: duckdb.DuckDBPyConnection,
    table: str = "observations",
    directory: Path | None = None,
) -> Path:
    """Write a Parquet snapshot readers can use without contending for the write lock.

    The table name is checked against an allowlist because DuckDB cannot parameterise an
    identifier, so it has to be interpolated into the statement.

    Exported EIA content leaves the database with an attribution file alongside it, since
    the TOS attribution requirement follows the content rather than the process.
    """
    if table not in EXPORTABLE_TABLES:
        raise ValueError(
            f"Refusing to export unknown table {table!r}; expected {EXPORTABLE_TABLES}"
        )

    out_dir = directory or SNAPSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{table}.parquet"
    conn.execute(f"COPY {table} TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    _write_attribution(conn, out_dir, table)
    return target


def _write_attribution(conn: duckdb.DuckDBPyConnection, out_dir: Path, table: str) -> None:
    """Ship the acknowledgment alongside exported data.

    Each table gets the notice for the agency that published it and no other. A snapshot of
    NOAA temperatures carrying an EIA acknowledgment would credit EIA with data it never
    published, which is exactly what the raw/derived split exists to prevent.

    Both agencies ask acknowledgments to carry a date, so a snapshot is dated from the most
    recent `ingested_at` in the table rather than from wall-clock time at export.
    """
    if table == "observations":
        notice = "\n".join([_dated(conn, table, eia_acknowledgment), NOT_AFFILIATED])
    elif table == "weather_observations":
        notice = "\n".join([_dated(conn, table, noaa_acknowledgment), NOAA_NOT_AFFILIATED])
    else:
        notice = "\n".join(
            [
                eia_acknowledgment(),
                NOT_AFFILIATED,
                noaa_acknowledgment(),
                NOAA_NOT_AFFILIATED,
                DERIVED_DISCLAIMER,
            ]
        )

    (out_dir / f"ATTRIBUTION-{table}.txt").write_text(f"{notice}\n", encoding="utf-8")


def _dated(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    acknowledgment: Callable[[date | None], str],
) -> str:
    """Build an acknowledgment dated from the table's most recent ingest.

    The date is read in SQL rather than as a Python datetime: handing a TIMESTAMPTZ back
    makes DuckDB import pytz, and only the date is needed.
    """
    stamp = conn.execute(
        f"SELECT max(ingested_at)::DATE FROM {table}"  # noqa: S608 - table is allowlisted
    ).fetchone()[0]
    return acknowledgment(stamp) if stamp else acknowledgment(None)
