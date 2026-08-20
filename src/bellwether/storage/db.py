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
from bellwether.config import SNAPSHOT_DIR, STORE_DIR, settings
from bellwether.ingest.eia import ObservationRow, OutageRow
from bellwether.ingest.ndfd import ForecastRow
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

-- NOAA's *forecast* temperature, from NDFD. Agency content like `weather_observations`,
-- but a different claim: what was expected, published at a stated time, rather than what
-- was measured. It is kept apart from both neighbours on purpose. Folding it into
-- `weather_observations` would let a forecast be read back as a measurement, which is the
-- exact leak this project is trying to close by using it at all.
--
-- `issued_at` is in the key rather than merely recorded. The same hour is forecast many
-- times as the runs advance, and which run a model may see depends on its own origin, so
-- a table that kept only the latest could not answer the question the experiment asks.
CREATE TABLE IF NOT EXISTS weather_forecasts (
    issued_at     TIMESTAMPTZ NOT NULL,  -- when NOAA published the run
    valid_at      TIMESTAMPTZ NOT NULL,  -- the hour it describes
    station_id    VARCHAR     NOT NULL,
    temperature_c DOUBLE,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (issued_at, valid_at, station_id)
);

-- EIA content on a different route and a different shape: per generating unit rather than
-- per balancing authority, and daily rather than hourly. Kept out of `observations`
-- because that table's key is (period, respondent, series_type) and a reactor has no
-- respondent. Which market a facility sits in is this project's mapping, not EIA's, so it
-- is applied at read time and never stored here as though the agency had said it.
CREATE TABLE IF NOT EXISTS nuclear_outages (
    period         DATE    NOT NULL,
    facility_id    VARCHAR NOT NULL,
    generator      VARCHAR NOT NULL,
    capacity_mw    DOUBLE,
    outage_mw      DOUBLE,
    percent_outage DOUBLE,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (period, facility_id, generator)
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


def upsert_nuclear_outages(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[OutageRow],
    batch_size: int = 10_000,
) -> int:
    """Insert outage rows idempotently. EIA restates recent days, so a re-run converges."""
    written = 0
    batch: list[tuple] = []

    def flush(records: list[tuple]) -> None:
        if not records:
            return
        conn.executemany(
            """
            INSERT OR REPLACE INTO nuclear_outages
                (period, facility_id, generator, capacity_mw, outage_mw, percent_outage)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    for row in rows:
        batch.append(
            (
                row.period,
                row.facility_id,
                row.generator,
                row.capacity_mw,
                row.outage_mw,
                row.percent_outage,
            )
        )
        if len(batch) >= batch_size:
            flush(batch)
            written += len(batch)
            batch = []

    flush(batch)
    written += len(batch)
    return written


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


def upsert_weather_forecasts(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[ForecastRow],
    batch_size: int = 10_000,
) -> int:
    """Insert forecast temperatures idempotently, replacing any row for the same key.

    A published NDFD run is immutable, so unlike the observation archive this should never
    actually change a stored value. Written as a replace anyway, so re-running an
    interrupted backfill over a window it already covered costs time rather than a
    constraint violation.
    """
    written = 0
    batch: list[tuple] = []

    def flush(records: list[tuple]) -> None:
        if not records:
            return
        conn.executemany(
            """
            INSERT OR REPLACE INTO weather_forecasts
                (issued_at, valid_at, station_id, temperature_c)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )

    for row in rows:
        batch.append((row.issued_at, row.valid_at, row.station_id, row.temperature_c))
        if len(batch) >= batch_size:
            flush(batch)
            written += len(batch)
            batch = []

    flush(batch)
    written += len(batch)
    return written


EXPORTABLE_TABLES = frozenset(
    {
        "observations",
        "weather_observations",
        "weather_forecasts",
        "nuclear_outages",
        "forecasts",
    }
)

# The tables a scheduled run has to be able to rebuild from, in the order they restore.
# `forecasts` is excluded deliberately: it is model output, regenerable from the three
# source tables, and the one table whose contents would grow without bound in a repo.
SOURCE_TABLES = (
    "observations",
    "weather_observations",
    "weather_forecasts",
    "nuclear_outages",
)

# Each table's primary key, used to sort every export.
#
# Without this a snapshot is only as ordered as the table happened to be, and a table
# rebuilt by `restore_table` is not in insertion order at all. Two consequences, both
# discovered by round-tripping rather than by reading: the same 187,084 rows re-exported
# 50% larger, because zstd compresses a clustered timestamp column and not a scattered
# one, and the bytes changed on every cycle. The second is the expensive one. A scheduled
# refresh decides whether to commit by asking git whether anything differs, so an export
# that reshuffles itself would commit a fresh megabyte every week and call it a data
# update. Sorting makes the file a function of the contents alone.
_EXPORT_ORDER = {
    "observations": ("period", "respondent", "series_type"),
    "weather_observations": ("observed_at", "station_id", "report_type"),
    "weather_forecasts": ("issued_at", "valid_at", "station_id"),
    "nuclear_outages": ("period", "facility_id", "generator"),
    "forecasts": ("origin", "period", "respondent", "series_type", "model_name", "quantile"),
}


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
    order = ", ".join(_EXPORT_ORDER[table])
    conn.execute(
        f"COPY (SELECT * FROM {table} ORDER BY {order}) "  # noqa: S608 - table is allowlisted
        f"TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    _write_attribution(conn, out_dir, table)
    return target


def restore_table(
    conn: duckdb.DuckDBPyConnection,
    table: str = "observations",
    directory: Path | None = None,
) -> int:
    """Load a Parquet mirror back into its table, returning the rows read.

    The inverse of `export_snapshot`, and the half that makes an ephemeral runner viable:
    a fresh checkout has no DuckDB file, and this rebuilds one from the committed store in
    seconds rather than re-ingesting two years from two agencies.

    Written as INSERT OR REPLACE against the primary key rather than a table swap, so
    restoring over a store that is already populated converges instead of duplicating, and
    so a restore that races a partial ingest cannot lose the newer row for a key it does
    not carry. `ingested_at` rides along in the file rather than defaulting to now(), which
    is what keeps the exported acknowledgments dated from the ingest and not from the
    restore.

    A missing file is not an error. The first scheduled run has no store to read, and a
    table that was never ingested locally should not stop the other two from restoring.
    """
    if table not in EXPORTABLE_TABLES:
        raise ValueError(
            f"Refusing to restore unknown table {table!r}; expected {EXPORTABLE_TABLES}"
        )

    source = (directory or STORE_DIR) / f"{table}.parquet"
    if not source.exists():
        return 0

    # Columns are named rather than SELECT *: COPY wrote them in table order, but a
    # positional insert would silently misfile every value if a column were ever added to
    # the schema ahead of a stored file being regenerated.
    columns = [
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
    ]
    projection = ", ".join(f'"{column}"' for column in columns)

    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({projection}) "  # noqa: S608 - table is allowlisted
        f"SELECT {projection} FROM read_parquet('{source.as_posix()}')"
    )
    counted = conn.execute(f"SELECT count(*) FROM read_parquet('{source.as_posix()}')").fetchone()[
        0
    ]
    return int(counted)


def restore_store(conn: duckdb.DuckDBPyConnection, directory: Path | None = None) -> dict[str, int]:
    """Rebuild every source table from the committed store."""
    return {table: restore_table(conn, table, directory) for table in SOURCE_TABLES}


def dump_store(conn: duckdb.DuckDBPyConnection, directory: Path | None = None) -> dict[str, Path]:
    """Write every source table to the committed store, attribution alongside."""
    out_dir = directory or STORE_DIR
    return {table: export_snapshot(conn, table, out_dir) for table in SOURCE_TABLES}


def _write_attribution(conn: duckdb.DuckDBPyConnection, out_dir: Path, table: str) -> None:
    """Ship the acknowledgment alongside exported data.

    Each table gets the notice for the agency that published it and no other. A snapshot of
    NOAA temperatures carrying an EIA acknowledgment would credit EIA with data it never
    published, which is exactly what the raw/derived split exists to prevent.

    Both agencies ask acknowledgments to carry a date, so a snapshot is dated from the most
    recent `ingested_at` in the table rather than from wall-clock time at export.
    """
    if table in ("observations", "nuclear_outages"):
        notice = "\n".join([_dated(conn, table, eia_acknowledgment), NOT_AFFILIATED])
    elif table in ("weather_observations", "weather_forecasts"):
        # NDFD joins its NOAA sibling rather than the derived branch below. It is a
        # forecast, but it is NOAA's forecast, and the acknowledgment follows who
        # published the content and not whether the content describes the future.
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
