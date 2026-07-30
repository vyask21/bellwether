"""DuckDB connection management and schema.

DuckDB is single-writer: the ingest job writes, and anything that reads concurrently
(dashboards, notebooks) should open read-only or read an exported Parquet snapshot.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from bellwether.config import SNAPSHOT_DIR, settings
from bellwether.ingest.eia import ObservationRow

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    period       TIMESTAMPTZ NOT NULL,
    respondent   VARCHAR     NOT NULL,
    series_type  VARCHAR     NOT NULL,
    value        DOUBLE,
    value_units  VARCHAR,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (period, respondent, series_type)
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


def export_snapshot(conn: duckdb.DuckDBPyConnection, name: str = "observations") -> Path:
    """Write a Parquet snapshot readers can use without contending for the write lock."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    target = SNAPSHOT_DIR / f"{name}.parquet"
    conn.execute(f"COPY observations TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    return target
