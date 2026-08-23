"""Move the DuckDB source tables to and from the committed Parquet store.

Usage: python scripts/sync_store.py dump
       python scripts/sync_store.py restore [--db data/bellwether.duckdb]

## What this is for

DuckDB is a single file that is deliberately not committed: `.gitignore` treats data
artifacts as regenerable. That rule was affordable while every run happened on one laptop.
A scheduled refresh runs on an ephemeral runner with an empty checkout, so "regenerable"
would mean re-ingesting two years from two agencies on every run, which is slow, rude to
the sources, and produces a store nobody can inspect.

The store closes that gap. The four source tables are 2.6 MB as zstd Parquet against the
35 MB DuckDB file, small enough to commit, diff and keep a history of. A refresh restores
from it, tops up the last few days, and writes it back.

## What is and is not in it

`observations` (EIA), `weather_observations` (NOAA ISD) and `weather_forecasts` (NDFD),
which are the three tables nothing can regenerate cheaply: the NDFD backfill alone is 13
hours. `forecasts` is excluded. It is model output rather than agency content, it is
reproduced exactly by re-running the harness, and it is the one table that would grow
without bound in a repository.

The store is a mirror, not a second source of truth. If it disagrees with a live ingest,
the ingest wins: every restore is an upsert on the primary key, so the newer row survives.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bellwether.config import STORE_DIR
from bellwether.storage.db import SOURCE_TABLES, connect, dump_store, restore_store

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dump", "restore"))
    parser.add_argument("--store", default=str(STORE_DIR), help="Where the Parquet mirror lives.")
    parser.add_argument("--db", default=None, help="DuckDB path; defaults to the configured one.")
    args = parser.parse_args()

    store = Path(args.store)
    db_path = Path(args.db) if args.db else None

    if args.action == "dump":
        with connect(db_path) as conn:
            written = dump_store(conn, store)
        for table, path in written.items():
            rows = _count(path)
            print(f"  {table:<22} {rows:>9,} rows  {path.stat().st_size / 1024:>7.0f} KB")
        print(f"Dumped {len(written)} tables to {store}")
        return

    with connect(db_path) as conn:
        restored = restore_store(conn, store)
    for table in SOURCE_TABLES:
        rows = restored[table]
        note = "" if rows else "  (no file in store)"
        print(f"  {table:<22} {rows:>9,} rows{note}")
    print(f"Restored into {db_path or 'the configured database'} from {store}")


def _count(path: Path) -> int:
    """Row count straight from the file, so the printed figure describes what was written."""
    import duckdb

    return int(duckdb.sql(f"SELECT count(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0])


if __name__ == "__main__":
    main()
