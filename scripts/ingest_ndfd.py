"""Backfill NOAA NDFD forecast temperature for the fourteen weather stations.

Usage:
    python scripts/ingest_ndfd.py --start 2024-07-31 --end 2024-08-31
    python scripts/ingest_ndfd.py --start 2024-07-31 --end 2026-07-31 --issue-hour 12

One issuance per day, the freshest run published at or before `--issue-hour` UTC. Later
runs exist and are deliberately not taken: a run published after the forecast origin it
would be scored at leaks information the forecaster did not have, which is the whole defect
this source exists to remove.

## What it costs

About 5.5 MB per day downloaded and roughly 300 stored rows, so the full two-year history
is about **4 GB over the wire** and a few hundred thousand rows. At the courtesy pace of
one request a second it is bounded by transfer rather than by pacing, and on a domestic
connection it is hours rather than minutes. **Run it in the foreground**: long background
jobs are not reliable on this machine, and this one is resumable, so an interrupted run
costs only the day it was on.

Re-running an already-covered window is safe and cheap in correctness terms, but it is not
free in time, so `--skip-stored` asks the database what it already has and fetches only the
gaps.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bellwether.ingest.ndfd import (  # noqa: E402
    ARCHIVE_START,
    NDFDClient,
    extract_station_forecasts,
    select_issuance,
    stations_with_coordinates,
)
from bellwether.storage.db import connect, upsert_weather_forecasts  # noqa: E402

log = logging.getLogger("ingest_ndfd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_day, help="first UTC day, inclusive")
    parser.add_argument("--end", required=True, type=_day, help="last UTC day, inclusive")
    parser.add_argument(
        "--issue-hour",
        type=int,
        default=12,
        help="take the freshest run published at or before this UTC hour (default 12)",
    )
    parser.add_argument(
        "--skip-stored",
        action="store_true",
        help="skip days that already have rows, so an interrupted backfill resumes",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.start < ARCHIVE_START:
        parser.error(f"the NDFD archive begins {ARCHIVE_START}")
    if args.end < args.start:
        parser.error("--end precedes --start")

    stations = stations_with_coordinates()
    log.info("%s stations, %s to %s", len(stations), args.start, args.end)

    total_rows, fetched, skipped, missing = 0, 0, 0, []
    with connect() as conn, NDFDClient() as client:
        stored = _days_already_stored(conn) if args.skip_stored else set()
        for day in _days(args.start, args.end):
            if day in stored:
                skipped += 1
                continue
            issuances = client.list_issuances(day)
            cutoff = datetime.combine(day, datetime.min.time(), tzinfo=UTC).replace(
                hour=args.issue_hour
            )
            chosen = select_issuance(issuances, cutoff)
            if chosen is None:
                # A day with no run at or before the cutoff is a real gap in the archive,
                # not an error. Recorded and reported rather than silently skipped.
                missing.append(day)
                continue
            rows = extract_station_forecasts(client.fetch(chosen.key), stations)
            written = upsert_weather_forecasts(conn, rows)
            total_rows += written
            fetched += 1
            log.info(
                "%s: run %s, %s rows (%.1f MB)",
                day,
                chosen.issued_at.strftime("%H:%MZ"),
                written,
                chosen.size_bytes / 1e6,
            )

    print(f"\n{fetched} days fetched, {skipped} skipped, {total_rows:,} rows written.")
    if missing:
        print(f"{len(missing)} days had no run at or before the cutoff:")
        for day in missing[:10]:
            print(f"  {day}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    return 0


def _days_already_stored(conn) -> set[date]:
    rows = conn.execute(
        "SELECT DISTINCT CAST(issued_at AS DATE) AS day FROM weather_forecasts"
    ).fetchall()
    return {row[0] for row in rows}


def _days(start: date, end: date):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def _day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


if __name__ == "__main__":
    sys.exit(main())
