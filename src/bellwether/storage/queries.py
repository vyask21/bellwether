"""Read helpers that turn stored observations into model-ready series."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC

import duckdb
import numpy as np


@dataclass(slots=True)
class Series:
    """A gap-filled hourly series on a regular time grid."""

    series_id: str
    timestamps: np.ndarray  # datetime64[ns]
    values: np.ndarray  # float, NaN where the source reported no value

    @property
    def missing_fraction(self) -> float:
        if self.values.size == 0:
            return 1.0
        return float(np.mean(~np.isfinite(self.values)))


def load_series(
    conn: duckdb.DuckDBPyConnection,
    respondent: str,
    series_type: str = "D",
) -> Series:
    """Load one series onto a continuous hourly grid.

    Missing hours are materialised as NaN rather than dropped. Silently omitting them
    would leave the array irregularly spaced while every seasonal-lag calculation assumes
    a fixed step, quietly shifting lags across any gap.
    """
    rows = conn.execute(
        """
        WITH bounds AS (
            SELECT min(period) AS lo, max(period) AS hi
            FROM observations
            WHERE respondent = ? AND series_type = ?
        ),
        grid AS (
            SELECT unnest(generate_series(lo, hi, INTERVAL 1 HOUR)) AS period
            FROM bounds
        )
        SELECT grid.period, obs.value
        FROM grid
        LEFT JOIN observations AS obs
            ON obs.period = grid.period
           AND obs.respondent = ?
           AND obs.series_type = ?
        ORDER BY grid.period
        """,
        [respondent, series_type, respondent, series_type],
    ).fetchall()

    if not rows:
        raise ValueError(
            f"No observations stored for respondent={respondent!r} series_type={series_type!r}"
        )

    # DuckDB returns TIMESTAMPTZ as tz-aware datetimes, which numpy cannot represent.
    # Normalise to UTC and drop the tzinfo so the conversion is explicit rather than a
    # warning, and so every series sits on the same clock regardless of machine locale.
    timestamps = np.array(
        [r[0].astimezone(UTC).replace(tzinfo=None) for r in rows], dtype="datetime64[ns]"
    )
    values = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=float)

    return Series(series_id=f"{respondent}:{series_type}", timestamps=timestamps, values=values)


def load_aligned_series(
    conn: duckdb.DuckDBPyConnection,
    respondent: str,
    series_types: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load several series types for one respondent onto a single shared time grid.

    Returns the shared timestamps and a value array per series type, all the same length
    and aligned index for index.

    `load_series` derives its grid from the bounds of the series it is loading, so two
    types whose first or last observation differ land on grids offset from each other.
    Comparing those positionally silently misaligns every value. Anything that compares
    two series must use this function, which builds one grid from the union of bounds and
    joins each type onto it by timestamp.
    """
    if not series_types:
        raise ValueError("Need at least one series type")

    placeholders = ", ".join("?" for _ in series_types)
    params: list = [respondent, *series_types]

    grid_rows = conn.execute(
        f"""
        WITH bounds AS (
            SELECT min(period) AS lo, max(period) AS hi
            FROM observations
            WHERE respondent = ? AND series_type IN ({placeholders})
        )
        SELECT unnest(generate_series(lo, hi, INTERVAL 1 HOUR)) AS period
        FROM bounds
        ORDER BY 1
        """,
        params,
    ).fetchall()

    if not grid_rows or grid_rows[0][0] is None:
        raise ValueError(f"No observations for respondent={respondent!r} types={series_types}")

    timestamps = np.array(
        [r[0].astimezone(UTC).replace(tzinfo=None) for r in grid_rows], dtype="datetime64[ns]"
    )
    index = {ts: i for i, ts in enumerate(timestamps)}

    values: dict[str, np.ndarray] = {}
    for series_type in series_types:
        column = np.full(timestamps.size, np.nan, dtype=float)
        rows = conn.execute(
            """
            SELECT period, value
            FROM observations
            WHERE respondent = ? AND series_type = ? AND value IS NOT NULL
            """,
            [respondent, series_type],
        ).fetchall()

        for period, value in rows:
            key = np.datetime64(period.astimezone(UTC).replace(tzinfo=None), "ns")
            position = index.get(key)
            if position is not None:
                column[position] = float(value)
        values[series_type] = column

    return timestamps, values


def coverage_report(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per-series row counts, time span, and null rate: the first thing to check
    after a backfill, before trusting any metric computed on top of it."""
    rows = conn.execute(
        """
        SELECT
            respondent,
            series_type,
            count(*)                                    AS rows,
            min(period)                                 AS first_period,
            max(period)                                 AS last_period,
            sum(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_values
        FROM observations
        GROUP BY respondent, series_type
        ORDER BY respondent, series_type
        """
    ).fetchall()

    return [
        {
            "respondent": r[0],
            "series_type": r[1],
            "rows": r[2],
            "first_period": r[3],
            "last_period": r[4],
            "null_values": r[5],
        }
        for r in rows
    ]
