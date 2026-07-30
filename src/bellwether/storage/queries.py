"""Read helpers that turn stored observations into model-ready series."""

from __future__ import annotations

from dataclasses import dataclass

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

    timestamps = np.array([r[0] for r in rows], dtype="datetime64[ns]")
    values = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=float)

    return Series(series_id=f"{respondent}:{series_type}", timestamps=timestamps, values=values)


def coverage_report(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per-series row counts, time span, and null rate — the first thing to check
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
