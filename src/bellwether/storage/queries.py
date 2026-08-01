"""Read helpers that turn stored observations into model-ready series."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC

import duckdb
import numpy as np

from bellwether.ingest.noaa import SUSPECT_QUALITY_CODES, stations_for

log = logging.getLogger(__name__)


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


# Half an hour: the widest a reading can sit from an hour mark once each reading is
# assigned to its nearest hour. Routine METARs land near :53, so most readings are within
# seven minutes of the hour they are credited to.
_NEAREST_HOUR_SQL = "date_trunc('hour', (observed_at AT TIME ZONE 'UTC') + INTERVAL 30 MINUTE)"


def load_station_temperatures(
    conn: duckdb.DuckDBPyConnection,
    station_ids: Sequence[str],
) -> dict[str, dict[np.datetime64, float]]:
    """Collapse raw readings to one temperature per station per hour.

    Stations report a routine observation near :53 plus irregular extras: three-hourly
    synoptic reports on the hour and unscheduled special reports whenever conditions
    change fast. So an hour can hold zero, one, or several readings.

    Each reading is credited to the hour it is nearest in time, and where several land in
    the same hour the closest to the hour mark wins. Rounding rather than truncating
    matters: a reading taken at 00:53 describes 01:00 far better than it describes 00:00,
    and truncating would shift the whole temperature series an hour late against demand.

    Readings NOAA flagged suspect or erroneous are dropped here rather than at ingest, so
    the stored table stays a faithful copy of the archive.
    """
    if not station_ids:
        raise ValueError("Need at least one station")

    placeholders = ", ".join("?" for _ in station_ids)
    rows = conn.execute(
        f"""
        WITH usable AS (
            SELECT
                station_id,
                temperature_c,
                {_NEAREST_HOUR_SQL} AS hour,
                abs(epoch((observed_at AT TIME ZONE 'UTC') - {_NEAREST_HOUR_SQL}))
                    AS distance_seconds
            FROM weather_observations
            WHERE station_id IN ({placeholders})
              AND temperature_c IS NOT NULL
              AND quality_code NOT IN ({", ".join("?" for _ in SUSPECT_QUALITY_CODES)})
        ),
        ranked AS (
            SELECT
                station_id,
                hour,
                temperature_c,
                row_number() OVER (
                    PARTITION BY station_id, hour ORDER BY distance_seconds
                ) AS rank
            FROM usable
        )
        SELECT station_id, hour, temperature_c
        FROM ranked
        WHERE rank = 1
        ORDER BY station_id, hour
        """,
        [*station_ids, *sorted(SUSPECT_QUALITY_CODES)],
    ).fetchall()

    by_station: dict[str, dict[np.datetime64, float]] = {sid: {} for sid in station_ids}
    for station_id, hour, temperature in rows:
        by_station[station_id][np.datetime64(hour, "ns")] = float(temperature)
    return by_station


def load_market_temperature(
    conn: duckdb.DuckDBPyConnection,
    respondent: str,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Population-weighted temperature for one market, on a caller-supplied hourly grid.

    Taking the grid as an argument rather than deriving one is deliberate. The alignment
    bug this project already hit came from two series each building their own grid from
    their own bounds; weather and demand have different bounds by construction, since the
    NCEI archive ends well before EIA's data does. Handing in the demand grid makes the
    weather series align with it by construction rather than by coincidence.

    Weights are renormalised over whichever stations reported in each hour, so a single
    station outage shifts the average toward the remaining cities rather than dragging it
    toward zero. Hours where no station reported come back as NaN.
    """
    stations = stations_for(respondent)
    by_station = load_station_temperatures(conn, [s.station_id for s in stations])

    weighted_sum = np.zeros(timestamps.size, dtype=float)
    weight_total = np.zeros(timestamps.size, dtype=float)

    for station in stations:
        readings = by_station[station.station_id]
        if not readings:
            log.warning("No usable readings stored for %s (%s)", station.call_sign, station.place)
            continue
        present = np.array([hour in readings for hour in timestamps], dtype=bool)
        values = np.array([readings.get(hour, 0.0) for hour in timestamps], dtype=float)
        weighted_sum += np.where(present, values * station.population, 0.0)
        weight_total += np.where(present, float(station.population), 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight_total > 0, weighted_sum / weight_total, np.nan)


def weather_coverage_report(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per-station hourly coverage: the weather counterpart to `coverage_report`."""
    rows = conn.execute(
        f"""
        SELECT
            station_id,
            count(DISTINCT {_NEAREST_HOUR_SQL})                          AS hours,
            min(observed_at)                                             AS first_observed,
            max(observed_at)                                             AS last_observed,
            sum(CASE WHEN temperature_c IS NULL THEN 1 ELSE 0 END)       AS missing_values,
            sum(CASE WHEN quality_code IN ({", ".join("?" for _ in SUSPECT_QUALITY_CODES)})
                     THEN 1 ELSE 0 END)                                  AS suspect_values
        FROM weather_observations
        GROUP BY station_id
        ORDER BY station_id
        """,
        sorted(SUSPECT_QUALITY_CODES),
    ).fetchall()

    return [
        {
            "station_id": r[0],
            "hours": r[1],
            "first_observed": r[2],
            "last_observed": r[3],
            "missing_values": r[4],
            "suspect_values": r[5],
        }
        for r in rows
    ]


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
