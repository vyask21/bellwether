"""Degrading a series to the forecast's cadence, and the no-leak rule behind the loader.

Two properties carry this experiment. The degraded arm has to differ from the observed one
*only* in resolution, or it stops being a control and becomes a third confounded arm. And
the forecast loader must never let a window see a run published after its origin, which is
the same perfect-foresight leak the whole NDFD exercise exists to remove, wearing a
different hat.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bellwether.eval.resolution import degrade_to_cadence, shared_coverage  # noqa: E402
from bellwether.ingest.noaa import stations_for  # noqa: E402
from bellwether.storage.db import SCHEMA  # noqa: E402
from bellwether.storage.queries import load_market_forecast_temperature  # noqa: E402


def _grid(hours: int, start: str = "2025-06-01T00:00") -> np.ndarray:
    return np.datetime64(start, "ns") + np.arange(hours) * np.timedelta64(1, "h")


class TestDegradeToCadence:
    def test_it_keeps_the_stamps_untouched(self):
        """A control that moved the values it kept would differ by more than resolution."""
        timestamps = _grid(48)
        series = np.sin(np.arange(48) / 3.0) * 10 + 20
        degraded = degrade_to_cadence(series, timestamps, cadence_hours=3)
        assert degraded[::3] == pytest.approx(series[::3])

    def test_it_interpolates_between_stamps_rather_than_holding(self):
        timestamps = _grid(7)
        series = np.array([0.0, 99, 99, 3.0, 99, 99, 6.0])  # 99s are dropped by the cadence
        degraded = degrade_to_cadence(series, timestamps, cadence_hours=3)
        assert degraded == pytest.approx([0, 1, 2, 3, 4, 5, 6])

    def test_it_anchors_stamps_to_the_origin_hour_not_the_array_start(self):
        """Anchoring to the array would put the control on a different phase from the
        forecast, and the arms would then differ by half a step of temperature too."""
        timestamps = _grid(9, start="2025-06-01T01:00")
        series = np.arange(9, dtype=float)
        degraded = degrade_to_cadence(series, timestamps, cadence_hours=3, origin_hour=0)
        # Stamps land on 03:00 and 06:00, which are positions 2 and 5 of this array.
        assert np.isnan(degraded[0]) and np.isnan(degraded[1])
        assert degraded[2] == pytest.approx(2.0)
        assert degraded[5] == pytest.approx(5.0)

    def test_it_refuses_to_extrapolate_past_the_last_stamp(self):
        """Continuing the last slope past the end of a forecast is inventing a reading."""
        timestamps = _grid(8)
        series = np.arange(8, dtype=float)
        degraded = degrade_to_cadence(series, timestamps, cadence_hours=3)
        assert np.isfinite(degraded[6]), "the last stamp should survive"
        assert np.isnan(degraded[7:]).all(), "hours past the last stamp are not interpolable"

    def test_a_series_of_gaps_degrades_to_nothing_rather_than_a_guess(self):
        timestamps = _grid(12)
        degraded = degrade_to_cadence(np.full(12, np.nan), timestamps)
        assert np.isnan(degraded).all()

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="differ"):
            degrade_to_cadence(np.zeros(5), _grid(6))


class TestSharedCoverage:
    def test_it_is_the_intersection(self):
        a = np.array([1.0, 2.0, np.nan, 4.0])
        b = np.array([1.0, np.nan, 3.0, 4.0])
        assert shared_coverage(a, b).tolist() == [True, False, False, True]

    def test_one_array_is_its_own_coverage(self):
        a = np.array([1.0, np.nan])
        assert shared_coverage(a).tolist() == [True, False]

    def test_no_arrays_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            shared_coverage()


class TestForecastLoaderRespectsTheOrigin:
    """The loader must serve a window only from runs published before it opened."""

    def _conn(self, rows: list[tuple]) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(":memory:")
        conn.execute(SCHEMA)
        conn.executemany(
            "INSERT INTO weather_forecasts (issued_at, valid_at, station_id, temperature_c)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )
        return conn

    def _run(self, issued: datetime, station: str, value: float, hours=range(0, 40, 3)):
        """One run, flat at `value`, stamped every three hours."""
        return [(issued, issued + timedelta(hours=h), station, value) for h in hours]

    def test_it_uses_the_run_published_before_the_origin(self):
        station = stations_for("CISO")[0].station_id
        # Two runs describing the same window with different values. The window opens at
        # 2025-06-02 00:00 UTC, so only the 06-01 12:00 run was available.
        rows = [
            *self._run(datetime(2025, 6, 1, 12, tzinfo=UTC), station, 10.0),
            *self._run(datetime(2025, 6, 2, 12, tzinfo=UTC), station, 30.0),
        ]
        conn = self._conn(rows)
        timestamps = _grid(24, start="2025-06-02T00:00")
        loaded = load_market_forecast_temperature(conn, "CISO", timestamps)
        assert np.nanmax(loaded) == pytest.approx(10.0), "used a run from the window's future"

    def test_a_window_with_no_earlier_run_is_empty_rather_than_borrowed(self):
        station = stations_for("CISO")[0].station_id
        conn = self._conn(self._run(datetime(2025, 6, 3, 12, tzinfo=UTC), station, 30.0))
        timestamps = _grid(24, start="2025-06-02T00:00")
        assert np.isnan(load_market_forecast_temperature(conn, "CISO", timestamps)).all()

    def test_it_interpolates_the_three_hourly_stamps_to_every_hour(self):
        station = stations_for("CISO")[0].station_id
        issued = datetime(2025, 6, 1, 12, tzinfo=UTC)
        # Ramp by lead so interpolation is checkable: value equals the lead in hours.
        rows = [(issued, issued + timedelta(hours=h), station, float(h)) for h in range(0, 40, 3)]
        conn = self._conn(rows)
        timestamps = _grid(24, start="2025-06-02T00:00")
        loaded = load_market_forecast_temperature(conn, "CISO", timestamps)
        assert np.isfinite(loaded).all(), "a full window should be covered out to lead 36"
        # Hour 0 of the window is lead 12, hour 23 is lead 35.
        assert loaded[0] == pytest.approx(12.0)
        assert loaded[23] == pytest.approx(35.0)

    def test_a_short_run_leaves_the_window_tail_empty(self):
        """The +27 cap covered 6 of 24 hours and this is the assertion that names it."""
        station = stations_for("CISO")[0].station_id
        issued = datetime(2025, 6, 1, 12, tzinfo=UTC)
        rows = [(issued, issued + timedelta(hours=h), station, 20.0) for h in range(0, 28, 3)]
        conn = self._conn(rows)
        timestamps = _grid(24, start="2025-06-02T00:00")
        loaded = load_market_forecast_temperature(conn, "CISO", timestamps)
        assert np.isfinite(loaded[:16]).all()
        assert np.isnan(loaded[16:]).all()
