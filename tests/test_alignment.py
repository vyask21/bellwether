"""Regression tests for cross-series alignment.

`load_series` derives its grid from the bounds of the one series it loads. Two series
whose first observation differs therefore land on grids offset from each other, while
still having the same length. Comparing them positionally misaligns every value and
produces a plausible-looking but wrong result. This is not hypothetical: it happened,
and it made a grid operator's day-ahead forecast look worse than seasonal-naive.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from bellwether.ingest.eia import ObservationRow
from bellwether.storage.db import connect, upsert_observations
from bellwether.storage.queries import Series, load_aligned_series, load_series

START = datetime(2025, 1, 1, tzinfo=UTC)


def _rows(series_type: str, start_hour: int, count: int, base: float) -> list[ObservationRow]:
    return [
        ObservationRow(
            START + timedelta(hours=start_hour + i),
            "TEST",
            series_type,
            base + i,
            "megawatthours",
        )
        for i in range(count)
    ]


@pytest.fixture
def offset_db(tmp_path):
    """Two series of equal length whose grids start one hour apart."""
    path = tmp_path / "offset.duckdb"
    with connect(path) as conn:
        upsert_observations(conn, _rows("D", start_hour=0, count=10, base=100.0))
        upsert_observations(conn, _rows("DF", start_hour=1, count=10, base=200.0))
    return path


def test_load_series_grids_can_be_offset_despite_equal_length(offset_db):
    """The trap itself: equal lengths, different grids."""
    with connect(offset_db, read_only=True) as conn:
        d = load_series(conn, "TEST", "D")
        df = load_series(conn, "TEST", "DF")

    assert d.values.size == df.values.size, "equal length is what makes this dangerous"
    assert d.timestamps[0] != df.timestamps[0], "grids start an hour apart"


def test_aligned_loader_puts_both_series_on_one_grid(offset_db):
    with connect(offset_db, read_only=True) as conn:
        timestamps, values = load_aligned_series(conn, "TEST", ["D", "DF"])

    assert values["D"].size == timestamps.size
    assert values["DF"].size == timestamps.size
    # Union of both spans: hour 0 through hour 10 inclusive.
    assert timestamps.size == 11


def test_values_land_on_their_own_timestamps(offset_db):
    """The actual correctness property: index i is the same instant in every series."""
    with connect(offset_db, read_only=True) as conn:
        timestamps, values = load_aligned_series(conn, "TEST", ["D", "DF"])

    d, df = values["D"], values["DF"]

    # D covers hours 0-9 starting at 100; DF covers hours 1-10 starting at 200.
    assert d[0] == 100.0
    assert np.isnan(df[0]), "DF has no observation at hour 0"

    assert d[1] == 101.0
    assert df[1] == 200.0, "DF's first value belongs at hour 1, not hour 0"

    assert np.isnan(d[10]), "D has no observation at hour 10"
    assert df[10] == 209.0


def test_missing_hours_are_nan_not_shifted(tmp_path):
    """A gap must leave a hole, not slide every later value one position earlier."""
    path = tmp_path / "gap.duckdb"
    rows = _rows("D", start_hour=0, count=5, base=10.0)
    del rows[2]  # drop hour 2
    with connect(path) as conn:
        upsert_observations(conn, rows)

    with connect(path, read_only=True) as conn:
        timestamps, values = load_aligned_series(conn, "TEST", ["D"])

    d = values["D"]
    assert timestamps.size == 5
    assert np.isnan(d[2])
    assert d[3] == 13.0, "values after a gap must keep their position"


def test_unknown_respondent_is_rejected(tmp_path):
    path = tmp_path / "empty.duckdb"
    with connect(path) as conn:
        upsert_observations(conn, _rows("D", 0, 3, 1.0))

    with connect(path, read_only=True) as conn, pytest.raises(ValueError, match="No observations"):
        load_aligned_series(conn, "NOPE", ["D"])


def test_requires_at_least_one_series_type(tmp_path):
    path = tmp_path / "x.duckdb"
    with connect(path) as conn:
        upsert_observations(conn, _rows("D", 0, 3, 1.0))

    with connect(path, read_only=True) as conn, pytest.raises(ValueError, match="at least one"):
        load_aligned_series(conn, "TEST", [])


@pytest.fixture
def wider_db(tmp_path):
    """One series reaching further in both directions, as NG does against D."""
    path = tmp_path / "wider.duckdb"
    with connect(path) as conn:
        upsert_observations(conn, _rows("D", start_hour=2, count=6, base=100.0))
        upsert_observations(conn, _rows("NG", start_hour=0, count=10, base=200.0))
    return path


def test_clip_to_another_series_span_equalises_the_grid(wider_db):
    """The reason clip exists: EIA publishes each series type on its own schedule."""
    with connect(wider_db, read_only=True) as conn:
        d = load_series(conn, "TEST", "D")
        ng = load_series(conn, "TEST", "NG")

    assert ng.values.size > d.values.size, "NG reaches further before clipping"

    clipped = ng.clip(d.timestamps[0], d.timestamps[-1])

    assert clipped.values.size == d.values.size
    assert clipped.timestamps[0] == d.timestamps[0]
    assert clipped.timestamps[-1] == d.timestamps[-1]


def test_clip_keeps_values_on_their_own_timestamps(wider_db):
    """Clipping must drop whole (timestamp, value) pairs, not slide the values."""
    with connect(wider_db, read_only=True) as conn:
        ng = load_series(conn, "TEST", "NG")

    clipped = ng.clip(ng.timestamps[2], ng.timestamps[5])

    assert clipped.values.size == 4, "both bounds are inclusive"
    assert clipped.values[0] == ng.values[2]
    assert clipped.values[-1] == ng.values[5]


def test_clip_beyond_the_span_is_a_no_op(wider_db):
    with connect(wider_db, read_only=True) as conn:
        ng = load_series(conn, "TEST", "NG")

    widened = ng.clip(ng.timestamps[0] - np.timedelta64(5, "h"), None)

    assert widened.values.size == ng.values.size
    assert widened.series_id == ng.series_id


def test_clip_preserves_gaps_as_nan():
    """A clipped series is still gap-filled, so window skipping still sees the holes."""
    timestamps = np.arange(
        "2025-01-01", "2025-01-02", np.timedelta64(1, "h"), dtype="datetime64[ns]"
    )
    values = np.arange(float(timestamps.size))
    values[5] = np.nan
    series = Series(series_id="TEST:NG", timestamps=timestamps, values=values)

    clipped = series.clip(timestamps[3], timestamps[8])

    assert clipped.values.size == 6
    assert np.isnan(clipped.values[2]), "the NaN at hour 5 lands at index 2"
