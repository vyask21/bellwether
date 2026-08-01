"""NCEI client behaviour, exercised against a mocked API rather than the live service."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from bellwether.ingest.noaa import (
    BASE_URL,
    DATA_ROUTE,
    MARKET_STATIONS,
    NCEIClient,
    _chunks,
    _parse_temperature,
    stations_for,
)

ENDPOINT = f"{BASE_URL}/{DATA_ROUTE}"
START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 3, tzinfo=UTC)


def _row(
    minute: int = 53,
    hour: int = 0,
    temperature: str = "+0106",
    quality: str = "1",
    report_type: str = "FM-15",
) -> dict:
    return {
        "DATE": f"2025-01-01T{hour:02d}:{minute:02d}:00",
        "STATION": "72259003927",
        "REPORT_TYPE": report_type,
        "QUALITY_CONTROL": "V020",
        "TMP": f"{temperature},{quality}",
        "SOURCE": "4",
    }


@pytest.fixture
def client() -> NCEIClient:
    # No throttle delay in tests; the live default is exercised separately.
    return NCEIClient(min_request_interval=0.0)


@respx.mock
def test_parses_tenths_of_a_degree_into_celsius(client: NCEIClient):
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, json=[_row(temperature="+0106")]))

    rows = list(client.fetch_temperatures("72259003927", START, END))
    assert rows[0].temperature_c == pytest.approx(10.6)


@respx.mock
def test_parses_negative_temperatures(client: NCEIClient):
    """A sign error here would be invisible in summer and catastrophic in a cold snap."""
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, json=[_row(temperature="-0061")]))

    rows = list(client.fetch_temperatures("72259003927", START, END))
    assert rows[0].temperature_c == pytest.approx(-6.1)


@respx.mock
def test_missing_sentinel_becomes_null_not_a_temperature(client: NCEIClient):
    """+9999 is ISD's missing marker. Read literally it is 999.9 C."""
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(200, json=[_row(temperature="+9999", quality="9")])
    )

    rows = list(client.fetch_temperatures("72259003927", START, END))
    assert rows[0].temperature_c is None
    assert rows[0].quality_code == "9"


@respx.mock
def test_suspect_readings_are_kept_with_their_quality_code(client: NCEIClient):
    """Screening belongs at read time; the stored table mirrors the archive."""
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(200, json=[_row(temperature="+0500", quality="7")])
    )

    rows = list(client.fetch_temperatures("72259003927", START, END))
    assert rows[0].temperature_c == pytest.approx(50.0)
    assert rows[0].quality_code == "7"


@respx.mock
def test_summary_rows_are_dropped(client: NCEIClient):
    """A summary-of-day row carries period aggregates stamped at a bookkeeping time."""
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json=[_row(report_type="SOD", minute=59, hour=5), _row(report_type="FM-15")],
        )
    )

    rows = list(client.fetch_temperatures("72259003927", START, END))
    assert [r.report_type for r in rows] == ["FM-15"]


@respx.mock
def test_timestamps_are_parsed_as_utc(client: NCEIClient):
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, json=[_row(hour=7, minute=53)]))

    row = next(iter(client.fetch_temperatures("72259003927", START, END)))
    assert row.observed_at == datetime(2025, 1, 1, 7, 53, tzinfo=UTC)
    assert row.observed_at.tzinfo is UTC


@respx.mock
def test_an_empty_window_is_not_an_error(client: NCEIClient):
    """Requesting past the archive's end returns an empty list, which is the normal case."""
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, json=[]))

    assert list(client.fetch_temperatures("72259003927", START, END)) == []


@respx.mock
def test_window_is_split_into_chunks(client: NCEIClient):
    chunked = NCEIClient(min_request_interval=0.0, chunk_days=1)
    route = respx.get(ENDPOINT).mock(return_value=httpx.Response(200, json=[]))

    list(chunked.fetch_temperatures("72259003927", START, END))
    assert route.call_count == 2


@respx.mock
def test_retries_transient_server_errors(client: NCEIClient):
    respx.get(ENDPOINT).mock(side_effect=[httpx.Response(500), httpx.Response(200, json=[_row()])])

    assert len(list(client.fetch_temperatures("72259003927", START, END))) == 1


@respx.mock
def test_does_not_retry_a_malformed_request(client: NCEIClient):
    """400 is permanent. Retrying it only delays a clear error by the whole backoff budget."""
    route = respx.get(ENDPOINT).mock(
        return_value=httpx.Response(400, json={"errorCode": 400, "errorMessage": "Bad Request"})
    )

    with pytest.raises(httpx.HTTPStatusError):
        list(client.fetch_temperatures("72259003927", START, END))

    assert route.call_count == 1


@respx.mock
def test_a_malformed_field_is_treated_as_missing_not_fatal(client: NCEIClient):
    """One unparseable record should not abort a backfill of thousands."""
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(200, json=[{**_row(), "TMP": "not-a-number,1"}])
    )

    rows = list(client.fetch_temperatures("72259003927", START, END))
    assert rows[0].temperature_c is None


def test_chunks_cover_the_window_without_gaps_or_overlap():
    chunks = list(_chunks(datetime(2025, 1, 1), datetime(2025, 1, 10), chunk_days=4))
    assert chunks[0][0] == datetime(2025, 1, 1)
    assert chunks[-1][1] == datetime(2025, 1, 10)
    # Deliberately not strict: the pairing is chunk N against chunk N+1, so the second
    # sequence is one shorter by construction.
    for (_, previous_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
        assert previous_end == next_start


def test_chunks_reject_a_nonsense_size():
    with pytest.raises(ValueError, match="at least 1"):
        list(_chunks(datetime(2025, 1, 1), datetime(2025, 1, 2), chunk_days=0))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+0000,1", 0.0),
        ("+0106,1", 10.6),
        ("-0061,5", -6.1),
        ("+0378,1", 37.8),
    ],
)
def test_temperature_parsing(raw: str, expected: float):
    value, _ = _parse_temperature(raw)
    assert value == pytest.approx(expected)


def test_unknown_market_is_rejected_by_name():
    with pytest.raises(ValueError, match="No weather stations configured"):
        stations_for("NOT_A_MARKET")


def test_every_market_has_stations_and_positive_weights():
    for market, stations in MARKET_STATIONS.items():
        assert stations, f"{market} has no stations"
        assert all(s.population > 0 for s in stations), f"{market} has a zero weight"


def test_station_ids_are_unique_across_markets():
    """A station shared between two markets would double-count in an ablation."""
    ids = [s.station_id for group in MARKET_STATIONS.values() for s in group]
    assert len(ids) == len(set(ids))


def test_isd_station_ids_are_well_formed():
    """ISD ids are an 11-digit USAF number concatenated with a WBAN number."""
    for group in MARKET_STATIONS.values():
        for station in group:
            assert station.station_id.isdigit()
            assert len(station.station_id) == 11
