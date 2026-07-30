"""EIA client behaviour, exercised against a mocked API rather than the live service."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from bellwether.ingest.eia import BASE_URL, REGION_DATA_ROUTE, EIAClient, _to_eia_hour

ENDPOINT = f"{BASE_URL}/{REGION_DATA_ROUTE}"
START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 2, tzinfo=UTC)


def _page(rows: list[dict], total: int) -> dict:
    return {"response": {"total": str(total), "data": rows}}


def _row(hour: int, value: float | None = 100.0) -> dict:
    return {
        "period": f"2025-01-01T{hour:02d}",
        "respondent": "CISO",
        "respondent-name": "California Independent System Operator",
        "type": "D",
        "type-name": "Demand",
        "value": value,
        "value-units": "megawatthours",
    }


@pytest.fixture
def client() -> EIAClient:
    # No throttle delay in tests; the live default is exercised separately.
    return EIAClient(api_key="test-key", min_request_interval=0.0)


@respx.mock
def test_api_key_is_sent_as_header_not_query_param(client: EIAClient):
    """The key must never appear in a URL, which could be logged or captured in traces."""
    route = respx.get(ENDPOINT).mock(return_value=httpx.Response(200, json=_page([_row(0)], 1)))

    list(client.fetch_region_data("CISO", "D", START, END))

    request = route.calls[0].request
    assert request.headers["X-Api-Key"] == "test-key"
    assert "test-key" not in str(request.url)
    assert "api_key" not in str(request.url)


@respx.mock
def test_paginates_until_total_is_exhausted(client: EIAClient):
    respx.get(ENDPOINT).mock(
        side_effect=[
            httpx.Response(200, json=_page([_row(h) for h in range(3)], total=5)),
            httpx.Response(200, json=_page([_row(h) for h in range(3, 5)], total=5)),
        ]
    )

    rows = list(client.fetch_region_data("CISO", "D", START, END))
    assert len(rows) == 5
    assert [r.period.hour for r in rows] == [0, 1, 2, 3, 4]


@respx.mock
def test_stops_when_a_page_returns_no_rows(client: EIAClient):
    """A total that overstates available rows must not spin forever."""
    respx.get(ENDPOINT).mock(
        side_effect=[
            httpx.Response(200, json=_page([_row(0)], total=99)),
            httpx.Response(200, json=_page([], total=99)),
        ]
    )

    assert len(list(client.fetch_region_data("CISO", "D", START, END))) == 1


@respx.mock
def test_null_values_are_preserved_not_zero_filled(client: EIAClient):
    """EIA reports genuine gaps as null; coercing them to 0 would corrupt any aggregate."""
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_page([_row(0, value=None)], total=1))
    )

    rows = list(client.fetch_region_data("CISO", "D", START, END))
    assert rows[0].value is None


@respx.mock
def test_periods_are_parsed_as_utc(client: EIAClient):
    respx.get(ENDPOINT).mock(return_value=httpx.Response(200, json=_page([_row(7)], total=1)))

    row = next(iter(client.fetch_region_data("CISO", "D", START, END)))
    assert row.period == datetime(2025, 1, 1, 7, tzinfo=UTC)
    assert row.period.tzinfo is UTC


@respx.mock
def test_retries_transient_server_errors(client: EIAClient):
    respx.get(ENDPOINT).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_page([_row(0)], total=1)),
        ]
    )

    assert len(list(client.fetch_region_data("CISO", "D", START, END))) == 1


def test_rejects_unknown_series_type(client: EIAClient):
    with pytest.raises(ValueError, match="Unknown series type"):
        list(client.fetch_region_data("CISO", "NOT_A_TYPE", START, END))


def test_hour_formatting_converts_to_utc():
    assert _to_eia_hour(datetime(2025, 3, 9, 14, tzinfo=UTC)) == "2025-03-09T14"
    # A naive datetime is assumed UTC rather than silently taking the machine's timezone,
    # which would shift every requested window depending on where the job runs.
    assert _to_eia_hour(datetime(2025, 3, 9, 14)) == "2025-03-09T14"
