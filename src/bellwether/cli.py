"""Command-line entry points."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from bellwether.attribution import (
    DERIVED_DISCLAIMER,
    NOT_AFFILIATED,
    eia_acknowledgment,
    noaa_acknowledgment,
)
from bellwether.config import settings
from bellwether.eval.backtest import rolling_origin_backtest
from bellwether.forecast.baseline import DailySeasonalNaive, WeeklySeasonalNaive
from bellwether.ingest.eia import BALANCING_AUTHORITIES, EIAClient
from bellwether.ingest.noaa import MARKET_STATIONS, NCEIClient, stations_for
from bellwether.ingest.nuclear import (
    MARKETS_WITHOUT_NUCLEAR,
    NUCLEAR_PLANTS,
    facility_ids,
)
from bellwether.storage.db import (
    connect,
    export_snapshot,
    upsert_nuclear_outages,
    upsert_observations,
    upsert_weather_observations,
)
from bellwether.storage.queries import coverage_report, load_series, weather_coverage_report

app = typer.Typer(help="Bellwether: probabilistic grid load forecasting.", no_args_is_help=True)
console = Console()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(
    respondent: str = typer.Option(
        "CISO", help=f"Balancing authority: {list(BALANCING_AUTHORITIES)}"
    ),
    series_type: str = typer.Option("D", help="D=demand, DF=day-ahead forecast, NG=net generation"),
    days: int = typer.Option(730, help="How far back to backfill from now."),
) -> None:
    """Pull hourly observations from EIA into DuckDB."""
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    with EIAClient() as client, connect() as conn:
        rows = client.fetch_region_data(respondent, series_type, start, end)
        written = upsert_observations(conn, rows)
        snapshot = export_snapshot(conn)

    console.print(f"[green]Wrote {written:,} rows[/green] for {respondent}/{series_type}")
    console.print(f"Snapshot: {snapshot}")
    console.print(f"[dim]{eia_acknowledgment()}[/dim]")


@app.command()
def ingest_outages(
    days: int = typer.Option(760, help="How far back to backfill from today."),
) -> None:
    """Pull daily nuclear outage state for the reactors in the tracked markets.

    Faceted by facility because EIA's route has no balancing-authority field. Which plant
    sits in which market is this project's mapping; see `bellwether.ingest.nuclear`.
    """
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)

    with EIAClient() as client, connect() as conn:
        rows = client.fetch_nuclear_outages(facility_ids(), start, end)
        written = upsert_nuclear_outages(conn, rows)
        snapshot = export_snapshot(conn, table="nuclear_outages")

    for plant in NUCLEAR_PLANTS:
        console.print(f"  {plant.name} ({plant.market})")
    for market in MARKETS_WITHOUT_NUCLEAR:
        console.print(f"  [yellow]{market}: no nuclear generation[/yellow]")

    console.print(f"[green]Wrote {written:,} rows[/green] over {start} to {end}")
    console.print(f"Snapshot: {snapshot}")
    console.print(f"[dim]{eia_acknowledgment()}[/dim]")


@app.command()
def ingest_weather(
    respondent: str = typer.Option("CISO", help=f"Balancing authority: {list(MARKET_STATIONS)}"),
    days: int = typer.Option(730, help="How far back to backfill from now."),
) -> None:
    """Pull hourly temperatures for a market's weighted stations from NOAA NCEI.

    NCEI archives lag live weather by months, so a request reaching to the present is
    normal and simply returns nothing past the archive's end.
    """
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    stations = stations_for(respondent)

    total = 0
    with NCEIClient() as client, connect() as conn:
        for station in stations:
            rows = client.fetch_temperatures(station.station_id, start, end)
            written = upsert_weather_observations(conn, rows)
            total += written
            console.print(
                f"  {station.call_sign} ({station.place}): [green]{written:,}[/green] readings"
            )
        snapshot = export_snapshot(conn, table="weather_observations")

    console.print(f"[green]Wrote {total:,} readings[/green] across {len(stations)} stations")
    console.print(f"Snapshot: {snapshot}")
    console.print(f"[dim]{noaa_acknowledgment()}[/dim]")


@app.command()
def status() -> None:
    """Show what is stored, and how complete it is."""
    if not settings.duckdb_path.exists():
        console.print("[yellow]No database yet: run `bellwether ingest` first.[/yellow]")
        raise typer.Exit(code=1)

    with connect(read_only=True) as conn:
        report = coverage_report(conn)
        weather = weather_coverage_report(conn)

    table = Table(title="Stored observations")
    for column in ("Respondent", "Type", "Rows", "First", "Last", "Nulls"):
        table.add_column(column)
    for row in report:
        table.add_row(
            row["respondent"],
            row["series_type"],
            f"{row['rows']:,}",
            str(row["first_period"]),
            str(row["last_period"]),
            f"{row['null_values']:,}",
        )
    console.print(table)
    console.print(f"[dim]{eia_acknowledgment()}[/dim]")

    if not weather:
        return

    # Reverse the registry so the display can name a station rather than show its ISD id.
    stations = {
        station.station_id: (market, station)
        for market, group in MARKET_STATIONS.items()
        for station in group
    }

    weather_table = Table(title="Stored weather")
    for column in ("Market", "Station", "Place", "Hours", "First", "Last", "Suspect"):
        weather_table.add_column(column)
    for row in weather:
        market, station = stations.get(row["station_id"], ("?", None))
        weather_table.add_row(
            market,
            station.call_sign if station else row["station_id"],
            station.place if station else "unknown station",
            f"{row['hours']:,}",
            str(row["first_observed"]),
            str(row["last_observed"]),
            f"{row['suspect_values']:,}",
        )
    console.print(weather_table)
    console.print(f"[dim]{noaa_acknowledgment()}[/dim]")


@app.command()
def backtest(
    respondent: str = typer.Option("CISO"),
    series_type: str = typer.Option("D"),
    horizon: int = typer.Option(24, help="Forecast horizon in hours."),
    max_windows: int = typer.Option(60, help="Cap windows for a fast run; 0 means no cap."),
    chronos: bool = typer.Option(
        False, "--chronos", help="Include Chronos-Bolt. Roughly 0.6s per window on CPU."
    ),
) -> None:
    """Run the rolling-origin backtest across the configured models."""
    with connect(read_only=True) as conn:
        series = load_series(conn, respondent, series_type)

    console.print(
        f"{series.series_id}: {series.values.size:,} hourly points, "
        f"{series.missing_fraction:.2%} missing"
    )

    models: list = [WeeklySeasonalNaive(), DailySeasonalNaive()]
    if chronos:
        from bellwether.forecast.chronos import ChronosBolt

        models.append(ChronosBolt())

    results = []
    for model in models:
        started = time.perf_counter()
        results.append(
            rolling_origin_backtest(
                model,
                series.values,
                series_id=series.series_id,
                horizon=horizon,
                max_windows=max_windows or None,
            )
        )
        console.print(f"[dim]{model.name}: {time.perf_counter() - started:.1f}s[/dim]")

    table = Table(title=f"Backtest: {series.series_id}, h={horizon}")
    for column in ("Model", "Windows", "MASE", "WQL", "sMAPE %", "80% coverage"):
        table.add_column(column)

    for result in results:
        s = result.summary()
        table.add_row(
            result.model_name,
            str(result.n_windows),
            f"{s['mase']:.3f}",
            f"{s['wql']:.4f}",
            f"{s['smape']:.2f}",
            f"{s['coverage_80']:.1%}",
        )
    console.print(table)
    console.print(f"[dim]{eia_acknowledgment()}[/dim]")
    console.print(f"[dim]{NOT_AFFILIATED}[/dim]")
    console.print(f"[dim]{DERIVED_DISCLAIMER}[/dim]")


if __name__ == "__main__":
    app()
