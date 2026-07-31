"""Command-line entry points."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from bellwether.attribution import DERIVED_DISCLAIMER, NOT_AFFILIATED, eia_acknowledgment
from bellwether.config import settings
from bellwether.eval.backtest import rolling_origin_backtest
from bellwether.forecast.baseline import DailySeasonalNaive, WeeklySeasonalNaive
from bellwether.ingest.eia import BALANCING_AUTHORITIES, EIAClient
from bellwether.storage.db import connect, export_snapshot, upsert_observations
from bellwether.storage.queries import coverage_report, load_series

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
def status() -> None:
    """Show what is stored, and how complete it is."""
    if not settings.duckdb_path.exists():
        console.print("[yellow]No database yet: run `bellwether ingest` first.[/yellow]")
        raise typer.Exit(code=1)

    with connect(read_only=True) as conn:
        report = coverage_report(conn)

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


@app.command()
def backtest(
    respondent: str = typer.Option("CISO"),
    series_type: str = typer.Option("D"),
    horizon: int = typer.Option(24, help="Forecast horizon in hours."),
    max_windows: int = typer.Option(60, help="Cap windows for a fast run; 0 means no cap."),
) -> None:
    """Run the rolling-origin backtest for the statistical baselines."""
    with connect(read_only=True) as conn:
        series = load_series(conn, respondent, series_type)

    console.print(
        f"{series.series_id}: {series.values.size:,} hourly points, "
        f"{series.missing_fraction:.2%} missing"
    )

    models = [WeeklySeasonalNaive(), DailySeasonalNaive()]
    results = [
        rolling_origin_backtest(
            model,
            series.values,
            series_id=series.series_id,
            horizon=horizon,
            max_windows=max_windows or None,
        )
        for model in models
    ]

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
