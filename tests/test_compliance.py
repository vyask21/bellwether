"""Guards on EIA Terms of Service obligations.

These are not style checks. Each one corresponds to a clause in
docs/EIA_COMPLIANCE.md, and failing one means the project is out of compliance.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bellwether.attribution import DERIVED_DISCLAIMER, EIA_SOURCE, attribution_block
from bellwether.ingest.eia import MIN_REQUEST_INTERVAL_SECONDS, EIAClient, ObservationRow
from bellwether.storage.db import connect, export_snapshot, upsert_observations

REPO_ROOT = Path(__file__).resolve().parents[1]

# EIA's published burst ceiling. Staying an order of magnitude under it is deliberate.
PUBLISHED_BURST_LIMIT_PER_SECOND = 5


class TestAttribution:
    """TOS: identify EIA as the source; do not imply endorsement."""

    def test_attribution_names_eia(self):
        assert "U.S. Energy Information Administration" in EIA_SOURCE

    @pytest.mark.parametrize(
        "phrase", ["courtesy of", "powered by", "in partnership with", "endorsed", "approved by"]
    )
    def test_attribution_avoids_endorsement_wording(self, phrase: str):
        assert phrase not in EIA_SOURCE.lower()

    def test_readme_carries_attribution_and_non_affiliation(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "U.S. Energy Information Administration" in readme
        assert "not affiliated with or endorsed by EIA" in readme

    def test_block_covers_both_eia_and_derived_content(self):
        block = attribution_block()
        assert EIA_SOURCE in block
        assert DERIVED_DISCLAIMER in block


class TestNoLogoUse:
    """TOS: the EIA logo is trademarked and may not be used."""

    def test_no_image_assets_reference_an_eia_logo(self):
        tracked = [
            p
            for p in REPO_ROOT.rglob("*")
            if p.is_file()
            and ".git" not in p.parts
            and ".venv" not in p.parts
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".ico"}
        ]
        offenders = [p.name for p in tracked if "eia" in p.name.lower()]
        assert not offenders, f"possible EIA logo assets: {offenders}"


class TestRateLimitCompliance:
    """TOS: do not exceed or circumvent access limits."""

    def test_default_pacing_is_well_under_the_published_burst_ceiling(self):
        requests_per_second = 1.0 / MIN_REQUEST_INTERVAL_SECONDS
        assert requests_per_second < PUBLISHED_BURST_LIMIT_PER_SECOND
        # Not merely under the ceiling, but with real headroom, since EIA states the true
        # limit varies by key, series demand, and IP.
        assert requests_per_second <= PUBLISHED_BURST_LIMIT_PER_SECOND / 4

    def test_client_uses_a_single_key(self):
        client = EIAClient(api_key="one-key")
        assert isinstance(client._api_key, str)


class TestNoModificationOfEIAContent:
    """TOS: do not modify or falsely represent API content and still claim EIA as source."""

    def test_nulls_are_stored_as_null_not_zero_filled(self, tmp_path: Path):
        rows = [
            ObservationRow(datetime(2025, 1, 1, 0, tzinfo=UTC), "CISO", "D", None, "megawatthours"),
            ObservationRow(
                datetime(2025, 1, 1, 1, tzinfo=UTC), "CISO", "D", 100.0, "megawatthours"
            ),
        ]
        with connect(tmp_path / "t.duckdb") as conn:
            upsert_observations(conn, rows)
            stored = conn.execute("SELECT value FROM observations ORDER BY period").fetchall()

        assert stored[0][0] is None, "an EIA gap must not be filled with a fabricated value"
        assert stored[1][0] == 100.0

    def test_forecasts_are_stored_apart_from_observations(self, tmp_path: Path):
        """Derived values must never be readable as EIA content."""
        with connect(tmp_path / "t.duckdb") as conn:
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
            assert {"observations", "forecasts"} <= tables

            observation_columns = {
                row[0] for row in conn.execute("DESCRIBE observations").fetchall()
            }
            assert "model_name" not in observation_columns

    def test_snapshot_export_rejects_unknown_tables(self, tmp_path: Path):
        with (
            connect(tmp_path / "t.duckdb") as conn,
            pytest.raises(ValueError, match="Refusing to export"),
        ):
            export_snapshot(conn, table="observations; DROP TABLE observations")

    def test_exported_observations_carry_eia_attribution(self, tmp_path: Path):
        rows = [
            ObservationRow(datetime(2025, 1, 1, tzinfo=UTC), "CISO", "D", 10.0, "megawatthours")
        ]
        out = tmp_path / "snapshots"
        with connect(tmp_path / "t.duckdb") as conn:
            upsert_observations(conn, rows)
            export_snapshot(conn, table="observations", directory=out)

        notice = (out / "ATTRIBUTION.txt").read_text(encoding="utf-8")
        assert "U.S. Energy Information Administration" in notice

    def test_exported_forecasts_carry_the_derived_disclaimer(self, tmp_path: Path):
        """A forecast export must not go out under EIA's name."""
        out = tmp_path / "snapshots"
        with connect(tmp_path / "t.duckdb") as conn:
            export_snapshot(conn, table="forecasts", directory=out)

        notice = (out / "ATTRIBUTION.txt").read_text(encoding="utf-8")
        assert "not by EIA" in notice


class TestComplianceDocumentation:
    def test_compliance_doc_exists_and_records_a_review_date(self):
        doc = (REPO_ROOT / "docs" / "EIA_COMPLIANCE.md").read_text(encoding="utf-8")
        assert "terms-of-service" in doc
        assert "Reviewed against:" in doc
