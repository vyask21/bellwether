"""Guards on EIA Terms of Service obligations.

These are not style checks. Each one corresponds to a clause in
docs/EIA_COMPLIANCE.md, and failing one means the project is out of compliance.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from bellwether.attribution import (
    DERIVED_DISCLAIMER,
    EIA_SOURCE,
    NOAA_SOURCE,
    NOT_AFFILIATED,
    attribution_block,
    eia_acknowledgment,
    noaa_acknowledgment,
)
from bellwether.ingest.eia import MIN_REQUEST_INTERVAL_SECONDS, EIAClient, ObservationRow
from bellwether.ingest.noaa import WeatherRow
from bellwether.storage.db import (
    connect,
    export_snapshot,
    upsert_observations,
    upsert_weather_observations,
)

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

    def test_block_covers_every_source_and_derived_content(self):
        block = attribution_block()
        assert "U.S. Energy Information Administration" in block
        assert "NOAA National Centers for Environmental Information" in block
        assert NOT_AFFILIATED in block
        assert DERIVED_DISCLAIMER in block


class TestNOAAAttribution:
    """NOAA data is a US Government work, so the obligation is citation, not permission."""

    def test_attribution_names_the_dataset_not_just_the_agency(self):
        """NOAA runs many archives; 'NOAA' alone does not identify which one."""
        assert "NOAA National Centers for Environmental Information" in NOAA_SOURCE
        assert "Integrated Surface Database" in NOAA_SOURCE

    @pytest.mark.parametrize(
        "phrase", ["courtesy of", "powered by", "in partnership with", "endorsed", "approved by"]
    )
    def test_attribution_avoids_endorsement_wording(self, phrase: str):
        assert phrase not in NOAA_SOURCE.lower()

    def test_acknowledgment_carries_a_retrieval_date(self):
        """NCEI revises the archive as late reports and QC land, so the date is load-bearing."""
        text = noaa_acknowledgment(date(2026, 7, 30))
        assert "Jul 2026" in text
        assert "retrieved" in text.lower()

    def test_readme_carries_noaa_attribution_and_non_affiliation(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "NOAA National Centers for Environmental Information" in readme
        assert "not affiliated with or endorsed by NOAA" in readme


class TestDatedAcknowledgment:
    """Reuse policy: acknowledgments should include a date."""

    def test_acknowledgment_carries_a_date(self):
        text = eia_acknowledgment(date(2026, 7, 30))
        assert "U.S. Energy Information Administration" in text
        assert "Jul 2026" in text

    def test_date_is_labelled_as_retrieval_not_publication(self):
        """APIv2 exposes no publication date, so claiming one would be a fabrication."""
        assert "retrieved" in eia_acknowledgment(date(2026, 7, 30)).lower()

    def test_accepts_a_datetime_as_well_as_a_date(self):
        moment = datetime(2026, 7, 30, 18, 45, tzinfo=UTC)
        assert "Jul 2026" in eia_acknowledgment(moment)

    def test_defaults_to_today_when_no_date_given(self):
        assert f"{datetime.now(UTC):%b %Y}" in eia_acknowledgment()

    def test_observation_snapshot_dates_from_ingest_not_export_time(self, tmp_path: Path):
        """The acknowledgment should describe the data, not when the file was written."""
        rows = [
            ObservationRow(datetime(2025, 1, 1, tzinfo=UTC), "CISO", "D", 10.0, "megawatthours")
        ]
        out = tmp_path / "snapshots"
        with connect(tmp_path / "t.duckdb") as conn:
            upsert_observations(conn, rows)
            ingested = conn.execute(
                "SELECT strftime(max(ingested_at) AT TIME ZONE 'UTC', '%b %Y') FROM observations"
            ).fetchone()[0]
            export_snapshot(conn, table="observations", directory=out)

        notice = (out / "ATTRIBUTION-observations.txt").read_text(encoding="utf-8")
        assert ingested in notice
        # Dated from the observation period (2025), not from when the export ran.
        assert "2025" not in notice


class TestProtectedMaterials:
    """Reuse policy: EIA marks are trademarked, and site imagery may be privately licensed.

    Public domain status covers EIA's data, not its logo, its Energy Ant servicemark, or
    the photographs and illustrations on eia.gov.
    """

    def test_no_image_assets_reference_an_eia_logo(self):
        tracked = [
            p
            for p in REPO_ROOT.rglob("*")
            if p.is_file()
            and ".git" not in p.parts
            and ".venv" not in p.parts
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".ico"}
        ]
        offenders = [
            p.name
            for p in tracked
            if "eia" in p.name.lower() or "energy-ant" in p.name.lower().replace("_", "-")
        ]
        assert not offenders, f"possible EIA branding assets: {offenders}"


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

    def test_noaa_content_is_stored_apart_from_eia_content(self, tmp_path: Path):
        """Two agencies, two policies. A shared table would blur which notice applies."""
        with connect(tmp_path / "t.duckdb") as conn:
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
            assert "weather_observations" in tables

            weather_columns = {
                row[0] for row in conn.execute("DESCRIBE weather_observations").fetchall()
            }
            assert "respondent" not in weather_columns, (
                "a balancing authority is our mapping, not something NOAA published"
            )

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

        notice = (out / "ATTRIBUTION-observations.txt").read_text(encoding="utf-8")
        assert "U.S. Energy Information Administration" in notice

    def test_exported_forecasts_carry_the_derived_disclaimer(self, tmp_path: Path):
        """A forecast export must not go out under EIA's name."""
        out = tmp_path / "snapshots"
        with connect(tmp_path / "t.duckdb") as conn:
            export_snapshot(conn, table="forecasts", directory=out)

        notice = (out / "ATTRIBUTION-forecasts.txt").read_text(encoding="utf-8")
        assert "not by EIA" in notice

    def test_an_eia_snapshot_does_not_claim_noaa_content(self, tmp_path: Path):
        """Each snapshot credits the agency that published it, and no other.

        One shared attribution file would mean the last export overwrote the previous
        one's notice, leaving EIA demand data labelled as NOAA weather or the reverse.
        """
        rows = [
            ObservationRow(datetime(2025, 1, 1, tzinfo=UTC), "CISO", "D", 10.0, "megawatthours")
        ]
        weather = [WeatherRow(datetime(2025, 1, 1, tzinfo=UTC), "72259003927", "FM-15", 10.0, "1")]
        out = tmp_path / "snapshots"
        with connect(tmp_path / "t.duckdb") as conn:
            upsert_observations(conn, rows)
            upsert_weather_observations(conn, weather)
            export_snapshot(conn, table="observations", directory=out)
            export_snapshot(conn, table="weather_observations", directory=out)

        eia_notice = (out / "ATTRIBUTION-observations.txt").read_text(encoding="utf-8")
        noaa_notice = (out / "ATTRIBUTION-weather_observations.txt").read_text(encoding="utf-8")

        assert "U.S. Energy Information Administration" in eia_notice
        assert "NOAA" not in eia_notice
        assert "NOAA National Centers for Environmental Information" in noaa_notice
        assert "Energy Information Administration" not in noaa_notice


class TestComplianceDocumentation:
    def test_compliance_doc_covers_both_policies_with_review_dates(self):
        doc = (REPO_ROOT / "docs" / "EIA_COMPLIANCE.md").read_text(encoding="utf-8")
        assert "terms-of-service" in doc
        assert "copyrights_reuse" in doc
        assert doc.count("Reviewed against:") >= 2
