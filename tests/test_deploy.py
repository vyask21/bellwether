"""The Space assembler ships an allowlist, and the guard that backs it actually fires.

The repository is private and the Space is public, so a mistake here publishes secrets to
the internet rather than merely breaking a chart. Two properties are worth testing: that
nothing outside the allowlist can reach the staged tree, and that the credential scan is
not decorative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import deploy_space  # noqa: E402


class TestAllowlist:
    def test_no_secret_bearing_path_is_listed(self):
        """The allowlist is the primary control; this asserts nobody widened it carelessly."""
        listed = {source for source, _ in deploy_space.ALLOWLIST}
        for source in listed:
            assert not source.startswith((".env", "data/", "src/", ".git")), source
            assert "secret" not in source.lower()

    def test_it_ships_no_source_code_beyond_the_three_dashboard_modules(self):
        """The Space needs the app, not the project. `src/bellwether` never goes."""
        python_files = {dest for _, dest in deploy_space.ALLOWLIST if dest.endswith(".py")}
        assert python_files == {"app.py", "loaders.py", "viz.py"}

    def test_the_env_file_exists_and_is_not_listed(self):
        """A guard that passes because the file is absent has proved nothing."""
        assert (ROOT / ".env").exists(), "expected a local .env to make this test meaningful"
        assert all(source != ".env" for source, _ in deploy_space.ALLOWLIST)


class TestSecretScan:
    @pytest.mark.parametrize(
        "content",
        [
            "EIA_API_KEY=abcdef1234567890abcdef",
            "HF_TOKEN=hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "token = 'hf_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'",
            "AWS key AKIAIOSFODNN7EXAMPLE here",
            "-----BEGIN RSA PRIVATE KEY-----",
            'api_key: "0123456789abcdef0123"',
        ],
    )
    def test_it_catches_a_planted_credential(self, tmp_path, content):
        (tmp_path / "leaked.py").write_text(content)
        findings = deploy_space._scan(tmp_path)
        assert findings, f"scanner missed: {content[:40]}"

    def test_it_passes_a_clean_tree(self):
        findings = deploy_space._scan(ROOT / "dashboard")
        assert findings == [], findings

    def test_it_does_not_read_parquet_as_text(self, tmp_path):
        """Demand readings are not credentials and scanning binary for a regex is theatre."""
        (tmp_path / "demand.parquet").write_bytes(b"PAR1\x00EIA_API_KEY\x00")
        assert deploy_space._scan(tmp_path) == []


class TestStaging:
    def test_it_copies_only_what_is_named(self, tmp_path):
        staged, _ = deploy_space._stage(tmp_path)
        relative = {p.relative_to(tmp_path).as_posix() for p in staged}
        allowed = {dest for _, dest in deploy_space.ALLOWLIST}
        allowed |= {
            f"snapshot/{kind}_{m}.parquet"
            for m in deploy_space.MARKETS
            for kind in ("demand", "forecasts")
        }
        assert relative <= allowed
        assert not any(".env" in name for name in relative)

    def test_a_missing_source_is_reported_rather_than_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            deploy_space, "ALLOWLIST", (("does/not/exist.py", "exist.py"),) + deploy_space.ALLOWLIST
        )
        _, missing = deploy_space._stage(tmp_path)
        assert "does/not/exist.py" in missing
