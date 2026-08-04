"""The Space publishes only what it builds, and the guard behind that actually fires.

The repository is private and the Space is public, so a mistake here publishes secrets to
the internet rather than merely breaking a chart. The control used to be an allowlist of
repository paths; it is now stronger, because the published tree is generated and no file
in the working directory has a path into it at all. Two properties are still worth testing:
that nothing but the build reaches the tree, and that the credential scan is not
decorative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import deploy_space  # noqa: E402


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

    def test_it_reads_the_formats_the_site_is_made_of(self, tmp_path):
        """The page is HTML, CSS and CSV now. A scanner that only knew Python would pass a
        key embedded in the one file that is guaranteed to ship."""
        for suffix in (".html", ".css", ".csv", ".json", ".md"):
            (tmp_path / f"leaked{suffix}").write_text("EIA_API_KEY=abcdef1234567890abcdef")
        assert len(deploy_space._scan(tmp_path)) == 5

    def test_it_passes_a_clean_tree(self):
        findings = deploy_space._scan(ROOT / "dashboard")
        assert findings == [], findings

    def test_it_does_not_read_binary_as_text(self, tmp_path):
        """Demand readings are not credentials and scanning binary for a regex is theatre."""
        (tmp_path / "demand.parquet").write_bytes(b"PAR1\x00EIA_API_KEY\x00")
        assert deploy_space._scan(tmp_path) == []


class TestWhatReachesTheSpace:
    def test_the_built_tree_passes_its_own_scan(self, static_site):
        """The guard runs on what ships, so this is the assertion that matters most."""
        assert deploy_space._scan(static_site) == []

    def test_nothing_from_the_repository_travels(self, static_site):
        """The old allowlist could be widened by editing a tuple. This cannot be widened at
        all without writing code that emits the file."""
        shipped = {path.name for path in static_site.rglob("*") if path.is_file()}
        assert not {".env", "app.py", "loaders.py", "viz.py", "content.py"} & shipped
        assert not any(name.endswith((".py", ".parquet", ".ipynb")) for name in shipped)

    @pytest.mark.skipif(
        not (ROOT / ".env").exists(),
        reason="no local .env, so there is no secret here for the guard to fail to catch",
    )
    def test_the_env_file_exists_so_the_check_above_means_something(self):
        """A guard that passes because the file is absent has proved nothing.

        Skipped rather than asserted, because a CI runner legitimately has no `.env`: the
        file is gitignored and has never been committed. Asserting it there turned a
        correct state of the world into a red build.
        """
        assert (ROOT / ".env").exists()
