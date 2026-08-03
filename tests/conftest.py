"""Fixtures shared by the tests that need a built Space.

The site is built once for the session against the real snapshot rather than a fabricated
one. What these tests assert is that the committed data produces a page with charts in it,
and a fixture holding invented data would only assert that the template has holes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dashboard"))

MARKETS = ("CISO", "ERCO", "PACE")


def has_snapshot() -> bool:
    """All three markets, not merely one: a partial export builds a page that is quietly
    missing a market rather than obviously broken."""
    exported = {
        path.stem.removeprefix("forecasts_")
        for path in (ROOT / "snapshot").glob("forecasts_*.parquet")
    }
    return set(MARKETS) <= exported


@pytest.fixture(scope="session")
def static_site(tmp_path_factory) -> Path:
    """The published Space, built into a temporary directory."""
    builder = pytest.importorskip("build_static_space")
    if not has_snapshot():
        pytest.skip("snapshot not fully exported")
    site = tmp_path_factory.mktemp("space") / "site"
    builder.build(site)
    return site


@pytest.fixture(scope="session")
def page(static_site: Path) -> str:
    return (static_site / "index.html").read_text(encoding="utf-8")
