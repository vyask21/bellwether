"""The static build carries the whole walkthrough, and carries it honestly.

The Space is a compiled artifact now, which introduces a failure this project did not have
before: the page can be built successfully and be missing a section, a chart, or a
paragraph, and nothing at build time would say so. Streamlit at least crashed. So the
tests here are about completeness rather than mechanics.

The prose tests are the point. `content.py` exists so a finding that gets re-measured is
corrected in one place, and the way that guarantee dies is a builder that silently stops
emitting one of the strings.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dashboard"))

pytest.importorskip("jinja2")
pytest.importorskip("markdown_it")
pytest.importorskip("streamlit")

import build_static_space as builder  # noqa: E402
import content as text  # noqa: E402

MARKETS = ("CISO", "ERCO", "PACE")


def _plain(value: str) -> str:
    """Text with its markup removed, so markdown and HTML can be compared to each other.

    List markers go too. A bullet is `- ` in the source and a rendered marker in the
    output, and the sentence after it is the thing under test either way.
    """
    without_tags = re.sub(r"<[^>]+>", " ", value)
    unescaped = html.unescape(without_tags)
    without_markers = re.sub(r"(?m)^\s*[-*]\s+", "", unescaped)
    return re.sub(r"[\s*`]+", " ", without_markers).strip()


# Two constants cannot reach a static page and are named rather than filtered out silently.
# `NO_SNAPSHOT` is Streamlit's warning for a missing export, which this builder treats as a
# build failure instead of a banner; `FALLBACK_ATTRIBUTION` renders only when the manifest
# carries no source notice, and the committed one does. A third exclusion has to be argued
# for here rather than quietly added.
ONLY_IN_STREAMLIT = {"NO_SNAPSHOT", "FALLBACK_ATTRIBUTION"}

# Every narrative constant, found by name rather than listed, so a new section added to
# content.py is covered the moment it exists instead of when someone remembers this file.
PROSE = [
    (name, getattr(text, name))
    for name in dir(text)
    if name.isupper()
    and name not in ONLY_IN_STREAMLIT
    and isinstance(getattr(text, name), str)
    and len(getattr(text, name)) > 60
    and "{" not in getattr(text, name)
]


class TestTheWalkthroughIsWhole:
    @pytest.mark.parametrize(("name", "value"), PROSE, ids=[name for name, _ in PROSE])
    def test_every_paragraph_reaches_the_page(self, name, value, page):
        """`content.py` is only a single source if both renderers actually read all of it."""
        assert _plain(value) in _plain(page), name

    @pytest.mark.parametrize(
        "heading",
        [
            text.S1_HEADING,
            text.S2_HEADING,
            text.S4_HEADING,
            text.S5_HEADING,
            text.S6_HEADING,
            text.S7_HEADING,
            text.S8_HEADING,
            text.METHOD_HEADING,
        ],
    )
    def test_every_section_has_its_heading(self, heading, page):
        assert heading in page

    @pytest.mark.parametrize("name", sorted(ONLY_IN_STREAMLIT))
    def test_the_excluded_constants_are_absent_for_the_stated_reason(self, name, page):
        """If one of these starts rendering, the exclusion above is wrong and this says so
        rather than letting the coverage check quietly skip a paragraph forever."""
        assert _plain(getattr(text, name)) not in _plain(page)

    def test_the_market_specific_heading_is_built_for_all_three(self, page):
        for market in MARKETS:
            assert text.S3_HEADING.format(market=market) in page

    def test_the_forecast_ablation_ships_its_numbers_and_not_only_its_prose(self, page):
        """The claim is a comparison of three arms, so the arms have to be on the page. The
        prose test above would pass on a section that asserts four fifths and shows none."""
        for label in ("Observed, 3-hourly", "NDFD forecast"):
            assert label in page
        assert "sMAPE change (%)" in page

    def test_the_withdrawn_claims_survive_the_port(self, page):
        """A results page that reports only what survived is not reporting, and a rewrite
        is exactly where the embarrassing section quietly fails to get carried over."""
        assert "turned out to be wrong" in page
        assert "unconditional" in page
        assert "duck curve" in page


class TestTheMarketControlSitsBesideWhatItScopes:
    """It used to render once, above section 1, and section 1 is a cross-market chart.

    So the control changed nothing the reader could see from where it sat, and read as
    broken. What makes that a test rather than a matter of taste is the invariant
    underneath it: a control belongs in the sections it scopes, and in no others.
    """

    @staticmethod
    def _sections(page: str) -> list[str]:
        return re.findall(r"<section>(.*?)</section>", page, re.S)

    def test_the_control_is_not_rendered_outside_a_section(self, page):
        """Anything above section 1 scopes nothing at all. That was the bug."""
        header = page.split("<section>", 1)[0]
        assert "data-market-input" not in header

    def test_every_market_scoped_section_carries_a_control(self, page):
        scoped = [s for s in self._sections(page) if 'data-market="' in s]
        assert len(scoped) == 2, "sections 3 and 7 are the market-scoped ones"
        for section in scoped:
            assert "data-market-input" in section

    def test_no_unscoped_section_carries_a_control(self, page):
        for section in self._sections(page):
            if "data-market-input" in section:
                assert 'data-market="' in section

    def test_each_copy_is_its_own_radio_group(self, page):
        """Same-named radios are one group for the whole document, so two copies sharing a
        name would leave the second one showing nothing selected."""
        names = re.findall(r'<input type="radio" name="([^"]+)"', page)
        assert len(names) == len(MARKETS) * 2
        assert len(set(names)) == 2
        for name in set(names):
            assert names.count(name) == len(MARKETS)

    def test_every_copy_offers_every_market(self, page):
        for section in self._sections(page):
            if "data-market-input" not in section:
                continue
            for market in MARKETS:
                assert f'value="{market}"' in section


@pytest.fixture(scope="session")
def specs(page) -> dict:
    raw = re.search(r"^const SPECS = (\{.*\});$", page, re.M)
    assert raw, "the page embeds no specifications at all"
    return json.loads(raw.group(1))


class TestEveryChartIsThere:
    def test_all_thirteen_charts_are_embedded(self, specs):
        expected = {"skill", "coverage", "models", "hours", "months", "offsets", "shape"}
        expected |= {f"{kind}_{m}" for kind in ("holidays", "forecast") for m in MARKETS}
        assert set(specs) == expected

    @pytest.mark.parametrize(
        "name", ["skill", "coverage", "models", "hours", "months", "offsets", "shape"]
    )
    def test_a_fixed_chart_carries_its_data_rather_than_a_url(self, name, specs):
        """These are tens of rows. Inlining them means the page draws before it fetches."""
        spec = specs[name]
        assert spec.get("datasets") or "values" in json.dumps(spec.get("data", {}))

    @pytest.mark.parametrize("market", MARKETS)
    def test_the_forecast_chart_streams_and_is_driven_by_a_signal(self, market, specs):
        spec = specs[f"forecast_{market}"]
        blob = json.dumps(spec)
        assert f"data/forecast_{market}.csv" in blob
        assert f"data/observed_{market}.csv" in blob
        assert any(param.get("name") == "originIdx" for param in spec.get("params", []))

    def test_every_chart_ships_a_table_view(self, page):
        """Carried by the palette, not by taste: the aqua slot is under the 3:1 bar and the
        documented relief is a reachable table. A touch device has no hover at all."""
        charts = page.count('class="chart-box"')
        tables = page.count("<summary>")
        assert charts == 13
        # Ten table views under charts, plus the method note, which is also a disclosure.
        assert tables >= 11

    def test_no_value_is_only_obtainable_from_a_chart(self, page):
        """The tables are the record. If the Vega CDN fails the page must still carry the
        numbers, so the fallback says where they are rather than showing a broken box."""
        assert "did not " in page and "table view beneath it" in page


class TestTheDataItShips:
    @pytest.mark.parametrize("market", MARKETS)
    def test_the_observed_series_is_indexed_by_row(self, market, static_site):
        frame = pd.read_csv(static_site / "data" / f"observed_{market}.csv")
        assert list(frame.columns) == ["idx", "period", "demand_mw"]
        assert frame["idx"].tolist() == list(range(len(frame)))
        assert len(frame) > 17_000

    @pytest.mark.parametrize("market", MARKETS)
    def test_every_forecast_window_is_twenty_four_hours(self, market, static_site):
        frame = pd.read_csv(static_site / "data" / f"forecast_{market}.csv")
        assert list(frame.columns) == ["origin", "period", "q10", "q50", "q90"]
        assert set(frame.groupby("origin").size()) == {24}
        assert (frame["q10"] <= frame["q50"]).all()
        assert (frame["q50"] <= frame["q90"]).all()

    @pytest.mark.parametrize("market", MARKETS)
    def test_the_lead_in_can_be_found_by_arithmetic(self, market, static_site):
        """The property `forecast_explorer` rests on, checked on what actually shipped
        rather than on the frames it was built from."""
        observed = pd.read_csv(static_site / "data" / f"observed_{market}.csv")
        forecast = pd.read_csv(static_site / "data" / f"forecast_{market}.csv")
        first = forecast.groupby("origin")["period"].min()
        at_index = observed.set_index("idx")["period"].reindex(first.index)
        assert (first.to_numpy() == at_index.to_numpy()).all()

    def test_it_publishes_no_python_and_no_snapshot(self, static_site):
        """The Space needs the page, not the project. Nothing executable ships at all."""
        shipped = {path.suffix for path in static_site.rglob("*") if path.is_file()}
        assert shipped == {".html", ".css", ".md", ".csv"}

    def test_the_card_says_static(self, static_site):
        card = (static_site / "README.md").read_text(encoding="utf-8")
        assert "sdk: static" in card
        assert "app_file: index.html" in card

    def test_the_card_passes_the_hub_metadata_limits(self, static_site):
        """The Hub validates the card before it accepts a single file, so a card it rejects
        fails the upload after the repository has already been created. Caught that way
        once, at 70 characters against a limit of 60."""
        card = (static_site / "README.md").read_text(encoding="utf-8")
        front = card.split("---", 2)[1]
        described = re.search(r"^short_description:\s*(.+)$", front, re.M)
        assert described, "the card carries no short_description"
        assert len(described.group(1).strip()) <= 60, described.group(1)


class TestTheOriginGuard:
    """The alignment assertion is load-bearing, so it is tested rather than trusted."""

    def _frames(self, shift: int = 0):
        periods = pd.date_range("2025-01-01", periods=96, freq="h")
        observed = pd.DataFrame({"idx": range(96), "period": periods, "demand_mw": range(96)})
        forecast = pd.DataFrame(
            {
                "origin": [24] * 24 + [48] * 24,
                "period": list(periods[24 + shift : 48 + shift])
                + list(periods[48 + shift : 72 + shift]),
                "q10": 0,
                "q50": 0,
                "q90": 0,
            }
        )
        return observed, forecast

    def test_it_passes_when_origin_is_the_row_index(self):
        observed, forecast = self._frames()
        builder._assert_origin_is_a_row_index("TEST", observed, forecast)

    def test_it_refuses_to_draw_a_window_that_is_off_by_an_hour(self):
        observed, forecast = self._frames(shift=1)
        with pytest.raises(SystemExit, match="row indices"):
            builder._assert_origin_is_a_row_index("TEST", observed, forecast)


class TestTables:
    def test_a_table_is_rendered_with_its_rounding(self):
        frame = pd.DataFrame({"market": ["CISO"], "value": [1234.5678]})
        assert "1,234.57" in builder._table(frame, 2)
        assert "1,235" in builder._table(frame, 0)

    def test_an_empty_frame_renders_nothing_rather_than_an_empty_table(self):
        assert builder._table(pd.DataFrame(), 0) == ""

    def test_it_escapes_rather_than_trusting_a_label(self):
        frame = pd.DataFrame({"arm": ["<script>alert(1)</script>"]})
        assert "<script>" not in builder._table(frame, 0)
