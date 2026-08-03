"""Compile the walkthrough into a static site.

Usage:
    python scripts/build_static_space.py            # build into .space/
    python scripts/build_static_space.py --out dir

## Why this exists

The dashboard was written for Streamlit and Streamlit needs a running Python process.
Hugging Face retired the `streamlit` Space SDK, and the Docker Space that replaces it is not
free, so the published artifact is a static page. Static is not a downgrade here: this
dashboard already computed everything ahead of time and read committed files at render
time, so there was never a server doing work worth paying for.

## What moves and what does not

Nothing about the analysis moves. `loaders.py` builds exactly the frames it builds for
Streamlit and `viz.py` builds exactly the same charts, because an Altair chart *is* a
Vega-Lite specification and a browser can draw one without Python. The prose comes from
`content.py`, which both renderers read.

What does move is the slicing. Streamlit re-ran the script on every widget change; here the
page ships each market's full snapshot and a Vega signal picks the window. That is the one
place where the two renderers do the same thing differently, so `forecast_explorer` sits
next to `forecast_window` in `viz.py` where the difference is visible.

## The allowlist, restated

`deploy_space.py` shipped a named list of files because the repository is private and the
Space is public, and an exclusion list leaks by default. A generated site keeps that
property more strongly: nothing reaches the Space that this module did not construct, so
there is no path by which a stray file travels. The credential scan still runs over the
output, because the allowlist argument protects against forgetting, not against mistakes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import content as text  # noqa: E402
import loaders as data  # noqa: E402
import viz  # noqa: E402
from jinja2 import Template  # noqa: E402
from markdown_it import MarkdownIt  # noqa: E402

# Pinned majors. Altair 6 emits the Vega-Lite 6 schema; a runtime that predates it fails
# with a blank chart and a console message nobody reading a results page will open.
VEGA = "https://cdn.jsdelivr.net/npm/vega@6"
VEGA_LITE = "https://cdn.jsdelivr.net/npm/vega-lite@6"
VEGA_EMBED = "https://cdn.jsdelivr.net/npm/vega-embed@7"

ARM = "scale"  # The shipped arm. Section 3 draws what the project would actually publish.

_md = MarkdownIt("commonmark")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".space", help="output directory")
    args = parser.parse_args()

    out = ROOT / args.out
    written = build(out)
    total = sum(path.stat().st_size for path in written)
    print(f"Built {len(written)} files into {out.relative_to(ROOT)}:")
    for path in sorted(written):
        print(f"  {str(path.relative_to(out)):<40}{path.stat().st_size / 1024:>9,.0f} KB")
    print(f"  {'total':<40}{total / 1024:>9,.0f} KB")
    return 0


def build(out: Path) -> list[Path]:
    """Write the whole site. Returns every path written, in no particular order."""
    if not data.snapshot_available():
        raise SystemExit(
            "No snapshot. Run scripts/export_snapshot.py for each market first: a Space "
            "built without one publishes a walkthrough whose forecast section is empty."
        )
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)

    written = []
    origins = {}
    for market in data.MARKETS:
        paths, index = _write_market_data(out, market)
        written.extend(paths)
        origins[market] = index

    specs, tables = _charts(origins)
    written.append(_write_page(out, specs, origins, tables))
    written.append(_write_readme(out))
    written.append(_write_css(out))
    return written


# ----------------------------------------------------------------------------------------
# Data


def _write_market_data(out: Path, market: str) -> tuple[list[Path], dict]:
    """One market's observed series and stored forecasts, as CSV the browser can stream.

    Parquet would be smaller and no browser reads it without a WebAssembly runtime that
    costs more to ship than the saving. These are the only two files the page fetches, and
    it fetches a market's pair the first time that market is shown.
    """
    observed = data.observations(market)[["period", "demand_mw"]].copy()
    observed.insert(0, "idx", range(len(observed)))

    forecasts = data.forecasts(market, ARM)[["origin", "period", "q10", "q50", "q90"]].copy()
    _assert_origin_is_a_row_index(market, observed, forecasts)

    demand_path = out / "data" / f"observed_{market}.csv"
    forecast_path = out / "data" / f"forecast_{market}.csv"
    observed.to_csv(
        demand_path, index=False, date_format="%Y-%m-%dT%H:%M", float_format="%.0f"
    )
    forecasts.to_csv(
        forecast_path, index=False, date_format="%Y-%m-%dT%H:%M", float_format="%.0f"
    )

    labels = data.origin_labels(forecasts)
    values = sorted(labels)
    return [demand_path, forecast_path], {
        "values": values,
        "labels": [labels[v] for v in values],
        "initial": len(values) // 2,
    }


def _assert_origin_is_a_row_index(
    market: str, observed: pd.DataFrame, forecasts: pd.DataFrame
) -> None:
    """The whole forecast section rests on this, so it is checked rather than assumed.

    `forecast_explorer` picks the 24 hours of lead-in by row arithmetic on `origin`. If the
    export ever numbered origins differently the page would keep drawing, confidently, two
    days that are not the two days named beside it. That is the failure this project has
    already shipped once, with a grid built from the wrong bounds.
    """
    first = forecasts.groupby("origin", observed=True)["period"].min()
    at_index = observed.set_index("idx")["period"].reindex(first.index)
    mismatched = int((first.to_numpy() != at_index.to_numpy()).sum())
    if mismatched:
        raise SystemExit(
            f"{market}: {mismatched} of {len(first)} origins are not row indices into the "
            "observed series. The snapshot export and the dashboard disagree about what "
            "`origin` means, and the forecast chart cannot be drawn safely."
        )


# ----------------------------------------------------------------------------------------
# Charts and tables


def _charts(origins: dict) -> tuple[dict, dict]:
    """Every Vega-Lite specification the page embeds, and every table beneath one."""
    viz.enable_theme()
    specs, tables = {}, {}

    skill = data.skill_frame()
    specs["skill"] = _spec(viz.skill_bars(skill))
    tables["skill"] = _table(skill, 3)

    coverage = data.coverage_width_frame()
    specs["coverage"] = _spec(viz.coverage_and_width(coverage))
    tables["coverage"] = _table(coverage, 2)

    hours = data.profile_frame("mae", "by_local_hour")
    specs["hours"] = _spec(viz.profile(hours, "bucket", *_axes(text.S5_HOUR_AXES)))
    tables["hours"] = _table(hours, 3)

    months = data.profile_frame("coverage", "by_month")
    specs["months"] = _spec(viz.profile(months, "bucket", *_axes(text.S5_MONTH_AXES)))
    tables["months"] = _table(months, 3)

    offsets = data.offsets_frame()
    specs["offsets"] = _spec(viz.learned_offsets(offsets))
    tables["offsets"] = _table(offsets, 0)

    for market in data.MARKETS:
        holidays = data.per_holiday_frame(market)
        specs[f"holidays_{market}"] = _spec(viz.paired_holidays(holidays))
        tables[f"holidays_{market}"] = _table(holidays, 0)
        specs[f"forecast_{market}"] = _spec(
            viz.forecast_explorer(
                f"data/observed_{market}.csv",
                f"data/forecast_{market}.csv",
                origins[market]["values"][origins[market]["initial"]],
            )
        )

    tables["operator"] = _table(data.operator_frame(), 3)
    tables["weather"] = _table(data.weather_frame(), 1)
    return specs, tables


def _axes(pair: tuple[str, str]) -> tuple[str, str, str]:
    """`viz.profile` takes (x, x_title, y, y_title) and the fields never vary."""
    return pair[0], "value", pair[1]


def _spec(chart) -> dict:
    """An Altair chart as Vega-Lite, sized to whatever column it lands in.

    `container` width is a single and layered view feature. A concatenation gets it applied
    to each child instead, and a facet gets nothing: its width is the sum of its columns,
    and the page lets that one scroll rather than squeezing 27 holiday labels.
    """
    spec = chart.to_dict()
    if "vconcat" in spec:
        for child in spec["vconcat"]:
            child["width"] = "container"
    elif not {"facet", "hconcat", "concat"} & set(spec):
        spec["width"] = "container"
    return spec


def _table(frame: pd.DataFrame, decimals: int) -> str:
    """The table view every chart carries.

    Carried by the palette, not by taste: the aqua slot sits at 2.74:1 against this
    surface, under the 3:1 bar, and the documented relief is a reachable table. It also
    means no value on this page exists only inside a tooltip, which matters more here than
    it did in Streamlit because a touch device has no hover at all.
    """
    if frame.empty:
        return ""
    rounded = frame.round(decimals)
    head = "".join(f"<th>{_escape(column)}</th>" for column in rounded.columns)
    rows = []
    for record in rounded.to_dict("records"):
        cells = "".join(f"<td>{_cell(value, decimals)}</td>" for value in record.values())
        rows.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _cell(value, decimals: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<span class="num">{value:,.{decimals}f}</span>'
    return _escape(str(value))


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _prose(markdown: str) -> str:
    return _md.render(markdown)


# ----------------------------------------------------------------------------------------
# The page


def _write_page(out: Path, specs: dict, origins: dict, tables: dict) -> Path:
    generated = data.manifest().get("generated", "not yet")
    page = Template(TEMPLATE, autoescape=False).render(
        text=text,
        markets=list(data.MARKETS),
        captions=text.MARKET_CAPTIONS,
        prose=_prose,
        tables=tables,
        attribution=data.attribution(),
        caption=text.SNAPSHOT_CAPTION.format(generated=generated),
        specs=json.dumps(specs, separators=(",", ":")),
        origins=json.dumps(origins, separators=(",", ":")),
        titles=json.dumps(
            {m: text.S3_TITLE for m in data.MARKETS}, separators=(",", ":")
        ),
        headings=json.dumps(
            {m: text.S3_HEADING.format(market=m) for m in data.MARKETS},
            separators=(",", ":"),
        ),
        vega=VEGA,
        vega_lite=VEGA_LITE,
        vega_embed=VEGA_EMBED,
    )
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def _write_readme(out: Path) -> Path:
    """The Space card, copied rather than composed.

    Its front matter is what Hugging Face reads to decide the Space is static, so it stays
    in `dashboard/README.md` where a person editing the card can see it, not in a template
    string here where a second copy would drift out of agreement with the first.
    """
    path = out / "README.md"
    shutil.copy2(ROOT / "dashboard" / "README.md", path)
    return path


def _write_css(out: Path) -> Path:
    path = out / "style.css"
    path.write_text(
        STYLE.format(
            surface=viz.SURFACE,
            ink=viz.INK_PRIMARY,
            secondary=viz.INK_SECONDARY,
            muted=viz.INK_MUTED,
            grid=viz.GRID,
            axis=viz.AXIS,
            accent=viz.SERIES[0],
            font=viz.FONT,
        ),
        encoding="utf-8",
    )
    return path


STYLE = """/* Generated by scripts/build_static_space.py. The palette is viz.py's, which was
   validated rather than chosen; changing a colour here changes it in only one of the two
   places it is used. Light mode only, deliberately: the dark steps have not been checked
   and shipping an unvalidated flip is how contrast failures reach production. */
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: {surface};
  color: {ink};
  font-family: {font};
  font-size: 16px;
  line-height: 1.6;
}}
main {{ max-width: 60rem; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }}
h1 {{ font-size: 2.25rem; letter-spacing: -0.02em; margin: 0 0 0.75rem; }}
h2 {{ font-size: 1.3rem; margin: 0 0 1rem; letter-spacing: -0.01em; }}
p {{ margin: 0 0 1rem; }}
a {{ color: {accent}; }}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.875em;
  background: #f0efec;
  padding: 0.1em 0.35em;
  border-radius: 3px;
}}
.lede {{ font-size: 1.0625rem; color: {ink}; }}
.caption {{ color: {muted}; font-size: 0.875rem; }}
section {{ padding: 2rem 0; border-top: 1px solid {grid}; }}
section:first-of-type {{ border-top: none; }}

/* One control row, above everything it scopes. Cross-market sections ignore it. */
.controls {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem 1.25rem;
  margin-bottom: 1.25rem;
}}
.controls .label {{ color: {secondary}; font-size: 0.875rem; }}
fieldset {{ border: none; margin: 0; padding: 0; display: flex; gap: 1.25rem; flex-wrap: wrap; }}
fieldset label {{ display: flex; align-items: baseline; gap: 0.4rem; cursor: pointer; }}
fieldset .market {{ font-weight: 500; }}
fieldset .place {{ color: {muted}; font-size: 0.8125rem; }}
input[type="range"] {{ accent-color: {accent}; flex: 1 1 18rem; max-width: 32rem; }}
output {{ font-variant-numeric: tabular-nums; color: {secondary}; }}

.chart {{ margin: 0 0 1.25rem; }}
.chart-box {{ width: 100%; overflow-x: auto; }}
/* vega-embed's UMD bundle ships no stylesheet, so its wrapper defaults to an inline box
   with no width. A chart sized `container` then measures zero and draws nothing, which is
   how a page full of correct specifications renders blank. */
.vega-embed, .vega-embed > div {{ display: block; width: 100%; }}
.vega-embed summary {{ display: none; }}
.chart-title {{
  color: {secondary};
  font-size: 0.9375rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
}}

details {{ border-top: 1px solid {grid}; margin-top: 0.75rem; }}
summary {{
  cursor: pointer;
  padding: 0.6rem 0;
  color: {secondary};
  font-size: 0.875rem;
  list-style-position: outside;
}}
summary:hover {{ color: {ink}; }}
.table-wrap {{ overflow-x: auto; max-height: 26rem; overflow-y: auto; margin-bottom: 1rem; }}
table {{ border-collapse: collapse; font-size: 0.875rem; width: 100%; }}
th, td {{ text-align: left; padding: 0.35rem 0.9rem 0.35rem 0; white-space: nowrap; }}
th {{
  color: {secondary};
  font-weight: 500;
  border-bottom: 1px solid {axis};
  position: sticky;
  top: 0;
  background: {surface};
}}
td {{ border-bottom: 1px solid {grid}; color: {ink}; }}
.num {{ font-variant-numeric: tabular-nums; }}

.tabs {{ display: flex; gap: 0.25rem; margin-bottom: 1.25rem; }}
.tabs button {{
  font: inherit;
  font-size: 0.9375rem;
  color: {muted};
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  padding: 0.4rem 0.75rem;
  cursor: pointer;
}}
.tabs button[aria-selected="true"] {{ color: {ink}; border-bottom-color: {accent}; }}

footer {{
  border-top: 1px solid {grid};
  padding-top: 1.5rem;
  color: {muted};
  font-size: 0.8125rem;
}}
[hidden] {{ display: none !important; }}
.fallback {{ color: {muted}; font-size: 0.875rem; padding: 1rem 0; }}
"""


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ text.PAGE_TITLE }}</title>
<link rel="stylesheet" href="style.css">
<script src="{{ vega }}"></script>
<script src="{{ vega_lite }}"></script>
<script src="{{ vega_embed }}"></script>
</head>
<body>
<main>

<h1>{{ text.TITLE }}</h1>
<div class="lede">{{ prose(text.LEDE) }}</div>
<p class="caption">{{ caption }}</p>

<div class="controls">
  <span class="label">{{ text.MARKET_PROMPT }}</span>
  <fieldset>
  {% for market in markets %}
    <label>
      <input type="radio" name="market" value="{{ market }}"{% if loop.first %} checked{% endif %}>
      <span class="market">{{ market }}</span>
      <span class="place">{{ captions[market] }}</span>
    </label>
  {% endfor %}
  </fieldset>
</div>

<section>
  <h2>{{ text.S1_HEADING }}</h2>
  <figure class="chart">
    <div class="chart-box" id="chart-skill"></div>
    <details><summary>{{ text.S1_TABLE }}</summary>{{ tables.skill }}</details>
  </figure>
  {{ prose(text.S1_PROSE) }}
</section>

<section>
  <h2>{{ text.S2_HEADING }}</h2>
  {{ tables.operator }}
  {{ prose(text.S2_PROSE) }}
</section>

<section>
  <h2 id="s3-heading">{{ text.S3_HEADING.format(market=markets[0]) }}</h2>
  <div class="controls">
    <label class="label" for="origin">{{ text.S3_SLIDER }}</label>
    <input type="range" id="origin" min="0" step="1" value="0">
    <output id="origin-label" for="origin"></output>
  </div>
  <p class="chart-title" id="forecast-title"></p>
  {% for market in markets %}
  <figure class="chart" data-market="{{ market }}"{% if not loop.first %} hidden{% endif %}>
    <div class="chart-box" id="chart-forecast-{{ market }}"></div>
  </figure>
  {% endfor %}
  <p class="caption">{{ text.S3_CAPTION }}</p>
  <details><summary>{{ text.S3_TABLE }}</summary><div id="forecast-table"></div></details>
</section>

<section>
  <h2>{{ text.S4_HEADING }}</h2>
  <figure class="chart">
    <div class="chart-box" id="chart-coverage"></div>
    <details><summary>{{ text.S4_TABLE }}</summary>{{ tables.coverage }}</details>
  </figure>
  {{ prose(text.S4_PROSE) }}
</section>

<section>
  <h2>{{ text.S5_HEADING }}</h2>
  <div class="tabs" role="tablist">
    <button role="tab" aria-selected="true" aria-controls="panel-hours"
      >{{ text.S5_HOUR_TAB }}</button>
    <button role="tab" aria-selected="false" aria-controls="panel-months"
      >{{ text.S5_MONTH_TAB }}</button>
  </div>
  <div id="panel-hours" role="tabpanel">
    <figure class="chart">
      <div class="chart-box" id="chart-hours"></div>
      <details><summary>{{ text.S5_HOUR_TABLE }}</summary>{{ tables.hours }}</details>
    </figure>
    {{ prose(text.S5_HOUR_PROSE) }}
  </div>
  <div id="panel-months" role="tabpanel" hidden>
    <figure class="chart">
      <div class="chart-box" id="chart-months"></div>
      <details><summary>{{ text.S5_MONTH_TABLE }}</summary>{{ tables.months }}</details>
    </figure>
    {{ prose(text.S5_MONTH_PROSE) }}
  </div>
</section>

<section>
  <h2>{{ text.S6_HEADING }}</h2>
  {{ prose(text.S6_LEDE) }}
  {{ tables.weather }}
  {{ prose(text.S6_PROSE) }}
</section>

<section>
  <h2>{{ text.S7_HEADING }}</h2>
  <figure class="chart">
    <div class="chart-box" id="chart-offsets"></div>
    <details><summary>{{ text.S7_OFFSETS_TABLE }}</summary>{{ tables.offsets }}</details>
  </figure>
  {{ prose(text.S7_PROSE) }}
  {% for market in markets %}
  <figure class="chart" data-market="{{ market }}"{% if not loop.first %} hidden{% endif %}>
    <div class="chart-box" id="chart-holidays-{{ market }}"></div>
    <details>
      <summary>{{ text.S7_HOLIDAY_TABLE.format(market=market) }}</summary>
      {{ tables['holidays_' + market] }}
    </details>
  </figure>
  {% endfor %}
  {{ prose(text.S7_CLOSING) }}
</section>

<section>
  <h2>{{ text.S8_HEADING }}</h2>
  {{ prose(text.S8_PROSE) }}
</section>

<section>
  <h2>{{ text.METHOD_HEADING }}</h2>
  <details><summary>{{ text.METHOD_SUMMARY }}</summary>{{ prose(text.METHOD_PROSE) }}</details>
  <footer>{{ attribution }}</footer>
</section>

</main>

<script>
const SPECS = {{ specs }};
const ORIGINS = {{ origins }};
const TITLES = {{ titles }};
const HEADINGS = {{ headings }};
const MARKETS = {{ markets | tojson }};

const views = {};      // chart-box id -> a promise of its embedded view
const rowCache = {};   // market -> parsed forecast rows, for the table under it
let market = MARKETS[0];
let index = null;      // the slider position, kept here so switching market holds the date

// Charts are the presentation and the tables are the record, so a page whose script host
// blocked the Vega CDN still carries every number. Nothing below reads a chart to work.
const HAS_VEGA = typeof vegaEmbed !== "undefined";

// Every chart waits until it is on screen. A `container` width measured inside a hidden
// tab panel or an unselected market is zero, and Vega only re-reads it on a window resize,
// so a chart embedded while hidden stays a correct specification drawn at no width. This
// also means the page opens without fetching two years of hours for three markets.
const pending = Object.assign({}, {
  "chart-skill": SPECS.skill,
  "chart-coverage": SPECS.coverage,
  "chart-hours": SPECS.hours,
  "chart-months": SPECS.months,
  "chart-offsets": SPECS.offsets,
});
MARKETS.forEach(m => {
  pending["chart-holidays-" + m] = SPECS["holidays_" + m];
  pending["chart-forecast-" + m] = SPECS["forecast_" + m];
});

function reveal() {
  if (!HAS_VEGA) return;
  document.querySelectorAll(".chart-box").forEach(box => {
    const spec = pending[box.id];
    if (!spec || box.offsetParent === null) return;
    delete pending[box.id];
    views[box.id] = vegaEmbed("#" + box.id, spec, opts).then(result => result.view);
  });
}

const slider = document.getElementById("origin");
const originLabel = document.getElementById("origin-label");
const forecastTitle = document.getElementById("forecast-title");
const forecastTable = document.getElementById("forecast-table");

const opts = {actions: false, renderer: "svg"};

// The forecast CSV is fetched a second time here rather than read out of the Vega view,
// whose internal dataset names are not a public interface. The browser serves it from
// cache, so the cost is a parse and not a request.
async function forecastRows(m) {
  if (!rowCache[m]) {
    rowCache[m] = fetch("data/forecast_" + m + ".csv")
      .then(response => response.text())
      .then(body => {
        const lines = body.trim().split("\\n");
        const rows = {};
        for (let i = 1; i < lines.length; i++) {
          const [origin, period, q10, q50, q90] = lines[i].split(",");
          (rows[origin] = rows[origin] || []).push([period, q10, q50, q90]);
        }
        return rows;
      });
  }
  return rowCache[m];
}

function number(value) {
  return Number(value).toLocaleString("en-US");
}

async function drawTable(m, origin) {
  const rows = (await forecastRows(m))[String(origin)] || [];
  const body = rows.map(([period, q10, q50, q90]) =>
    "<tr><td>" + period.replace("T", " ") + "</td>" +
    [q10, q50, q90].map(v => '<td><span class="num">' + number(v) + "</span></td>").join("") +
    "</tr>").join("");
  forecastTable.innerHTML =
    '<div class="table-wrap"><table><thead><tr><th>period</th><th>q10</th><th>q50</th>' +
    "<th>q90</th></tr></thead><tbody>" + body + "</tbody></table></div>";
}

async function setOrigin(at) {
  const {values, labels} = ORIGINS[market];
  const origin = values[at], date = labels[at];
  originLabel.textContent = date;
  forecastTitle.textContent = TITLES[market].replace("{market}", market).replace("{date}", date);
  drawTable(market, origin);
  const view = await views["chart-forecast-" + market];
  if (view) view.signal("originIdx", origin).run();
}

// The slider position is held rather than read back off the input, so switching market
// keeps the date the reader was looking at instead of snapping to the middle, and index 0
// survives the round trip that a falsy check would eat.
function setMarket(next) {
  market = next;
  document.querySelectorAll("[data-market]").forEach(el => {
    el.hidden = el.dataset.market !== market;
  });
  document.getElementById("s3-heading").textContent = HEADINGS[market];
  const {values, initial} = ORIGINS[market];
  if (index === null) index = initial;
  index = Math.min(index, values.length - 1);
  slider.max = values.length - 1;
  slider.value = index;
  reveal();
  setOrigin(index);
}

document.querySelectorAll('input[name="market"]').forEach(input => {
  input.addEventListener("change", () => setMarket(input.value));
});
slider.addEventListener("input", () => {
  index = Number(slider.value);
  setOrigin(index);
});

document.querySelectorAll('[role="tab"]').forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll('[role="tab"]').forEach(other => {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
    });
    reveal();
  });
});

if (!HAS_VEGA) {
  document.querySelectorAll(".chart-box").forEach(box => {
    box.innerHTML = '<p class="fallback">This chart needs the Vega runtime, which did not ' +
      "load. Every number it draws is in the table view beneath it.</p>";
  });
}
setMarket(market);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
