---
title: Bellwether
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Diagnosing grid demand anomalies from stored evidence
---

# Bellwether

When grid demand departs from what was expected, what caused it?

This forecasts demand for three US balancing authorities with an uncertainty band, flags
the hours that fall outside it, and attributes each of those episodes to a cause computed
from stored history. A cause was found for 28 of the 30 largest anomalies, and every
number in an explanation traces to a measurement rather than being written for it.

The forecast is the instrument rather than the argument. A band you cannot trust cannot
tell you an hour is unusual, so most of this page is the work of establishing that it can.
Two published findings were withdrawn after re-measurement and both are still here,
because a results page that reports only what survived is not reporting.

## What this Space is

A rendering of committed results. Every number was computed before the page was built, so
it runs no model, makes no API call, and touches no database. That is a deliberate
property: a dashboard that needs a network call to draw a chart it already computed is a
dashboard that breaks in public.

It is a static site rather than a running app, which is the same statement made stronger.
The charts are Vega-Lite specifications compiled from the project's own Altair code, and
the two files each market loads are its observed demand and its stored forecasts. There is
no server here to be down.

## Sources

Demand is from the U.S. Energy Information Administration. Temperature is from NOAA's
National Centers for Environmental Information, Integrated Surface Database. Both are
public domain; the acknowledgments and retrieval dates are carried in the app footer and
in `snapshot/manifest.json`.

Forecasts and every derived value are produced by this project, not by EIA or NOAA. They
are not authoritative and carry no warranty. This project is not affiliated with or
endorsed by either agency.
