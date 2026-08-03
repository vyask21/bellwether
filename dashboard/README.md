---
title: Bellwether
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Probabilistic demand forecasting, measured against controls
---

# Bellwether

Does a time-series foundation model beat the people who run the grid?

Three US balancing authorities, two years of hourly demand, every claim measured against a
control rather than against nothing. Two published findings were withdrawn after
re-measurement and both are still on the page, because a results page that reports only
what survived is not reporting.

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
