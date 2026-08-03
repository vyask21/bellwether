---
title: Bellwether
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.60.0
app_file: app.py
pinned: false
license: mit
short_description: Probabilistic electricity demand forecasting, measured against controls
---

# Bellwether

Does a time-series foundation model beat the people who run the grid?

Three US balancing authorities, two years of hourly demand, every claim measured against a
control rather than against nothing. Two published findings were withdrawn after
re-measurement and both are still on the page, because a results page that reports only
what survived is not reporting.

## What this Space is

A rendering of committed results. It reads Parquet and JSON checked into the project and
runs no model, makes no API call, and touches no database. That is a deliberate property:
a dashboard that needs a network call to draw a chart it already computed is a dashboard
that breaks in public.

## Sources

Demand is from the U.S. Energy Information Administration. Temperature is from NOAA's
National Centers for Environmental Information, Integrated Surface Database. Both are
public domain; the acknowledgments and retrieval dates are carried in the app footer and
in `snapshot/manifest.json`.

Forecasts and every derived value are produced by this project, not by EIA or NOAA. They
are not authoritative and carry no warranty. This project is not affiliated with or
endorsed by either agency.
