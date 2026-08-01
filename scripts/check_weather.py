"""Sanity-check the weather pipeline against demand before any model uses it.

Every number here is one a physical argument already predicts, which is the point: if
temperature and demand do not line up the way climate says they should, the fault is in
the gridding, the weighting, or the alignment, and no downstream metric will reveal it.

    python scripts/check_weather.py

Checks, in order:

1. **Overlap.** How many hours have both demand and temperature, and where the weather
   record stops relative to demand.
2. **Range.** Temperatures a Texas summer and a Wyoming winter should actually produce.
   Catches a scaling error, a sign error, or a Celsius/Fahrenheit mixup.
3. **Diurnal phase.** The hottest hour of the day, in local time. Mid-afternoon is right;
   a result near local midnight means the series is offset by a timezone.
4. **Correlation with demand.** Signed and seasonal. Summer cooling load makes demand rise
   with temperature; winter heating load makes it fall. A market that shows the wrong sign
   in both seasons is misaligned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bellwether.eval.operator import BA_TIMEZONES
from bellwether.ingest.noaa import stations_for
from bellwether.storage.db import connect
from bellwether.storage.queries import load_market_temperature, load_series

MARKETS = ("CISO", "ERCO", "PACE")


def main() -> None:
    with connect(read_only=True) as conn:
        for market in MARKETS:
            demand = load_series(conn, market, "D")
            temperature = load_market_temperature(conn, market, demand.timestamps)
            _report(market, demand.timestamps, demand.values, temperature)


def _report(
    market: str,
    timestamps: np.ndarray,
    demand: np.ndarray,
    temperature: np.ndarray,
) -> None:
    both = np.isfinite(demand) & np.isfinite(temperature)
    stations = stations_for(market)

    print(f"\n{'=' * 72}\n{market}: {len(stations)} stations\n{'=' * 72}")
    print(f"demand hours       {np.isfinite(demand).sum():>6,}")
    print(f"temperature hours  {np.isfinite(temperature).sum():>6,}")
    print(f"both               {both.sum():>6,}  ({both.mean():.1%} of the demand grid)")

    if not both.any():
        print("NO OVERLAP: check that weather has been ingested for this market")
        return

    overlap = timestamps[both]
    print(f"overlap window     {overlap[0]} .. {overlap[-1]}")
    print(f"weather ends       {timestamps[np.isfinite(temperature)][-1]}")
    print(f"demand ends        {timestamps[np.isfinite(demand)][-1]}")

    usable_temperature = temperature[both]
    print(
        f"temperature C      min {usable_temperature.min():>6.1f}  "
        f"mean {usable_temperature.mean():>6.1f}  max {usable_temperature.max():>6.1f}"
    )

    # Local time, because "the hottest hour of the day" is a statement about the sun.
    local = pd.DatetimeIndex(timestamps[both]).tz_localize("UTC").tz_convert(BA_TIMEZONES[market])
    local_hour = local.hour.to_numpy()

    by_hour = [usable_temperature[local_hour == h].mean() for h in range(24)]
    print(f"warmest local hour {int(np.argmax(by_hour)):>6d}:00  (expect 14-17)")
    print(f"coolest local hour {int(np.argmin(by_hour)):>6d}:00  (expect 5-8)")

    usable_demand = demand[both]
    print(f"correlation        {np.corrcoef(usable_temperature, usable_demand)[0, 1]:>6.3f}")

    months = local.month.to_numpy()
    summer = np.isin(months, (6, 7, 8))
    winter = np.isin(months, (12, 1, 2))
    for label, mask, expected in (
        ("summer (JJA)", summer, "positive: cooling load"),
        ("winter (DJF)", winter, "negative or weak: heating load"),
    ):
        if mask.sum() < 24:
            print(f"  {label:<13} too few hours to report")
            continue
        r = np.corrcoef(usable_temperature[mask], usable_demand[mask])[0, 1]
        print(f"  {label:<13} r = {r:>6.3f}   ({mask.sum():,} hours, expect {expected})")


if __name__ == "__main__":
    main()
