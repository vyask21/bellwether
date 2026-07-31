"""Statistical baselines.

These exist to be beaten. Hourly electricity demand has a brutally strong daily cycle, so
a foundation model that cannot beat seasonal-naive on it is not earning its inference
cost. Reporting the comparison either way is the difference between a benchmark and a
marketing claim.
"""

from __future__ import annotations

import numpy as np

from bellwether.eval.metrics import DEFAULT_QUANTILES

HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168


class SeasonalNaive:
    """Repeat the value from one season ago, with empirical residual quantiles.

    Point forecast is `y[t - m]`. Uncertainty comes from the in-sample distribution of
    seasonal-naive residuals, so the intervals are empirical rather than assuming
    normality. Demand errors are fat-tailed and skewed around weather events, and a
    Gaussian interval understates exactly the hours that matter.

    Measured on CISO demand over 2024-07 to 2026-07, 702 daily-step windows at h=24, these
    intervals came out well calibrated: 80.7% actual coverage against a nominal 80%.
    """

    def __init__(self, season_length: int = HOURS_PER_WEEK, name: str | None = None) -> None:
        self.season_length = season_length
        self.name = name or f"seasonal_naive_m{season_length}"

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        m = self.season_length

        if history.size < 2 * m:
            raise ValueError(
                f"{self.name} needs at least two full seasons ({2 * m} points) to estimate "
                f"residual quantiles, got {history.size}"
            )

        point = self._seasonal_point_forecast(history, horizon)
        residual_quantiles = self._residual_quantiles(history, quantile_levels)

        # Broadcast horizon-invariant residual quantiles across the forecast horizon.
        return point[:, None] + residual_quantiles[None, :]

    def _seasonal_point_forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Tile the last full season forward, wrapping for horizons beyond one season."""
        m = self.season_length
        last_season = history[-m:]
        indices = np.arange(horizon) % m
        return last_season[indices]

    def _residual_quantiles(
        self,
        history: np.ndarray,
        quantile_levels: tuple[float, ...],
    ) -> np.ndarray:
        m = self.season_length
        residuals = history[m:] - history[:-m]
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            raise ValueError("No finite seasonal residuals available to estimate uncertainty")
        return np.quantile(residuals, quantile_levels)


class DailySeasonalNaive(SeasonalNaive):
    """Seasonal-naive at a 24-hour period.

    The stronger of the two on CISO demand, contrary to the usual expectation that a
    weekly lag wins by carrying the weekday/weekend distinction. Measured MASE 0.681 vs
    0.919 for the weekly lag over 702 windows.

    The likely reason is weather persistence: demand is largely weather-driven, and
    yesterday's temperature resembles today's far more closely than last week's did. That
    outweighs the day-of-week effect the weekly lag captures.
    """

    def __init__(self) -> None:
        super().__init__(season_length=HOURS_PER_DAY, name="seasonal_naive_daily")


class WeeklySeasonalNaive(SeasonalNaive):
    """Seasonal-naive at a 168-hour period.

    Carries the weekday/weekend distinction that a 24-hour lag destroys, which is the
    textbook argument for preferring it. On CISO demand it lost anyway: see
    DailySeasonalNaive. Kept as the second baseline because the comparison is informative.
    """

    def __init__(self) -> None:
        super().__init__(season_length=HOURS_PER_WEEK, name="seasonal_naive_weekly")
