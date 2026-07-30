from __future__ import annotations

import numpy as np
import pytest

from bellwether.forecast.baseline import DailySeasonalNaive, SeasonalNaive, WeeklySeasonalNaive


def _daily_cycle(days: int, amplitude: float = 100.0, base: float = 500.0) -> np.ndarray:
    hours = np.arange(days * 24)
    return base + amplitude * np.sin(2 * np.pi * hours / 24)


def test_point_forecast_repeats_last_season():
    series = _daily_cycle(days=10)
    model = SeasonalNaive(season_length=24)
    forecasts = model.predict(series, horizon=24)
    median = forecasts[:, 4]  # 0.5 quantile in DEFAULT_QUANTILES

    # A pure daily cycle repeats exactly, so seasonal-naive should be near perfect.
    np.testing.assert_allclose(median, series[-24:], atol=1e-6)


def test_forecast_shape_matches_horizon_and_quantiles():
    series = _daily_cycle(days=10)
    model = SeasonalNaive(season_length=24)
    levels = (0.1, 0.5, 0.9)
    assert model.predict(series, horizon=48, quantile_levels=levels).shape == (48, 3)


def test_quantiles_are_monotonically_increasing():
    rng = np.random.default_rng(42)
    series = _daily_cycle(days=20) + rng.normal(scale=20, size=20 * 24)
    forecasts = SeasonalNaive(season_length=24).predict(series, horizon=24)

    diffs = np.diff(forecasts, axis=1)
    assert np.all(diffs >= -1e-9), "quantile crossing detected"


def test_horizon_beyond_one_season_wraps():
    series = _daily_cycle(days=10)
    forecasts = SeasonalNaive(season_length=24).predict(series, horizon=48)
    median = forecasts[:, 4]
    np.testing.assert_allclose(median[:24], median[24:], atol=1e-9)


def test_requires_two_full_seasons_of_history():
    model = SeasonalNaive(season_length=24)
    with pytest.raises(ValueError, match="two full seasons"):
        model.predict(np.arange(30, dtype=float), horizon=24)


def test_named_variants_use_expected_periods():
    assert DailySeasonalNaive().season_length == 24
    assert WeeklySeasonalNaive().season_length == 168
