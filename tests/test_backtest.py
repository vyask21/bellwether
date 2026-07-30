from __future__ import annotations

import numpy as np
import pytest

from bellwether.eval.backtest import rolling_origin_backtest
from bellwether.forecast.baseline import SeasonalNaive


def _series(days: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    hours = np.arange(days * 24)
    signal = 500 + 100 * np.sin(2 * np.pi * hours / 24)
    if noise:
        signal = signal + np.random.default_rng(seed).normal(scale=noise, size=hours.size)
    return signal


def test_backtest_produces_one_result_per_origin():
    series = _series(days=40, noise=10)
    result = rolling_origin_backtest(
        SeasonalNaive(season_length=24),
        series,
        series_id="test",
        horizon=24,
        step=24,
        initial_train_size=24 * 20,
        season_length=24,
    )
    # 40 days total, 20 held for training, horizon 24h, stepping daily.
    assert result.n_windows == 20
    assert len(result.windows) == 20


def test_backtest_never_shows_the_model_future_data():
    """The forecaster must only ever receive history strictly before the origin."""
    seen_lengths: list[int] = []
    origin = 24 * 20

    class SpyForecaster:
        name = "spy"

        def predict(self, history, horizon, quantile_levels):
            seen_lengths.append(history.size)
            return np.repeat(history[-1], horizon)[:, None] * np.ones((1, len(quantile_levels)))

    rolling_origin_backtest(
        SpyForecaster(),
        _series(days=25),
        series_id="test",
        horizon=24,
        step=24,
        initial_train_size=origin,
        season_length=24,
    )
    assert seen_lengths, "forecaster was never called"
    assert min(seen_lengths) == origin
    assert all(length % 24 == 0 for length in seen_lengths)


def test_windows_with_gaps_are_skipped_not_imputed():
    series = _series(days=40, noise=5)
    series[24 * 25 : 24 * 25 + 3] = np.nan  # a gap inside one test window

    result = rolling_origin_backtest(
        SeasonalNaive(season_length=24),
        series,
        series_id="test",
        horizon=24,
        step=24,
        initial_train_size=24 * 20,
        season_length=24,
    )
    # The affected windows drop out rather than being filled with invented values.
    assert result.n_windows < 20
    assert all(np.isfinite(w.mase) for w in result.windows)


def test_summary_reports_signed_coverage_error():
    result = rolling_origin_backtest(
        SeasonalNaive(season_length=24),
        _series(days=40, noise=15),
        series_id="test",
        horizon=24,
        step=24,
        initial_train_size=24 * 20,
        season_length=24,
    )
    summary = result.summary()
    assert summary["coverage_80_error"] == pytest.approx(summary["coverage_80"] - 0.80)


def test_rejects_series_too_short_for_one_window():
    with pytest.raises(ValueError, match="need at least"):
        rolling_origin_backtest(
            SeasonalNaive(season_length=24),
            _series(days=5),
            series_id="short",
            horizon=24,
            initial_train_size=24 * 10,
            season_length=24,
        )
