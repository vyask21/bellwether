"""Metrics are checked against hand-computable cases, not just self-consistency."""

from __future__ import annotations

import numpy as np
import pytest

from bellwether.eval.metrics import (
    interval_coverage,
    interval_width,
    mae,
    mase,
    quantile_loss,
    seasonal_naive_mae,
    smape,
    weighted_quantile_loss,
)


def test_mae_is_mean_absolute_difference():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    assert mae(actual, predicted) == pytest.approx((2 + 2 + 3) / 3)


def test_seasonal_naive_mae_uses_lagged_differences():
    train = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
    # Lag-2 differences: |3-1|, |5-2|, |8-3|, |13-5| -> 2, 3, 5, 8
    assert seasonal_naive_mae(train, season_length=2) == pytest.approx(4.5)


def test_seasonal_naive_mae_tolerates_gaps_in_the_training_window():
    """A gap anywhere in history must not void the MASE scale for clean windows."""
    train = np.array([1.0, 2.0, 3.0, 5.0, np.nan, 13.0, 4.0, 6.0])
    scale = seasonal_naive_mae(train, season_length=2)
    # Only the finite lag-2 differences count: |3-1|, |5-2|, |4-nan|->dropped,
    # |13-5|, |6-13| -> 2, 3, 8, 7
    assert scale == pytest.approx((2 + 3 + 8 + 7) / 4)
    assert np.isfinite(scale)


def test_seasonal_naive_mae_rejects_all_gap_window():
    train = np.array([np.nan] * 10)
    with pytest.raises(ValueError, match="No finite seasonal differences"):
        seasonal_naive_mae(train, season_length=2)


def test_mase_of_one_means_parity_with_seasonal_naive():
    train = np.array([0.0, 10.0] * 10)
    actual = np.array([0.0, 10.0])
    # Perfectly periodic training data has zero seasonal-naive error, which is undefined
    # as a scale; the metric must refuse rather than divide by zero.
    with pytest.raises(ValueError, match="zero"):
        mase(actual, actual, train, season_length=2)


def test_mase_scales_error_by_seasonal_difficulty():
    rng = np.random.default_rng(0)
    train = rng.normal(size=200)
    actual = np.array([1.0, 2.0, 3.0])
    predicted = actual + 1.0
    scale = seasonal_naive_mae(train, season_length=24)
    assert mase(actual, predicted, train, 24) == pytest.approx(1.0 / scale)


def test_quantile_loss_penalises_asymmetrically():
    actual = np.array([10.0])
    # At q=0.9 under-forecasting should be punished far harder than over-forecasting.
    under = quantile_loss(actual, np.array([8.0]), q=0.9)
    over = quantile_loss(actual, np.array([12.0]), q=0.9)
    assert under > over
    assert under == pytest.approx(2 * 0.9 * 2)
    assert over == pytest.approx(2 * 0.1 * 2)


def test_weighted_quantile_loss_is_zero_for_perfect_forecast():
    actual = np.array([100.0, 200.0])
    levels = (0.1, 0.5, 0.9)
    forecasts = np.repeat(actual[:, None], len(levels), axis=1)
    assert weighted_quantile_loss(actual, forecasts, levels) == pytest.approx(0.0)


def test_weighted_quantile_loss_rejects_shape_mismatch():
    actual = np.array([1.0, 2.0, 3.0])
    forecasts = np.zeros((2, 3))
    with pytest.raises(ValueError, match="does not match"):
        weighted_quantile_loss(actual, forecasts, (0.1, 0.5, 0.9))


def test_interval_coverage_counts_actuals_inside_the_band():
    levels = (0.1, 0.5, 0.9)
    #                       q10    q50    q90
    forecasts = np.array(
        [
            [0.0, 5.0, 10.0],  # actual 5 -> inside
            [0.0, 5.0, 10.0],  # actual 20 -> outside
            [0.0, 5.0, 10.0],
        ]
    )  # actual 10 -> inside (inclusive bound)
    actual = np.array([5.0, 20.0, 10.0])
    assert interval_coverage(actual, forecasts, levels, 0.1, 0.9) == pytest.approx(2 / 3)


def test_smape_ignores_zero_denominator_points():
    actual = np.array([0.0, 100.0])
    predicted = np.array([0.0, 110.0])
    # First point contributes nothing; second is |100-110| / 105 * 100.
    assert smape(actual, predicted) == pytest.approx(10 / 105 * 100)


class TestIntervalWidth:
    """Sharpness. Coverage alone cannot distinguish calibration from hedging."""

    def test_width_is_the_gap_between_the_named_quantiles(self):
        levels = (0.1, 0.5, 0.9)
        forecasts = np.array([[90.0, 100.0, 110.0], [80.0, 100.0, 130.0]])

        # Widths of 20 and 50.
        assert interval_width(forecasts, levels, 0.1, 0.9) == pytest.approx(35.0)

    def test_a_hedging_forecast_buys_coverage_with_width(self):
        """The pairing that makes reporting coverage alone incomplete."""
        levels = (0.1, 0.5, 0.9)
        actual = np.array([100.0, 100.0])
        tight = np.array([[99.0, 100.0, 101.0], [99.0, 100.0, 101.0]])
        hedged = np.array([[0.0, 100.0, 200.0], [0.0, 100.0, 200.0]])

        assert interval_coverage(actual, hedged, levels, 0.1, 0.9) == 1.0
        assert interval_coverage(actual, tight, levels, 0.1, 0.9) == 1.0
        assert interval_width(hedged, levels, 0.1, 0.9) > interval_width(tight, levels, 0.1, 0.9)

    def test_rejects_a_level_the_forecast_does_not_carry(self):
        with pytest.raises(ValueError, match="not among forecast levels"):
            interval_width(np.array([[1.0, 2.0]]), (0.25, 0.75), 0.1, 0.9)
