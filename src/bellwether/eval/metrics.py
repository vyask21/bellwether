"""Forecast accuracy metrics.

Point accuracy alone hides the thing that matters operationally: whether the stated
uncertainty is honest. So every metric here is reported alongside quantile loss and
interval coverage rather than on its own.
"""

from __future__ import annotations

import numpy as np

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def seasonal_naive_mae(train: np.ndarray, season_length: int) -> float:
    """In-sample MAE of a seasonal-naive forecast: the MASE denominator.

    Computed on the training window only, so it is never contaminated by the test period.
    """
    if train.size <= season_length:
        raise ValueError(
            f"Need more than {season_length} training points to scale MASE, got {train.size}"
        )
    diffs = np.abs(train[season_length:] - train[:-season_length])

    # Drop differences touching a gap. A single missing hour anywhere in a long history
    # would otherwise make the whole denominator NaN and silently void every MASE that
    # depends on it, including for forecast windows that are themselves complete.
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("No finite seasonal differences in the training window to scale MASE")

    denominator = float(np.mean(diffs))
    if denominator == 0.0:
        raise ValueError("Seasonal-naive MAE is zero; series is perfectly periodic or constant")
    return denominator


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    train: np.ndarray,
    season_length: int,
) -> float:
    """Mean Absolute Scaled Error.

    Scale-free, so errors are comparable across balancing authorities of very different
    size. MASE < 1 means the model beats a seasonal-naive forecast; MASE > 1 means it
    does not, which is the only baseline comparison that cannot be gamed.
    """
    return mae(actual, predicted) / seasonal_naive_mae(train, season_length)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE, reported as a percentage."""
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    safe = denominator != 0
    if not np.any(safe):
        return 0.0
    return float(np.mean(np.abs(actual[safe] - predicted[safe]) / denominator[safe]) * 100.0)


def quantile_loss(actual: np.ndarray, predicted: np.ndarray, q: float) -> float:
    """Pinball loss at level q, summed (not averaged) over the horizon."""
    delta = actual - predicted
    return float(2.0 * np.sum(np.maximum(q * delta, (q - 1.0) * delta)))


def weighted_quantile_loss(
    actual: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
) -> float:
    """Mean weighted quantile loss (WQL): a discrete approximation of CRPS.

    `quantile_forecasts` has shape (horizon, n_quantiles), aligned to `quantile_levels`.
    Normalising by the sum of absolute actuals makes the score scale-free, so a 30 GW
    market and a 3 GW market can sit in the same leaderboard.
    """
    _validate_quantile_shape(quantile_forecasts, actual, quantile_levels)

    scale = float(np.sum(np.abs(actual)))
    if scale == 0.0:
        raise ValueError("Cannot normalise WQL: actuals sum to zero")

    losses = [
        quantile_loss(actual, quantile_forecasts[:, i], q) / scale
        for i, q in enumerate(quantile_levels)
    ]
    return float(np.mean(losses))


def interval_coverage(
    actual: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: tuple[float, ...],
    lower: float,
    upper: float,
) -> float:
    """Fraction of actuals falling inside the [lower, upper] quantile band.

    A well-calibrated 80% interval should cover ~80% of observations. Systematic
    over-coverage means the model is hedging; under-coverage means it is overconfident,
    which is the failure mode that burns an operator acting on the forecast.
    """
    _validate_quantile_shape(quantile_forecasts, actual, quantile_levels)
    lo_idx = _quantile_index(quantile_levels, lower)
    hi_idx = _quantile_index(quantile_levels, upper)

    inside = (actual >= quantile_forecasts[:, lo_idx]) & (actual <= quantile_forecasts[:, hi_idx])
    return float(np.mean(inside))


def _quantile_index(quantile_levels: tuple[float, ...], level: float) -> int:
    matches = [i for i, q in enumerate(quantile_levels) if np.isclose(q, level)]
    if not matches:
        raise ValueError(f"Quantile level {level} not among forecast levels {quantile_levels}")
    return matches[0]


def _validate_quantile_shape(
    quantile_forecasts: np.ndarray,
    actual: np.ndarray,
    quantile_levels: tuple[float, ...],
) -> None:
    if quantile_forecasts.ndim != 2:
        raise ValueError(
            f"Expected 2-D (horizon, n_quantiles) array, got {quantile_forecasts.shape}"
        )
    if quantile_forecasts.shape != (actual.size, len(quantile_levels)):
        raise ValueError(
            f"Forecast shape {quantile_forecasts.shape} does not match "
            f"({actual.size}, {len(quantile_levels)})"
        )
