"""The forecaster interface every model implements.

Keeping statistical baselines and foundation models behind one protocol is what makes the
champion/challenger comparison honest — the backtest harness cannot tell them apart, so
no model gets an accidental advantage from a different evaluation path.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from bellwether.eval.metrics import DEFAULT_QUANTILES


@runtime_checkable
class Forecaster(Protocol):
    """Produces quantile forecasts from a univariate history."""

    name: str

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        """Return a (horizon, n_quantiles) array of forecasts.

        `history` is the observed series up to the forecast origin, oldest first.
        Implementations must not look beyond it — that would leak the test period.
        """
        ...
