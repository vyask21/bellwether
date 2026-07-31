"""Chronos wrapper behaviour, exercised without downloading weights.

A fake pipeline is injected so these run in CI. The real checkpoint is covered by a
separate test marked `slow`, which is skipped unless the extra is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from bellwether.eval.metrics import DEFAULT_QUANTILES
from bellwether.forecast.base import Forecaster
from bellwether.forecast.chronos import ChronosBolt

torch = pytest.importorskip("torch", reason="requires the forecast extra")


class FakePipeline:
    """Records what it was asked to forecast and returns a correctly shaped answer."""

    model_context_length = 512

    def __init__(self) -> None:
        self.last_inputs: torch.Tensor | None = None
        self.last_prediction_length: int | None = None
        self.last_quantile_levels: list[float] | None = None

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.last_inputs = inputs
        self.last_prediction_length = prediction_length
        self.last_quantile_levels = quantile_levels

        batch = inputs.shape[0]
        quantiles = torch.zeros(batch, prediction_length, len(quantile_levels))
        # Monotonic across the quantile axis, like a real predictive distribution.
        for i in range(len(quantile_levels)):
            quantiles[:, :, i] = float(i)
        mean = torch.zeros(batch, prediction_length)
        return quantiles, mean


@pytest.fixture
def model() -> ChronosBolt:
    return ChronosBolt(pipeline=FakePipeline())


def test_satisfies_the_forecaster_protocol(model: ChronosBolt):
    assert isinstance(model, Forecaster)


def test_returns_horizon_by_quantiles(model: ChronosBolt):
    out = model.predict(np.arange(200, dtype=float), horizon=24)
    assert out.shape == (24, len(DEFAULT_QUANTILES))


def test_passes_requested_quantile_levels_through(model: ChronosBolt):
    levels = (0.1, 0.5, 0.9)
    model.predict(np.arange(200, dtype=float), horizon=12, quantile_levels=levels)
    assert model.pipeline.last_quantile_levels == list(levels)
    assert model.pipeline.last_prediction_length == 12


def test_context_is_trimmed_to_the_model_limit(model: ChronosBolt):
    history = np.arange(5000, dtype=float)
    model.predict(history, horizon=24)

    sent = model.pipeline.last_inputs
    assert sent.shape == (1, 512), "context should be capped at the model limit"


def test_trimming_keeps_the_most_recent_values(model: ChronosBolt):
    """Keeping the head instead of the tail would forecast from the wrong seasonal phase."""
    history = np.arange(5000, dtype=float)
    model.predict(history, horizon=24)

    sent = model.pipeline.last_inputs[0].numpy()
    np.testing.assert_array_equal(sent, history[-512:])
    assert sent[-1] == 4999.0


def test_short_history_is_passed_through_untrimmed(model: ChronosBolt):
    history = np.arange(100, dtype=float)
    model.predict(history, horizon=24)
    assert model.pipeline.last_inputs.shape == (1, 100)


def test_empty_history_is_rejected(model: ChronosBolt):
    with pytest.raises(ValueError, match="empty history"):
        model.predict(np.array([]), horizon=24)


def test_shape_mismatch_from_the_model_is_caught():
    class WrongShapePipeline(FakePipeline):
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            return torch.zeros(1, prediction_length + 5, len(quantile_levels)), None

    model = ChronosBolt(pipeline=WrongShapePipeline())
    with pytest.raises(RuntimeError, match="expected"):
        model.predict(np.arange(200, dtype=float), horizon=24)


def test_model_is_not_loaded_until_first_use():
    """Constructing must not pull weights, so listing models stays cheap."""
    model = ChronosBolt()
    assert model._pipeline is None
    assert model.name == "chronos_bolt_base"
