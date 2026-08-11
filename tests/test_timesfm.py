"""TimesFM wrapper behaviour, exercised without downloading weights.

A fake model is injected so these run wherever the package is installed. What they are
really guarding is the join between two conventions: the head emits a mean followed by
nine deciles, and this project asks for nine quantile levels. Off by one there produces a
complete, plausible, and wrong forecast, which is the failure mode this file exists for.
"""

from __future__ import annotations

import numpy as np
import pytest

from bellwether.eval.metrics import DEFAULT_QUANTILES
from bellwether.forecast.base import Forecaster
from bellwether.forecast.timesfm import EMITTED_QUANTILES, MATCHED_CONTEXT, TimesFM

pytest.importorskip("timesfm", reason="requires the forecast extra")


class FakeModel:
    """Records what it was asked for and answers in the real output's shape.

    The channel values are the giveaway: channel 0 carries a mean that is deliberately not
    the median, and channels 1 to 9 carry their own level times 1,000. A wrapper that
    passes the mean through as a quantile, or that reads the channels off by one, produces
    numbers these tests can name exactly.
    """

    def __init__(self) -> None:
        self.last_inputs: list[np.ndarray] | None = None
        self.last_horizon: int | None = None
        self.configs: list = []

    def compile(self, forecast_config) -> None:
        self.configs.append(forecast_config)

    def forecast(self, horizon: int, inputs: list[np.ndarray]):
        self.last_inputs = inputs
        self.last_horizon = horizon

        quantiles = np.zeros((len(inputs), horizon, 1 + len(EMITTED_QUANTILES)))
        quantiles[:, :, 0] = -1.0
        for channel, level in enumerate(EMITTED_QUANTILES, start=1):
            quantiles[:, :, channel] = level * 1000
        point = quantiles[:, :, 5]
        return point, quantiles


@pytest.fixture
def model() -> TimesFM:
    return TimesFM(model=FakeModel())


def test_satisfies_the_forecaster_protocol(model: TimesFM):
    assert isinstance(model, Forecaster)


def test_returns_horizon_by_quantiles(model: TimesFM):
    out = model.predict(np.arange(200, dtype=float), horizon=24)
    assert out.shape == (24, len(DEFAULT_QUANTILES))


def test_the_mean_channel_is_dropped_rather_than_read_as_a_quantile(model: TimesFM):
    """Channel 0 is a mean, and on a real forecast it sits between the 0.5 and 0.6 columns.
    Reading it as the first quantile would shift every level by one and still return a
    correctly shaped, monotonic, entirely wrong band."""
    out = model.predict(np.arange(200, dtype=float), horizon=4)
    np.testing.assert_allclose(out[0], [level * 1000 for level in DEFAULT_QUANTILES])
    assert -1.0 not in out


def test_a_subset_of_levels_selects_the_matching_channels(model: TimesFM):
    out = model.predict(np.arange(200, dtype=float), horizon=4, quantile_levels=(0.1, 0.5, 0.9))
    np.testing.assert_allclose(out[0], [100.0, 500.0, 900.0])


def test_a_level_the_head_cannot_emit_is_refused(model: TimesFM):
    """Answering 0.05 with the 0.1 column would narrow every interval built from it, and
    nothing downstream would look wrong."""
    with pytest.raises(ValueError, match="deciles"):
        model.predict(np.arange(200, dtype=float), horizon=4, quantile_levels=(0.05, 0.5, 0.95))


def test_it_compiles_once_for_a_repeated_horizon(model: TimesFM):
    for _ in range(3):
        model.predict(np.arange(200, dtype=float), horizon=24)
    assert len(model.model.configs) == 1
    assert model.model.configs[0].max_horizon == 24
    assert model.model.configs[0].max_context == MATCHED_CONTEXT


def test_it_recompiles_for_a_longer_horizon(model: TimesFM):
    """The model accepts a horizon longer than it was compiled for without raising, so the
    guard cannot be replaced by trusting it to complain."""
    model.predict(np.arange(200, dtype=float), horizon=24)
    model.predict(np.arange(200, dtype=float), horizon=48)
    model.predict(np.arange(200, dtype=float), horizon=12)
    assert [config.max_horizon for config in model.model.configs] == [24, 48]


def test_the_compiled_configuration_is_the_one_of_record(model: TimesFM):
    """Pinned so a silent change to either flag is visible. Neither is load-bearing on this
    data: normalisation moves the forecast by about one part in ten million, and the
    crossing fix never fired across 40 real windows with it off. They stay on because the
    metrics do assume ordered levels and the insurance costs nothing."""
    assert model.model is not None
    model.predict(np.arange(200, dtype=float), horizon=24)
    assert model.model.configs[0].fix_quantile_crossing is True
    assert model.model.configs[0].normalize_inputs is True


def test_context_is_trimmed_to_the_matched_limit(model: TimesFM):
    """Capped at Chronos's limit on purpose. Two models reading different amounts of
    history would make the comparison one of evidence rather than of method."""
    model.predict(np.arange(5000, dtype=float), horizon=24)
    assert model.model.last_inputs[0].shape == (MATCHED_CONTEXT,)


def test_trimming_keeps_the_most_recent_values(model: TimesFM):
    history = np.arange(5000, dtype=float)
    model.predict(history, horizon=24)

    sent = model.model.last_inputs[0]
    np.testing.assert_array_equal(sent, history[-MATCHED_CONTEXT:])
    assert sent[-1] == 4999.0


def test_short_history_is_passed_through_untrimmed(model: TimesFM):
    model.predict(np.arange(100, dtype=float), horizon=24)
    assert model.model.last_inputs[0].shape == (100,)


def test_empty_history_is_rejected(model: TimesFM):
    with pytest.raises(ValueError, match="empty history"):
        model.predict(np.array([]), horizon=24)


def test_shape_mismatch_from_the_model_is_caught():
    class WrongShapeModel(FakeModel):
        def forecast(self, horizon: int, inputs: list[np.ndarray]):
            point, quantiles = super().forecast(horizon + 5, inputs)
            self.last_horizon = horizon
            return point, quantiles

    model = TimesFM(model=WrongShapeModel())
    with pytest.raises(RuntimeError, match="expected"):
        model.predict(np.arange(200, dtype=float), horizon=24)


def test_model_is_not_loaded_until_first_use():
    model = TimesFM()
    assert model._model is None
    assert model.name == "timesfm_2p5_200m"
