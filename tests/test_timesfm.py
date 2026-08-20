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
from bellwether.forecast.timesfm import (
    CONTEXT_CEILING,
    EMITTED_QUANTILES,
    LONG_CONTEXT,
    MATCHED_CONTEXT,
    TimesFM,
)

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


# The long-context arm. What these guard is that it stays a *second* arm: a different name,
# a different compiled context, and a ceiling that is refused rather than silently rounded.


@pytest.fixture
def long_model() -> TimesFM:
    return TimesFM(context_limit=LONG_CONTEXT, model=FakeModel())


def test_the_long_arm_reads_to_the_checkpoint_ceiling(long_model: TimesFM):
    long_model.predict(np.arange(20000, dtype=float), horizon=24)
    assert long_model.model.last_inputs[0].shape == (LONG_CONTEXT,)
    assert long_model.model.configs[0].max_context == LONG_CONTEXT


def test_the_long_arm_is_named_apart_from_the_matched_one(long_model: TimesFM):
    """Sharing a name would not raise anywhere. `run_backtest.py` skips a model already in
    the results file, so the long run would print a skip and leave the matched numbers in
    place under a heading that now claims to be about context."""
    assert long_model.name == "timesfm_2p5_200m_long"
    assert TimesFM(model=FakeModel()).name == "timesfm_2p5_200m"


def test_an_intermediate_context_names_itself_after_its_length():
    assert TimesFM(context_limit=4096, model=FakeModel()).name == "timesfm_2p5_200m_ctx4096"


def test_a_context_beyond_the_ceiling_is_refused():
    """The library's own error names the horizon rounded up to 128, so the number it
    complains about is not the number that was asked for. Refused here instead, before any
    weights load, and after a run has not spent an hour finding out."""
    with pytest.raises(ValueError, match="exceeds this checkpoint"):
        TimesFM(context_limit=LONG_CONTEXT + 1)

    with pytest.raises(ValueError, match="must be positive"):
        TimesFM(context_limit=0)


def test_the_ceiling_leaves_room_for_the_rounded_up_horizon():
    """16,256 rather than the 16,360 that 16,384 minus a 24-hour horizon suggests."""
    assert LONG_CONTEXT + 128 == CONTEXT_CEILING
    assert LONG_CONTEXT % 32 == 0  # or the library rounds it back up, over the ceiling


def test_a_short_history_is_not_padded_up_to_the_long_context(long_model: TimesFM):
    """The wrapper hands over what it has. Padding to the compiled length is the library's
    job and it masks what it pads; doing it here would feed the model real zeros."""
    long_model.predict(np.arange(3000, dtype=float), horizon=24)
    assert long_model.model.last_inputs[0].shape == (3000,)
