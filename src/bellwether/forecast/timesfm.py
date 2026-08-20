"""TimesFM 2.5, a second foundation model, behind the same Forecaster protocol.

The point of a second model is to tell a property of Chronos-Bolt apart from a property of
foundation models in general. Every distributional claim in this project so far rests on
one checkpoint, and the ones that matter most, that the interval is well conditioned and
runs a few points under nominal, are exactly the kind that could belong to either.

**The torch build, not the JAX one.** TimesFM ships `[torch]` and `[flax]` extras; the
flax path pulls `jax[cuda]`, which was the known risk on this machine. The torch extra
adds the `timesfm` package and nothing else: torch is already here for Chronos.

**Context is capped at Chronos's limit rather than at this model's.** TimesFM 2.5 accepts a
far longer context, and letting it read years of history where Chronos reads 2,048 hours
would compare two different amounts of evidence and call the difference a model
difference. The cap is a control, not a limitation, and `context_limit` lifts it.

That arm now exists: `LONG_CONTEXT` reads to this checkpoint's own ceiling instead, which
is what separates what the extra history is worth from what the architecture is worth. It
is a second arm rather than a replacement, because the matched one is what makes finding 21
a statement about models. Both are scored over the same windows.

Requires the `forecast` extra: pip install -e ".[forecast]"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bellwether.eval.metrics import DEFAULT_QUANTILES

log = logging.getLogger(__name__)

DEFAULT_MODEL = "google/timesfm-2.5-200m-pytorch"

# Chronos-Bolt's context limit, adopted so the two models read the same history.
MATCHED_CONTEXT = 2048

# The ceiling the checkpoint enforces, and the patch length the horizon is rounded up to
# before that ceiling is checked. Hard-coded rather than read off the definition class,
# which lives behind the lazy import that keeps this module free of torch at import time.
CONTEXT_CEILING = 16384
OUTPUT_PATCH = 128

# This checkpoint's own limit, for the arm that measures what the longer context buys.
# Not the 16,360 the arithmetic suggests: a 24-hour horizon is rounded up to the 128-point
# output patch *before* the ceiling is tested, so the real headroom is 16,256. That is also
# a multiple of the 32-point input patch, which matters because the library rounds a ragged
# context up rather than down -- back into the ceiling it had just cleared.
LONG_CONTEXT = CONTEXT_CEILING - OUTPUT_PATCH

# What the quantile head emits, in the order it emits them. Channel 0 of the output is the
# mean and channels 1 to 9 are these, so the mean has to be dropped rather than mistaken
# for a median. It is not one: on a real forecast it lands between the 0.5 and 0.6 columns.
EMITTED_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class TimesFM:
    """Zero-shot quantile forecasts from a pretrained TimesFM 2.5 checkpoint.

    Loaded lazily on first use, so constructing this is cheap and importing the module
    pulls no weights.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        context_limit: int = MATCHED_CONTEXT,
        name: str | None = None,
        model: Any | None = None,
    ) -> None:
        if context_limit < 1:
            raise ValueError(f"context_limit must be positive, got {context_limit}")
        if context_limit > LONG_CONTEXT:
            # Caught here rather than at the first forecast, where the library reports the
            # rounded-up horizon and the arithmetic stops looking like the number asked for.
            raise ValueError(
                f"context_limit {context_limit} exceeds this checkpoint's {LONG_CONTEXT}: "
                f"the {CONTEXT_CEILING} ceiling is tested against the horizon rounded up "
                f"to {OUTPUT_PATCH}"
            )

        self.model_id = model_id
        self.context_limit = context_limit
        self.name = name or self._default_name(context_limit)
        self._model = model
        self._compiled_horizon = 0

    @staticmethod
    def _default_name(context_limit: int) -> str:
        """Name the arm after the history it reads.

        Two contexts must not share a name. `run_backtest.py` skips a model whose name is
        already in the results file, so a long-context run under the matched name would
        report success and leave the matched numbers untouched.
        """
        if context_limit == MATCHED_CONTEXT:
            return "timesfm_2p5_200m"
        if context_limit == LONG_CONTEXT:
            return "timesfm_2p5_200m_long"
        return f"timesfm_2p5_200m_ctx{context_limit}"

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        try:
            import timesfm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TimesFM requires the forecast extra: pip install -e '.[forecast]'"
            ) from exc

        log.info("loading %s", self.model_id)
        return timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)

    def _compile_for(self, horizon: int) -> None:
        """Fix the context and horizon the graph is built for, once.

        Recompiled only when a longer horizon arrives, which never happens in this project
        but is guarded anyway: the model accepts a horizon longer than it was compiled for
        without raising, and silence is not a guarantee that it did the right thing.

        Two flags are set and **neither turned out to matter on this data**, which is worth
        recording rather than leaving as an implied claim. `normalize_inputs` moves the
        forecast by about one part in ten million, at demand scale and at a thousand times
        it, because the checkpoint normalises its context either way. `fix_quantile_crossing`
        never fired: 40 real ERCO windows produced no inverted level with it off. Both stay
        on, because the metrics do assume ordered levels and insurance that costs nothing
        is worth keeping, but no number in this project rests on either.
        """
        if horizon <= self._compiled_horizon:
            return

        from timesfm import ForecastConfig

        self.model.compile(
            ForecastConfig(
                max_context=self.context_limit,
                max_horizon=horizon,
                normalize_inputs=True,
                fix_quantile_crossing=True,
            )
        )
        self._compiled_horizon = horizon

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        columns = self._quantile_columns(quantile_levels)
        context = self._prepare_context(history)
        self._compile_for(horizon)

        _point, quantiles = self.model.forecast(horizon=horizon, inputs=[context])

        # (batch, horizon, 10) with a single series in the batch.
        forecast = np.asarray(quantiles, dtype=float)[0][:, columns]

        if forecast.shape != (horizon, len(quantile_levels)):
            raise RuntimeError(
                f"{self.name} returned {forecast.shape}, expected "
                f"({horizon}, {len(quantile_levels)})"
            )
        return forecast

    def _quantile_columns(self, quantile_levels: tuple[float, ...]) -> list[int]:
        """Which output channels answer the levels asked for.

        Chronos interpolates to whatever levels it is handed. This head emits nine fixed
        deciles, so a level it cannot produce is refused rather than quietly served by its
        nearest neighbour: a 0.05 answered with a 0.1 would silently narrow every interval
        built from it, and nothing downstream would look wrong.
        """
        columns = []
        for level in quantile_levels:
            matches = [i for i, q in enumerate(EMITTED_QUANTILES, start=1) if abs(q - level) < 1e-9]
            if not matches:
                raise ValueError(
                    f"{self.name} emits the deciles 0.1 to 0.9 only, and was asked for {level}"
                )
            columns.append(matches[0])
        return columns

    def _prepare_context(self, history: np.ndarray) -> np.ndarray:
        """Trim history to the most recent values the model is compiled to read.

        The model does this itself, and keeps the tail as it should: handed 3,000 points
        against a 512 limit it returns exactly what the last 512 alone return. Done here
        anyway, because the context length is then a property of this wrapper that a test
        can assert rather than one inherited from a library that reports neither the limit
        it applied nor the end it kept.
        """
        context = np.asarray(history, dtype=float)
        if context.size == 0:
            raise ValueError(f"{self.name} received an empty history")

        if context.size > self.context_limit:
            context = context[-self.context_limit :]
        return context
