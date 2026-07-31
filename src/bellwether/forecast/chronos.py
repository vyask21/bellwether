"""Chronos-Bolt, a time-series foundation model, behind the Forecaster protocol.

Chronos-Bolt is zero-shot: it was pretrained on a large corpus of time series and is not
fitted to this data. So there is no training step and no risk of leaking the test period
through fitting, but also no reason to assume it beats a baseline tuned to the seasonality
of this specific series. That is what the backtest decides.

Requires the `forecast` extra: pip install -e ".[forecast]"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bellwether.eval.metrics import DEFAULT_QUANTILES

log = logging.getLogger(__name__)

DEFAULT_MODEL = "amazon/chronos-bolt-base"


class ChronosBolt:
    """Zero-shot quantile forecasts from a pretrained Chronos-Bolt checkpoint.

    The model is loaded lazily on first use, so constructing this is cheap and importing
    the module does not pull weights.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cpu",
        name: str | None = None,
        pipeline: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.name = name or model_id.split("/")[-1].replace("-", "_")
        self._pipeline = pipeline

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._load()
        return self._pipeline

    def _load(self) -> Any:
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Chronos requires the forecast extra: pip install -e '.[forecast]'"
            ) from exc

        log.info("loading %s onto %s", self.model_id, self.device)
        return BaseChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self.device,
            # float32 on CPU. bfloat16 saves memory but is slow without hardware support,
            # and the memory is not the constraint at this context length.
            torch_dtype=torch.float32,
        )

    @property
    def context_limit(self) -> int:
        """Longest context the checkpoint accepts, used to trim history."""
        return int(getattr(self.pipeline, "model_context_length", 2048))

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        import torch

        context = self._prepare_context(history)

        quantiles, _mean = self.pipeline.predict_quantiles(
            inputs=torch.tensor(context, dtype=torch.float32).unsqueeze(0),
            prediction_length=horizon,
            quantile_levels=list(quantile_levels),
        )

        # (batch, horizon, n_quantiles) with a single series in the batch.
        forecast = quantiles[0].to(torch.float32).cpu().numpy()

        if forecast.shape != (horizon, len(quantile_levels)):
            raise RuntimeError(
                f"{self.name} returned {forecast.shape}, expected "
                f"({horizon}, {len(quantile_levels)})"
            )
        return forecast

    def _prepare_context(self, history: np.ndarray) -> np.ndarray:
        """Trim history to the most recent values the model can accept.

        Trimming keeps the tail rather than the head: recent observations carry the
        seasonal phase the forecast continues from.
        """
        context = np.asarray(history, dtype=float)
        if context.size == 0:
            raise ValueError(f"{self.name} received an empty history")

        limit = self.context_limit
        if context.size > limit:
            context = context[-limit:]
        return context


class ChronosBoltSmall(ChronosBolt):
    """The 48M-parameter checkpoint. Roughly 4x faster than base on CPU."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__("amazon/chronos-bolt-small", device=device)
