"""Weather-conditioned correction of a base model's forecast errors.

Chronos-Bolt is univariate: it sees demand history and nothing else, so it cannot know
that tomorrow will be ten degrees hotter than today. This layer asks the narrower and more
interesting question that follows from that, which is not "can weather forecast demand"
but **"does weather explain what a foundation model gets wrong"**.

The method is a quantile regression on the base model's residuals. For each quantile level
it fits

    actual - base_median  ~  features

on residuals from origins strictly before the one being forecast, then rebuilds the
predictive distribution as `base_median + residual_quantiles`.

Rebuilding the whole distribution rather than shifting the base model's own quantiles is
deliberate. A location-only correction cannot change interval width, and interval width is
what the open question about ERCOT's coverage is about. The cost is that the base model's
learned distribution shape is discarded and re-estimated, which is exactly why the
calendar-only corrector exists as a control: it isolates how much of any gain is
recalibration that any correction would deliver, and how much is weather.

**Both feature sets read temperature at the target hour, which is perfect foresight.** No
operator has tomorrow's observed weather. What this measures is the ceiling on what
temperature can contribute, not what it would contribute in production. The prediction
being tested is a falsification, so the ceiling is the right instrument: if perfect
temperature does not close ERCO's coverage gap, imperfect temperature will not either.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from bellwether.eval.metrics import DEFAULT_QUANTILES

log = logging.getLogger(__name__)

# Base temperature for degree-days, 18 C (65 F). The convention in load forecasting, and
# roughly the outdoor temperature at which a building needs neither heating nor cooling.
DEGREE_DAY_BASE_C = 18.0

# Origins of residual history required before the corrector will produce a forecast. Below
# this it is fitting a nine-quantile model on too few windows to be worth anything, and a
# badly fit corrector would make the base model look worse for reasons unrelated to weather.
DEFAULT_MIN_TRAIN_ORIGINS = 60


def fit_quantile_regression(
    features: np.ndarray,
    target: np.ndarray,
    quantile: float,
    ridge: float = 1e-6,
    max_iterations: int = 40,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Linear quantile regression by iteratively reweighted least squares.

    Pinball loss is `|r| * (q if r >= 0 else 1 - q)`, and `|r| = r^2 / |r|`, so the loss is
    a weighted sum of squares whose weights depend on the current residuals. Alternating
    between recomputing weights and solving the weighted least squares problem is a
    majorise-minimise scheme that decreases the loss at every step.

    Chosen over a gradient method because it is deterministic, has no learning rate, and
    converges in a few dozen iterations on problems this small. scipy would provide this,
    but it is not a dependency and pulling one in for thirty lines is a poor trade.
    """
    n_features = features.shape[1]
    identity = np.eye(n_features)

    # Least squares start: already the right answer for the median under symmetric noise,
    # and a good start for the rest.
    beta = np.linalg.solve(
        features.T @ features + ridge * identity,
        features.T @ target,
    )

    # Floors the weight denominator so a residual at zero cannot produce an infinite
    # weight. Scaled to the data, since a floor of 1e-3 means something different for
    # megawatts than for degrees.
    epsilon = max(1e-9, 1e-6 * float(np.std(target)))

    for _ in range(max_iterations):
        residuals = target - features @ beta
        weights = np.where(residuals >= 0, quantile, 1.0 - quantile) / np.maximum(
            np.abs(residuals), epsilon
        )
        weighted = features * weights[:, None]
        try:
            updated = np.linalg.solve(
                features.T @ weighted + ridge * identity,
                weighted.T @ target,
            )
        except np.linalg.LinAlgError:  # pragma: no cover - guarded by the ridge term
            log.warning("quantile regression hit a singular system at q=%s", quantile)
            break
        if np.max(np.abs(updated - beta)) < tolerance:
            beta = updated
            break
        beta = updated

    return beta


@dataclass(slots=True)
class FeatureSpec:
    """Which columns a corrector builds, and what to call the resulting arm."""

    name: str
    use_weather: bool


CALENDAR_ONLY = FeatureSpec(name="calendar", use_weather=False)
WEATHER = FeatureSpec(name="weather", use_weather=True)


def build_features(
    spec: FeatureSpec,
    horizon_steps: np.ndarray,
    local_hours: np.ndarray,
    is_weekend: np.ndarray,
    temperature: np.ndarray | None = None,
    temperature_yesterday: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the design matrix, one row per forecast hour.

    Calendar columns, present in both arms:

    * `horizon_step`, because error grows with distance from the origin and the correction
      has to be allowed to grow with it.
    * hour of day as a sine and cosine pair, so that 23:00 and 00:00 are adjacent rather
      than 23 units apart.
    * a weekend flag.

    Weather columns, in the treatment arm only:

    * **`delta_t`, the change from the same hour yesterday.** The most important column by
      construction: it is precisely the information a univariate model cannot recover from
      demand history, since yesterday's demand already encodes yesterday's weather.
    * `abs_delta_t`, so the interval can widen symmetrically when tomorrow differs sharply
      from today in either direction.
    * cooling and heating degrees, which carry the nonlinearity. Load responds to 35 C far
      more steeply than to 20 C, and a linear term in temperature cannot express that.
    * the day-over-day change in each, which is the nonlinearity in increment form.
    """
    columns = [
        np.ones_like(horizon_steps, dtype=float),
        horizon_steps.astype(float),
        np.sin(2.0 * np.pi * local_hours / 24.0),
        np.cos(2.0 * np.pi * local_hours / 24.0),
        is_weekend.astype(float),
    ]

    if spec.use_weather:
        if temperature is None or temperature_yesterday is None:
            raise ValueError(f"Feature spec {spec.name!r} needs temperature columns")
        cooling = np.maximum(temperature - DEGREE_DAY_BASE_C, 0.0)
        heating = np.maximum(DEGREE_DAY_BASE_C - temperature, 0.0)
        cooling_yesterday = np.maximum(temperature_yesterday - DEGREE_DAY_BASE_C, 0.0)
        heating_yesterday = np.maximum(DEGREE_DAY_BASE_C - temperature_yesterday, 0.0)
        delta = temperature - temperature_yesterday
        columns.extend(
            [
                delta,
                np.abs(delta),
                cooling,
                heating,
                cooling - cooling_yesterday,
                heating - heating_yesterday,
            ]
        )

    return np.column_stack(columns)


class ResidualQuantileCorrector:
    """Fits residual quantiles on past origins and applies them to the next forecast.

    Holds no state between fits: `fit` is called afresh at each origin with exactly the
    residuals available before it, which is what keeps the evaluation honest. There is no
    path by which a later origin's data can reach an earlier fit.
    """

    def __init__(
        self,
        spec: FeatureSpec,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> None:
        self.spec = spec
        self.quantile_levels = quantile_levels
        self._coefficients: np.ndarray | None = None
        self._centre: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> ResidualQuantileCorrector:
        """Fit one linear model per quantile level on `residuals`."""
        if features.shape[0] != residuals.size:
            raise ValueError(
                f"Feature rows {features.shape[0]} do not match residuals {residuals.size}"
            )
        if features.shape[0] < features.shape[1]:
            raise ValueError(
                f"Cannot fit {features.shape[1]} coefficients on {features.shape[0]} rows"
            )

        standardised = self._standardise(features, fit=True)
        self._coefficients = np.column_stack(
            [fit_quantile_regression(standardised, residuals, q) for q in self.quantile_levels]
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Residual quantiles for each row, shape (rows, n_quantiles).

        Quantile levels are fitted independently, so nothing in the fit stops the 0.6 line
        from crossing below the 0.5 line where the data is thin. Sorting each row restores
        monotonicity, which every consumer of a quantile forecast assumes and which the
        coverage metric would otherwise read as a negative-width interval.
        """
        if self._coefficients is None:
            raise RuntimeError("Corrector used before fit")
        predicted = self._standardise(features, fit=False) @ self._coefficients
        return np.sort(predicted, axis=1)

    def _standardise(self, features: np.ndarray, fit: bool) -> np.ndarray:
        """Centre and scale, so the ridge term penalises every column comparably.

        Without this, degree-days in the tens and a weekend flag in {0, 1} would be
        regularised on wildly different effective scales.
        """
        if fit:
            self._centre = features.mean(axis=0)
            self._scale = features.std(axis=0)
            # The intercept column is constant by design; leave it alone rather than
            # dividing it to zero.
            constant = self._scale < 1e-12
            self._centre = np.where(constant, 0.0, self._centre)
            self._scale = np.where(constant, 1.0, self._scale)
        if self._centre is None or self._scale is None:
            raise RuntimeError("Corrector used before fit")
        return (features - self._centre) / self._scale


def apply_correction(base_median: np.ndarray, residual_quantiles: np.ndarray) -> np.ndarray:
    """Rebuild a predictive distribution as the base point forecast plus residual quantiles."""
    return base_median[:, None] + residual_quantiles
