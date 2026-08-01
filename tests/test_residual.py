"""The residual corrector and the weather ablation.

The tests that matter most here are the leakage ones. A corrector that sees the window it
is forecasting produces a beautiful result and a worthless one, and nothing about the
output looks wrong when it happens.
"""

from __future__ import annotations

import numpy as np
import pytest

from bellwether.eval.ablation import (
    LAG_HOURS,
    CachedForecast,
    run_weather_ablation,
    usable_origins,
)
from bellwether.eval.metrics import DEFAULT_QUANTILES
from bellwether.forecast.residual import (
    CALENDAR_ONLY,
    WEATHER,
    ResidualQuantileCorrector,
    apply_correction,
    build_features,
    fit_quantile_regression,
)


class TestQuantileRegression:
    def test_recovers_a_known_line_at_the_median(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 1))
        design = np.column_stack([np.ones(600), x])
        y = 3.0 + 2.0 * x[:, 0] + rng.normal(scale=0.3, size=600)

        beta = fit_quantile_regression(design, y, 0.5)
        assert beta[0] == pytest.approx(3.0, abs=0.1)
        assert beta[1] == pytest.approx(2.0, abs=0.1)

    def test_upper_quantile_sits_above_the_median(self):
        rng = np.random.default_rng(1)
        design = np.ones((800, 1))
        y = rng.normal(loc=10.0, scale=2.0, size=800)

        median = fit_quantile_regression(design, y, 0.5)[0]
        upper = fit_quantile_regression(design, y, 0.9)[0]

        assert upper > median
        # An intercept-only quantile regression estimates the quantile of the sample it
        # was given, not of the distribution the sample came from. Comparing against the
        # population value would be testing the sample size, not the solver.
        assert upper == pytest.approx(np.quantile(y, 0.9), abs=0.02)
        assert median == pytest.approx(np.quantile(y, 0.5), abs=0.02)

    def test_covers_the_intended_fraction_of_the_data(self):
        """The property that makes a quantile regression worth fitting at all."""
        rng = np.random.default_rng(2)
        design = np.ones((2000, 1))
        y = rng.normal(loc=0.0, scale=1.0, size=2000)

        cut = fit_quantile_regression(design, y, 0.2)[0]
        assert np.mean(y <= cut) == pytest.approx(0.2, abs=0.03)

    def test_is_robust_to_an_extreme_outlier(self):
        """Median regression should barely move where least squares would be dragged."""
        design = np.ones((101, 1))
        y = np.concatenate([np.zeros(100), [10_000.0]])

        assert fit_quantile_regression(design, y, 0.5)[0] == pytest.approx(0.0, abs=0.5)
        assert np.mean(y) > 90.0


class TestFeatures:
    def _calendar_args(self, n: int = 24) -> dict:
        return {
            "horizon_steps": np.arange(1, n + 1),
            "local_hours": np.arange(n) % 24,
            "is_weekend": np.zeros(n, dtype=bool),
        }

    def test_weather_arm_has_more_columns_than_the_control(self):
        calendar = build_features(CALENDAR_ONLY, **self._calendar_args())
        weather = build_features(
            WEATHER,
            **self._calendar_args(),
            temperature=np.full(24, 20.0),
            temperature_yesterday=np.full(24, 15.0),
        )
        assert weather.shape[1] > calendar.shape[1]
        assert weather.shape[0] == calendar.shape[0] == 24

    def test_hour_of_day_wraps_continuously(self):
        """23:00 and 00:00 are an hour apart, not 23 units apart."""
        features = build_features(
            CALENDAR_ONLY,
            horizon_steps=np.array([1, 2, 3]),
            local_hours=np.array([23, 0, 12]),
            is_weekend=np.zeros(3, dtype=bool),
        )
        sin_cos = features[:, 2:4]
        near = np.linalg.norm(sin_cos[0] - sin_cos[1])
        far = np.linalg.norm(sin_cos[0] - sin_cos[2])
        assert near < far

    def test_degree_days_are_one_sided(self):
        """A cold day contributes heating and no cooling, and the reverse."""
        cold = build_features(
            WEATHER,
            horizon_steps=np.array([1]),
            local_hours=np.array([12]),
            is_weekend=np.zeros(1, dtype=bool),
            temperature=np.array([-5.0]),
            temperature_yesterday=np.array([-5.0]),
        )
        hot = build_features(
            WEATHER,
            horizon_steps=np.array([1]),
            local_hours=np.array([12]),
            is_weekend=np.zeros(1, dtype=bool),
            temperature=np.array([35.0]),
            temperature_yesterday=np.array([35.0]),
        )
        # Columns after the 5 calendar ones: delta, abs_delta, cooling, heating, ...
        assert cold[0, 7] == 0.0 and cold[0, 8] == pytest.approx(23.0)
        assert hot[0, 7] == pytest.approx(17.0) and hot[0, 8] == 0.0

    def test_delta_is_the_change_from_the_same_hour_yesterday(self):
        features = build_features(
            WEATHER,
            horizon_steps=np.array([1]),
            local_hours=np.array([12]),
            is_weekend=np.zeros(1, dtype=bool),
            temperature=np.array([25.0]),
            temperature_yesterday=np.array([15.0]),
        )
        assert features[0, 5] == pytest.approx(10.0)
        assert features[0, 6] == pytest.approx(10.0)

    def test_weather_spec_without_temperature_is_rejected(self):
        with pytest.raises(ValueError, match="needs temperature"):
            build_features(WEATHER, **self._calendar_args())


class TestCorrector:
    def _fit(self, n: int = 500, seed: int = 3):
        rng = np.random.default_rng(seed)
        temperature = rng.uniform(0.0, 40.0, size=n)
        features = build_features(
            WEATHER,
            horizon_steps=rng.integers(1, 25, size=n),
            local_hours=rng.integers(0, 24, size=n),
            is_weekend=rng.random(n) > 0.7,
            temperature=temperature,
            temperature_yesterday=temperature - rng.normal(scale=3.0, size=n),
        )
        # Residual that genuinely depends on cooling load, plus noise.
        residuals = 50.0 * np.maximum(temperature - 18.0, 0.0) + rng.normal(scale=100.0, size=n)
        return features, residuals

    def test_predicted_quantiles_are_monotonic(self):
        """Independently fitted quantile lines can cross; consumers assume they do not."""
        features, residuals = self._fit()
        corrector = ResidualQuantileCorrector(WEATHER).fit(features, residuals)

        predicted = corrector.predict(features)
        assert np.all(np.diff(predicted, axis=1) >= 0)

    def test_output_shape_matches_rows_and_quantile_levels(self):
        features, residuals = self._fit()
        corrector = ResidualQuantileCorrector(WEATHER).fit(features, residuals)

        assert corrector.predict(features[:7]).shape == (7, len(DEFAULT_QUANTILES))

    def test_learns_a_real_temperature_effect(self):
        """A hot hour should get a larger positive correction than a mild one."""
        features, residuals = self._fit()
        corrector = ResidualQuantileCorrector(WEATHER).fit(features, residuals)

        hot = build_features(
            WEATHER,
            horizon_steps=np.array([12]),
            local_hours=np.array([15]),
            is_weekend=np.zeros(1, dtype=bool),
            temperature=np.array([38.0]),
            temperature_yesterday=np.array([38.0]),
        )
        mild = build_features(
            WEATHER,
            horizon_steps=np.array([12]),
            local_hours=np.array([15]),
            is_weekend=np.zeros(1, dtype=bool),
            temperature=np.array([18.0]),
            temperature_yesterday=np.array([18.0]),
        )
        median = len(DEFAULT_QUANTILES) // 2
        assert corrector.predict(hot)[0, median] > corrector.predict(mild)[0, median]

    def test_refuses_to_predict_before_being_fitted(self):
        with pytest.raises(RuntimeError, match="before fit"):
            ResidualQuantileCorrector(WEATHER).predict(np.ones((3, 11)))

    def test_rejects_mismatched_rows(self):
        features, residuals = self._fit(n=100)
        with pytest.raises(ValueError, match="do not match"):
            ResidualQuantileCorrector(WEATHER).fit(features, residuals[:50])

    def test_rejects_an_underdetermined_fit(self):
        features, residuals = self._fit(n=500)
        with pytest.raises(ValueError, match="Cannot fit"):
            ResidualQuantileCorrector(WEATHER).fit(features[:3], residuals[:3])

    def test_correction_is_added_to_the_base_forecast(self):
        base = np.array([100.0, 200.0])
        residual_quantiles = np.array([[-5.0, 0.0, 5.0], [-1.0, 0.0, 1.0]])

        corrected = apply_correction(base, residual_quantiles)
        assert corrected.shape == (2, 3)
        assert corrected[0].tolist() == [95.0, 100.0, 105.0]
        assert corrected[1].tolist() == [199.0, 200.0, 201.0]


class TestUsableOrigins:
    def _series(self, n: int = 2000) -> np.ndarray:
        return np.sin(np.arange(n) * 2 * np.pi / 24) * 100.0 + 1000.0

    def test_skips_origins_whose_weather_is_missing(self):
        series = self._series()
        temperature = np.full(series.size, 20.0)
        temperature[1500:1520] = np.nan

        with_gap = usable_origins(series, temperature, horizon=24, initial_train_size=672)
        without_gap = usable_origins(
            series, np.full(series.size, 20.0), horizon=24, initial_train_size=672
        )
        assert len(with_gap) < len(without_gap)

    def test_requires_yesterdays_weather_as_well_as_the_windows(self):
        """The day-over-day feature reaches back a further 24 hours."""
        series = self._series()
        temperature = np.full(series.size, 20.0)
        # Blank the day *before* an otherwise clean window.
        origin = 1000
        temperature[origin - LAG_HOURS : origin - LAG_HOURS + 24] = np.nan

        origins = usable_origins(series, temperature, horizon=24, initial_train_size=672)
        assert origin not in origins

    def test_skips_origins_whose_demand_is_missing(self):
        series = self._series()
        series[1500:1510] = np.nan
        temperature = np.full(series.size, 20.0)

        origins = usable_origins(series, temperature, horizon=24, initial_train_size=672)
        assert all(np.all(np.isfinite(series[o : o + 24])) for o in origins)


class _ConstantForecaster:
    """A stand-in base model with a fixed, deliberately wrong forecast."""

    name = "constant"

    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def predict(self, history, horizon, quantile_levels=DEFAULT_QUANTILES):
        spread = np.linspace(-50.0, 50.0, len(quantile_levels))
        return self.value + np.tile(spread, (horizon, 1))


class TestAblation:
    def _inputs(self, n: int = 3000, seed: int = 5):
        rng = np.random.default_rng(seed)
        timestamps = np.arange(
            np.datetime64("2025-01-01T00", "ns"),
            np.datetime64("2025-01-01T00", "ns") + np.timedelta64(n, "h"),
            np.timedelta64(1, "h"),
        )
        hours = np.arange(n) % 24
        temperature = (
            20.0 + 10.0 * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(scale=2, size=n)
        )
        # Demand that genuinely responds to cooling load.
        series = (
            1000.0
            + 100.0 * np.sin(2 * np.pi * hours / 24)
            + 40.0 * np.maximum(temperature - 18.0, 0.0)
            + rng.normal(scale=20.0, size=n)
        )
        return series, timestamps, temperature

    def test_all_three_arms_cover_identical_windows(self):
        """Comparing arms scored on different window sets compares window sets."""
        series, timestamps, temperature = self._inputs()

        results = run_weather_ablation(
            _ConstantForecaster(),
            series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
        )

        assert set(results) == {"constant", "constant+calendar", "constant+weather"}
        counts = {name: r.n_windows for name, r in results.items()}
        assert len(set(counts.values())) == 1, counts
        origins = [tuple(w.origin_index for w in r.windows) for r in results.values()]
        assert origins[0] == origins[1] == origins[2]

    def test_weather_beats_the_calendar_control_when_demand_is_weather_driven(self):
        """On data built so temperature drives demand, the weather arm must win.

        If this fails the corrector is not using its temperature columns, and a null
        result on real data would be uninterpretable.
        """
        series, timestamps, temperature = self._inputs()

        results = run_weather_ablation(
            _ConstantForecaster(),
            series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
        )

        weather = results["constant+weather"].summary()
        calendar = results["constant+calendar"].summary()
        assert weather["mae"] < calendar["mae"]

    def test_the_corrector_cannot_see_the_window_it_forecasts(self):
        """The leakage guard.

        A corrupted final window changes the score of that window for every arm, but must
        not change any *earlier* window's score. If it does, information is flowing
        backwards through the fit.
        """
        series, timestamps, temperature = self._inputs()

        clean = run_weather_ablation(
            _ConstantForecaster(),
            series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
        )["constant+weather"]

        # Exactly the final window: origins step 24, so a wider slice would corrupt two
        # windows and the assertion below would flag the second one as a false positive.
        tampered_series = series.copy()
        tampered_series[-24:] += 50_000.0
        tampered = run_weather_ablation(
            _ConstantForecaster(),
            tampered_series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
        )["constant+weather"]

        clean_by_origin = {w.origin_index: w.mae for w in clean.windows}
        tampered_by_origin = {w.origin_index: w.mae for w in tampered.windows}
        last_origin = max(clean_by_origin)

        for origin, value in clean_by_origin.items():
            if origin == last_origin:
                continue
            assert tampered_by_origin[origin] == pytest.approx(value), (
                f"window at origin {origin} changed when only the final window was corrupted"
            )

    def test_refuses_to_run_without_enough_origins_for_warmup(self):
        series, timestamps, temperature = self._inputs(n=800)

        with pytest.raises(ValueError, match="warmup"):
            run_weather_ablation(
                _ConstantForecaster(),
                series,
                timestamps,
                temperature,
                series_id="TEST:D",
                timezone="UTC",
                min_train_origins=60,
            )

    def test_cached_forecasts_are_reused_rather_than_recomputed(self):
        series, timestamps, temperature = self._inputs()
        origins = usable_origins(series, temperature, 24, 672)

        counter = {"calls": 0}

        class _Counting(_ConstantForecaster):
            def predict(self, history, horizon, quantile_levels=DEFAULT_QUANTILES):
                counter["calls"] += 1
                return super().predict(history, horizon, quantile_levels)

        cached = [
            CachedForecast(origin=o, quantiles=_ConstantForecaster().predict(series[:o], 24))
            for o in origins
        ]
        run_weather_ablation(
            _Counting(),
            series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
            cached=cached,
        )

        assert counter["calls"] == 0
