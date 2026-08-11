"""The residual corrector and the weather ablation.

The tests that matter most here are the leakage ones. A corrector that sees the window it
is forecasting produces a beautiful result and a worthless one, and nothing about the
output looks wrong when it happens.
"""

from __future__ import annotations

import numpy as np
import pytest

from bellwether.eval.ablation import (
    FEDERAL_ONLY,
    LAG_HOURS,
    ORDINARY,
    WIDELY_OBSERVED,
    CachedForecast,
    holiday_class_flags,
    holiday_flags,
    realised_volatility,
    run_corrector_ablation,
    usable_origins,
)
from bellwether.eval.metrics import DEFAULT_QUANTILES
from bellwether.forecast.residual import (
    CALENDAR_ONLY,
    CLASS_PRIOR_HOURS,
    HOUR_PRIOR_HOURS,
    VOLATILITY,
    WEATHER,
    WEATHER_VOLATILITY,
    HolidayClassScaleCorrector,
    HolidayHourScaleCorrector,
    HolidayScaleCorrector,
    QuantileScaleCorrector,
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

    def test_volatility_spec_without_volatility_is_rejected(self):
        with pytest.raises(ValueError, match="needs volatility"):
            build_features(VOLATILITY, **self._calendar_args())

    def test_volatility_columns_are_appended_after_weather_columns(self):
        """Weather column positions must not shift when volatility is switched on.

        Several tests here index weather columns by number, and more importantly the two
        arms have to stay comparable across runs.
        """
        args = self._calendar_args()
        weather = build_features(
            WEATHER,
            **args,
            temperature=np.full(24, 20.0),
            temperature_yesterday=np.full(24, 15.0),
        )
        both = build_features(
            WEATHER_VOLATILITY,
            **args,
            temperature=np.full(24, 20.0),
            temperature_yesterday=np.full(24, 15.0),
            volatility_24=np.full(24, 100.0),
            volatility_168=np.full(24, 200.0),
        )
        assert both.shape[1] == weather.shape[1] + 3
        assert np.array_equal(both[:, : weather.shape[1]], weather)

    def test_volatility_interaction_grows_with_horizon(self):
        """Uncertainty accumulates with lead time at a rate set by current volatility."""
        features = build_features(
            VOLATILITY,
            horizon_steps=np.array([1, 4, 16]),
            local_hours=np.array([12, 12, 12]),
            is_weekend=np.zeros(3, dtype=bool),
            volatility_24=np.full(3, 10.0),
            volatility_168=np.full(3, 20.0),
        )
        # Columns after the 5 calendar ones: vol_168, vol_24, vol_24 * sqrt(step).
        assert features[:, 5].tolist() == [20.0, 20.0, 20.0]
        assert features[:, 6].tolist() == [10.0, 10.0, 10.0]
        assert features[:, 7].tolist() == [10.0, 20.0, 40.0]


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


class TestQuantileScaleCorrector:
    """Stretches the base interval instead of replacing it.

    Built specifically because the residual corrector destroys the base model's
    conditioning. The test that matters most here is the one asserting this cannot.
    """

    LEVELS = (0.1, 0.5, 0.9)

    def _forecasts(self, half_width: np.ndarray) -> np.ndarray:
        """Symmetric bands about a median of 100, one row per element of `half_width`."""
        return np.column_stack(
            [100.0 - half_width, np.full(half_width.size, 100.0), 100.0 + half_width]
        )

    def test_a_calibrated_base_model_is_left_alone(self):
        """k = 1 at every level means the corrector is the identity."""
        rng = np.random.default_rng(0)
        n = 4000
        actual = 100.0 + rng.normal(scale=10.0, size=n)
        # A band that genuinely covers 80%: +/- 1.2816 sigma.
        base = self._forecasts(np.full(n, 12.816))

        scales = QuantileScaleCorrector(self.LEVELS).fit(base, actual).scales
        assert scales[0] == pytest.approx(1.0, abs=0.06)
        assert scales[2] == pytest.approx(1.0, abs=0.06)

    def test_a_too_narrow_base_model_is_widened(self):
        rng = np.random.default_rng(1)
        n = 4000
        actual = 100.0 + rng.normal(scale=10.0, size=n)
        base = self._forecasts(np.full(n, 6.0))  # far too tight

        scales = QuantileScaleCorrector(self.LEVELS).fit(base, actual).scales
        assert scales[0] > 1.5
        assert scales[2] > 1.5

    def test_a_too_wide_base_model_is_narrowed(self):
        rng = np.random.default_rng(2)
        n = 4000
        actual = 100.0 + rng.normal(scale=10.0, size=n)
        base = self._forecasts(np.full(n, 40.0))

        scales = QuantileScaleCorrector(self.LEVELS).fit(base, actual).scales
        assert scales[0] < 0.7
        assert scales[2] < 0.7

    def test_scaling_reaches_nominal_coverage(self):
        """The whole point: fix the level error."""
        rng = np.random.default_rng(3)
        n = 6000
        actual = 100.0 + rng.normal(scale=10.0, size=n)
        base = self._forecasts(np.full(n, 6.0))

        corrector = QuantileScaleCorrector(self.LEVELS).fit(base, actual)
        scaled = corrector.predict(base)
        covered = np.mean((actual >= scaled[:, 0]) & (actual <= scaled[:, 2]))

        assert np.mean((actual >= base[:, 0]) & (actual <= base[:, 2])) < 0.55
        assert covered == pytest.approx(0.8, abs=0.03)

    def test_conditioning_survives_scaling(self):
        """The property the residual corrector lacks, and the reason this class exists.

        If the base model widens its band for some hours and not others, that ratio must
        come through untouched. A single scalar per level cannot flatten it.
        """
        rng = np.random.default_rng(4)
        n = 4000
        half_width = np.where(np.arange(n) % 2 == 0, 5.0, 20.0)
        sigma = np.where(np.arange(n) % 2 == 0, 4.0, 16.0)
        actual = 100.0 + rng.normal(scale=sigma)
        base = self._forecasts(half_width)

        scaled = QuantileScaleCorrector(self.LEVELS).fit(base, actual).predict(base)
        widths = scaled[:, 2] - scaled[:, 0]

        # Base widths differ 4x between the two groups; so must the scaled ones.
        assert widths[1::2].mean() / widths[0::2].mean() == pytest.approx(4.0, abs=0.01)

    def test_the_median_is_never_moved(self):
        """A scale correction is about spread. Moving the median would be a different claim."""
        rng = np.random.default_rng(5)
        base = self._forecasts(np.full(500, 8.0))
        actual = 100.0 + rng.normal(scale=20.0, size=500)

        scaled = QuantileScaleCorrector(self.LEVELS).fit(base, actual).predict(base)
        assert np.allclose(scaled[:, 1], base[:, 1])

    def test_output_quantiles_stay_sorted(self):
        rng = np.random.default_rng(6)
        base = self._forecasts(rng.uniform(1.0, 30.0, size=500))
        actual = 100.0 + rng.normal(scale=10.0, size=500)

        scaled = QuantileScaleCorrector(self.LEVELS).fit(base, actual).predict(base)
        assert np.all(np.diff(scaled, axis=1) >= 0)

    def test_scales_never_collapse_the_interval(self):
        """A zero scale would report certainty, which is never an honest forecast."""
        base = self._forecasts(np.full(400, 50.0))
        actual = np.full(400, 100.0)  # base is absurdly wide; every ratio is zero

        scales = QuantileScaleCorrector(self.LEVELS).fit(base, actual).scales
        assert np.all(scales > 0.0)

    def test_refuses_to_predict_before_being_fitted(self):
        with pytest.raises(RuntimeError, match="before fit"):
            QuantileScaleCorrector(self.LEVELS).predict(np.ones((3, 3)))

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError, match="forecasts and"):
            QuantileScaleCorrector(self.LEVELS).fit(np.ones((5, 3)), np.ones(4))

    def test_rejects_a_forecast_with_the_wrong_quantile_count(self):
        with pytest.raises(ValueError, match="expected 3"):
            QuantileScaleCorrector(self.LEVELS).fit(np.ones((5, 4)), np.ones(5))


class TestRealisedVolatility:
    """The one feature computed from the target series itself, so leakage is the risk."""

    def test_reads_only_history_before_the_origin(self):
        """The whole point: a forecaster at the origin has not seen what comes after it."""
        series = np.concatenate([np.full(100, 10.0), np.full(100, 10.0)])
        calm = realised_volatility(series, origin=100, lookback=24)

        # Make everything from the origin onward violently volatile.
        series[100:] = np.tile([0.0, 1000.0], 50)
        assert realised_volatility(series, origin=100, lookback=24) == calm == 0.0

    def test_rises_with_a_more_volatile_history(self):
        steady = np.arange(200, dtype=float)
        jumpy = np.arange(200, dtype=float) + np.tile([0.0, 50.0], 100)

        assert realised_volatility(jumpy, 150, 24) > realised_volatility(steady, 150, 24)

    def test_measures_changes_not_levels(self):
        """A hundredfold bigger market with the same swings is not more volatile.

        The corrector is fitted per market and its columns are standardised, so absolute
        scale is handled elsewhere. What this column has to carry is movement.
        """
        small = np.tile([100.0, 110.0], 100)
        large = np.tile([10_000.0, 10_010.0], 100)

        assert realised_volatility(large, 150, 24) == pytest.approx(
            realised_volatility(small, 150, 24)
        )

    def test_a_lookback_longer_than_history_is_clipped(self):
        assert realised_volatility(np.arange(10, dtype=float), origin=5, lookback=168) >= 0.0

    def test_too_little_history_is_zero_rather_than_nan(self):
        """NaN here would silently void every quantile fitted on the column."""
        assert realised_volatility(np.arange(10, dtype=float), origin=1, lookback=24) == 0.0


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

    def test_every_arm_covers_identical_windows(self):
        """Comparing arms scored on different window sets compares window sets."""
        series, timestamps, temperature = self._inputs()

        output = run_corrector_ablation(
            _ConstantForecaster(),
            series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
        )

        results = output.results
        assert set(results) == {
            "constant",
            "constant+calendar",
            "constant+weather",
            "constant+volatility",
            "constant+weather+volatility",
            "constant+scale",
            "constant+scale+holiday",
            "constant+scale+holidayclass",
            "constant+scale+holidayhour",
        }
        counts = {name: r.n_windows for name, r in results.items()}
        assert len(set(counts.values())) == 1, counts
        origins = [tuple(w.origin_index for w in r.windows) for r in results.values()]
        assert len(set(origins)) == 1, "the arms were scored on different origins"

        # The retained forecasts must line up with the origins they were scored on, since
        # the breach analysis pairs them positionally.
        assert len(output.scored_origins) == results["constant"].n_windows
        for name, windows in output.forecasts.items():
            assert len(windows) == len(output.scored_origins), name
        assert list(output.scored_origins) == list(origins[0])

    def test_weather_beats_the_calendar_control_when_demand_is_weather_driven(self):
        """On data built so temperature drives demand, the weather arm must win.

        If this fails the corrector is not using its temperature columns, and a null
        result on real data would be uninterpretable.
        """
        series, timestamps, temperature = self._inputs()

        results = run_corrector_ablation(
            _ConstantForecaster(),
            series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
        ).results

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

        clean = run_corrector_ablation(
            _ConstantForecaster(),
            series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
        ).results["constant+weather"]

        # Exactly the final window: origins step 24, so a wider slice would corrupt two
        # windows and the assertion below would flag the second one as a false positive.
        tampered_series = series.copy()
        tampered_series[-24:] += 50_000.0
        tampered = run_corrector_ablation(
            _ConstantForecaster(),
            tampered_series,
            timestamps,
            temperature,
            series_id="TEST:D",
            timezone="UTC",
            min_train_origins=20,
        ).results["constant+weather"]

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
            run_corrector_ablation(
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
        run_corrector_ablation(
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


class TestHolidayScaleCorrector:
    """Scales the interval and shifts it on holidays.

    Built on the scale corrector rather than the residual one deliberately: the residual
    correctors flatten the base model's seasonal conditioning, and a holiday fix built on
    that would inherit the damage to buy back a few days a year.
    """

    LEVELS = (0.1, 0.5, 0.9)

    def _forecasts(self, n: int, half_width: float = 10.0) -> np.ndarray:
        return np.column_stack(
            [np.full(n, 100.0 - half_width), np.full(n, 100.0), np.full(n, 100.0 + half_width)]
        )

    def _holiday_data(self, n: int = 800, drop: float = 30.0, seed: int = 0):
        """Demand that sits `drop` MW lower on holidays, which is one hour in ten here."""
        rng = np.random.default_rng(seed)
        is_holiday = np.arange(n) % 10 == 0
        actual = 100.0 + rng.normal(scale=5.0, size=n) - np.where(is_holiday, drop, 0.0)
        return self._forecasts(n), actual, is_holiday

    def test_it_learns_the_holiday_shift(self):
        base, actual, is_holiday = self._holiday_data(drop=30.0)
        corrector = HolidayScaleCorrector(self.LEVELS).fit(base, actual, is_holiday)

        assert corrector.offset == pytest.approx(-30.0, abs=2.0)

    def test_the_shift_applies_only_to_holiday_hours(self):
        base, actual, is_holiday = self._holiday_data()
        corrector = HolidayScaleCorrector(self.LEVELS).fit(base, actual, is_holiday)

        probe = self._forecasts(2)
        adjusted = corrector.predict(probe, np.array([True, False]))
        assert adjusted[0, 1] < adjusted[1, 1], "the holiday hour should shift down"
        assert adjusted[1, 1] == pytest.approx(100.0), "an ordinary hour is untouched"

    def test_it_improves_holiday_coverage(self):
        """The point of the whole exercise, stated as a measurement."""
        base, actual, is_holiday = self._holiday_data(drop=30.0)
        corrector = HolidayScaleCorrector(self.LEVELS).fit(base, actual, is_holiday)
        adjusted = corrector.predict(base, is_holiday)

        before = np.mean(
            (actual[is_holiday] >= base[is_holiday, 0])
            & (actual[is_holiday] <= base[is_holiday, 2])
        )
        after = np.mean(
            (actual[is_holiday] >= adjusted[is_holiday, 0])
            & (actual[is_holiday] <= adjusted[is_holiday, 2])
        )
        assert before < 0.2
        assert after > 0.7

    def test_ordinary_hours_are_not_made_worse(self):
        """A calendar fix that costs accuracy on the other 97% of hours is not a fix."""
        base, actual, is_holiday = self._holiday_data()
        corrector = HolidayScaleCorrector(self.LEVELS).fit(base, actual, is_holiday)
        adjusted = corrector.predict(base, is_holiday)

        ordinary = ~is_holiday
        assert np.allclose(
            adjusted[ordinary],
            corrector_scaled := super(HolidayScaleCorrector, corrector).predict(base)[ordinary],
        )
        assert corrector_scaled.shape[1] == len(self.LEVELS)

    def test_the_offset_nets_out_the_baseline_bias(self):
        """The raw holiday median carries whatever bias the model has every other day.

        Applying that on holidays would double-count a bias the scaling is not addressing
        and the shift was never meant to.
        """
        n = 800
        is_holiday = np.arange(n) % 10 == 0
        # Uniformly 20 MW under-forecast, plus a further 30 MW drop on holidays.
        actual = np.full(n, 120.0) - np.where(is_holiday, 30.0, 0.0)
        corrector = HolidayScaleCorrector(self.LEVELS).fit(self._forecasts(n), actual, is_holiday)

        assert corrector.offset == pytest.approx(-30.0, abs=1.0)

    def test_too_few_holiday_hours_means_no_shift(self):
        """Early in a run almost no holidays have been seen; a fit on one afternoon is noise."""
        n = 400
        is_holiday = np.arange(n) < 10
        actual = np.full(n, 100.0) - np.where(is_holiday, 30.0, 0.0)
        corrector = HolidayScaleCorrector(self.LEVELS).fit(self._forecasts(n), actual, is_holiday)

        assert corrector.offset == 0.0

    def test_mismatched_flags_are_rejected(self):
        with pytest.raises(ValueError, match="holiday flags"):
            HolidayScaleCorrector(self.LEVELS).fit(
                self._forecasts(5), np.ones(5), np.zeros(4, dtype=bool)
            )


class TestHolidayClassScaleCorrector:
    """One offset per observance class, shrunk toward the pooled one.

    The pooled corrector improved 28 of 33 widely-observed holidays across three markets
    and 10 of 27 federal-only ones. Below a coin flip on the second group is a correction
    being applied where nothing needs correcting, which is what the split addresses.
    """

    LEVELS = (0.1, 0.5, 0.9)

    def _forecasts(self, n: int, half_width: float = 10.0) -> np.ndarray:
        return np.column_stack(
            [np.full(n, 100.0 - half_width), np.full(n, 100.0), np.full(n, 100.0 + half_width)]
        )

    def _class_data(self, n: int = 2000, major_drop: float = 30.0, minor_drop: float = 2.0):
        """Demand that falls hard on widely-observed holidays and barely on federal-only ones.

        One hour in ten is a holiday, split evenly between the classes, which is roughly the
        real ratio: six widely-observed federal holidays against five federal-only ones.
        """
        codes = np.zeros(n, dtype=np.int8)
        codes[np.arange(n) % 20 == 0] = WIDELY_OBSERVED
        codes[np.arange(n) % 20 == 10] = FEDERAL_ONLY
        drop = np.where(codes == WIDELY_OBSERVED, major_drop, 0.0) + np.where(
            codes == FEDERAL_ONLY, minor_drop, 0.0
        )
        return self._forecasts(n), np.full(n, 100.0) - drop, codes

    def test_it_learns_a_separate_offset_per_class(self):
        base, actual, codes = self._class_data(major_drop=30.0, minor_drop=2.0)
        corrector = HolidayClassScaleCorrector(self.LEVELS).fit(base, actual, codes)

        assert corrector.offsets[WIDELY_OBSERVED] < -20.0
        assert corrector.offsets[FEDERAL_ONLY] > -10.0

    def test_the_pooled_offset_splits_the_difference_and_fits_neither(self):
        """Why the split exists, stated as a measurement against its own control."""
        base, actual, codes = self._class_data(major_drop=30.0, minor_drop=0.0)
        pooled = HolidayScaleCorrector(self.LEVELS).fit(base, actual, codes > 0)
        split = HolidayClassScaleCorrector(self.LEVELS).fit(base, actual, codes)

        minor = codes == FEDERAL_ONLY
        pooled_error = np.abs(actual[minor] - pooled.predict(base, codes > 0)[minor, 1])
        split_error = np.abs(actual[minor] - split.predict(base, codes)[minor, 1])
        assert split_error.mean() < pooled_error.mean()
        # And it does not pay for that on the holidays the pooled arm already handled.
        major = codes == WIDELY_OBSERVED
        assert (
            np.abs(actual[major] - split.predict(base, codes)[major, 1]).mean()
            <= np.abs(actual[major] - pooled.predict(base, codes > 0)[major, 1]).mean() + 1.0
        )

    def test_a_single_class_reduces_to_the_pooled_corrector(self):
        """Shrinkage toward the pooled offset is a no-op when there is nothing to split.

        This is what makes the pooled arm a fair control: with one class the two arms are
        the same estimator, so any difference between them comes from the split alone.
        """
        n = 2000
        codes = np.where(np.arange(n) % 10 == 0, WIDELY_OBSERVED, ORDINARY).astype(np.int8)
        actual = np.full(n, 100.0) - np.where(codes > 0, 30.0, 0.0)
        base = self._forecasts(n)

        pooled = HolidayScaleCorrector(self.LEVELS).fit(base, actual, codes > 0)
        split = HolidayClassScaleCorrector(self.LEVELS).fit(base, actual, codes)
        assert split.offsets[WIDELY_OBSERVED] == pytest.approx(pooled.offset)
        assert np.allclose(split.predict(base, codes), pooled.predict(base, codes > 0))

    def test_a_barely_seen_class_leans_on_the_pooled_offset(self):
        """A class estimate from one afternoon is noise; shrinkage says so by weight."""
        n = 2000
        codes = np.zeros(n, dtype=np.int8)
        codes[np.arange(n) % 10 == 0] = WIDELY_OBSERVED
        codes[:4] = FEDERAL_ONLY  # four hours, against CLASS_PRIOR_HOURS of 48
        actual = np.full(n, 100.0) - np.where(codes == WIDELY_OBSERVED, 30.0, 0.0)

        corrector = HolidayClassScaleCorrector(self.LEVELS).fit(self._forecasts(n), actual, codes)
        weight = 4 / (4 + CLASS_PRIOR_HOURS)
        assert corrector.offsets[FEDERAL_ONLY] == pytest.approx(
            weight * 0.0 + (1 - weight) * corrector.offset, abs=1e-6
        )

    def test_an_unseen_class_falls_back_to_the_pooled_offset(self):
        """Juneteenth appears late in a two-year window, and 'it is a holiday' still holds."""
        n = 2000
        codes = np.where(np.arange(n) % 10 == 0, WIDELY_OBSERVED, ORDINARY).astype(np.int8)
        actual = np.full(n, 100.0) - np.where(codes > 0, 30.0, 0.0)
        corrector = HolidayClassScaleCorrector(self.LEVELS).fit(self._forecasts(n), actual, codes)

        probe = self._forecasts(2)
        adjusted = corrector.predict(probe, np.array([FEDERAL_ONLY, ORDINARY], dtype=np.int8))
        assert adjusted[0, 1] == pytest.approx(100.0 + corrector.offset)
        assert adjusted[1, 1] == pytest.approx(100.0)

    def test_too_few_holiday_hours_means_no_shift_at_all(self):
        """With no pooled offset there is nothing to split, and no class offsets are made up."""
        n = 400
        codes = np.zeros(n, dtype=np.int8)
        codes[:10] = WIDELY_OBSERVED
        actual = np.full(n, 100.0) - np.where(codes > 0, 30.0, 0.0)
        corrector = HolidayClassScaleCorrector(self.LEVELS).fit(self._forecasts(n), actual, codes)

        assert corrector.offset == 0.0
        assert corrector.offsets == {}

    def test_ordinary_hours_are_untouched(self):
        base, actual, codes = self._class_data()
        corrector = HolidayClassScaleCorrector(self.LEVELS).fit(base, actual, codes)
        adjusted = corrector.predict(base, codes)

        ordinary = codes == ORDINARY
        assert np.allclose(
            adjusted[ordinary], QuantileScaleCorrector.predict(corrector, base)[ordinary]
        )

    def test_mismatched_codes_are_rejected(self):
        with pytest.raises(ValueError, match="class codes"):
            HolidayClassScaleCorrector(self.LEVELS).fit(
                self._forecasts(5), np.ones(5), np.zeros(4, dtype=np.int8)
            )


class TestHolidayHourScaleCorrector:
    """One offset per class and hour, shrunk toward the class offset.

    Finding 19: a scalar shift over 24 hours over-corrects the small hours to reach the
    large ones, because load barely moves overnight and falls hard through the working day.
    """

    LEVELS = (0.1, 0.5, 0.9)

    def _forecasts(self, n: int, half_width: float = 10.0) -> np.ndarray:
        return np.column_stack(
            [np.full(n, 100.0 - half_width), np.full(n, 100.0), np.full(n, 100.0 + half_width)]
        )

    def _shaped_data(self, days: int = 120, night: float = 2.0, working: float = 40.0):
        """A holiday every fifth day, with a drop that is small at night and large by day.

        The shape is the whole point: an arm that learns one number per day cannot express
        it, and this is the smallest data that says so.
        """
        n = days * 24
        hours = np.tile(np.arange(24), days)
        codes = np.zeros(n, dtype=np.int8)
        is_holiday = np.repeat(np.arange(days) % 5 == 0, 24)
        codes[is_holiday] = WIDELY_OBSERVED

        working_hours = (hours >= 9) & (hours < 18)
        drop = np.where(working_hours, working, night)
        actual = np.full(n, 100.0) - np.where(codes > 0, drop, 0.0)
        return self._forecasts(n), actual, codes, hours

    def test_it_learns_a_different_shift_for_night_and_working_hours(self):
        base, actual, codes, hours = self._shaped_data()
        corrector = HolidayHourScaleCorrector(self.LEVELS).fit(base, actual, codes, hours)

        assert corrector.profile[(WIDELY_OBSERVED, 3)] > -20.0, "overnight barely moves"
        assert corrector.profile[(WIDELY_OBSERVED, 12)] < -25.0, "midday falls hard"

    def test_it_beats_the_flat_arm_on_data_with_a_shape(self):
        """The claim finding 19 makes, as a measurement against the arm it replaces."""
        base, actual, codes, hours = self._shaped_data()
        flat = HolidayClassScaleCorrector(self.LEVELS).fit(base, actual, codes)
        shaped = HolidayHourScaleCorrector(self.LEVELS).fit(base, actual, codes, hours)

        holiday = codes > 0
        flat_error = np.abs(actual[holiday] - flat.predict(base, codes)[holiday, 1])
        shaped_error = np.abs(actual[holiday] - shaped.predict(base, codes, hours)[holiday, 1])
        assert shaped_error.mean() < flat_error.mean()

    def test_the_ordinary_baseline_is_hour_matched(self):
        """The design decision, stated as the failure it prevents.

        The base model here is biased by hour on perfectly ordinary days and holidays are
        no different from ordinary days at all. An arm that subtracted a single all-hours
        ordinary median would read that diurnal bias as a holiday shape and apply it on
        holidays only. The correct answer is a profile of zeros.
        """
        days, n = 120, 120 * 24
        hours = np.tile(np.arange(24), days)
        codes = np.zeros(n, dtype=np.int8)
        codes[np.repeat(np.arange(days) % 5 == 0, 24)] = WIDELY_OBSERVED

        # A diurnal bias present on every day, holiday or not.
        actual = np.full(n, 100.0) + np.where((hours >= 9) & (hours < 18), -40.0, 2.0)
        corrector = HolidayHourScaleCorrector(self.LEVELS).fit(
            self._forecasts(n), actual, codes, hours
        )
        for hour in (3, 12, 20):
            assert corrector.profile[(WIDELY_OBSERVED, hour)] == pytest.approx(0.0, abs=1e-6)

    def test_a_shape_whose_day_average_is_zero_is_still_learned(self):
        """The strongest case for shaping, and the one its parents cannot see at all.

        Demand runs above forecast all morning and below it all afternoon by the same
        amount, so the whole-day offset is zero and the flat arms correctly learn nothing.
        There is still a large shape, and an arm that gated on the offset being non-zero
        would report no holiday effect on a holiday that visibly has one.
        """
        days, n = 200, 200 * 24
        hours = np.tile(np.arange(24), days)
        codes = np.zeros(n, dtype=np.int8)
        codes[np.repeat(np.arange(days) % 5 == 0, 24)] = WIDELY_OBSERVED

        swing = np.where(hours < 12, 30.0, -30.0)
        actual = np.full(n, 100.0) + np.where(codes > 0, swing, 0.0)
        base = self._forecasts(n)

        flat = HolidayClassScaleCorrector(self.LEVELS).fit(base, actual, codes)
        shaped = HolidayHourScaleCorrector(self.LEVELS).fit(base, actual, codes, hours)

        assert flat.offset == pytest.approx(0.0, abs=1e-6), "the day average really is zero"
        assert shaped.profile[(WIDELY_OBSERVED, 6)] > 20.0
        assert shaped.profile[(WIDELY_OBSERVED, 18)] < -20.0

        holiday = codes > 0
        flat_error = np.abs(actual[holiday] - flat.predict(base, codes)[holiday, 1])
        shaped_error = np.abs(actual[holiday] - shaped.predict(base, codes, hours)[holiday, 1])
        assert shaped_error.mean() < flat_error.mean() / 2

    def test_a_barely_seen_cell_leans_on_its_class_offset(self):
        base, actual, codes, hours = self._shaped_data()
        # Blank every holiday at hour 7 except two, leaving that cell almost unevidenced.
        keep = np.flatnonzero((codes > 0) & (hours == 7))[:2]
        thin = (codes > 0) & (hours == 7)
        thin[keep] = False
        codes = codes.copy()
        codes[thin] = ORDINARY

        corrector = HolidayHourScaleCorrector(self.LEVELS).fit(base, actual, codes, hours)
        weight = 2 / (2 + HOUR_PRIOR_HOURS)
        class_offset = corrector.offsets[WIDELY_OBSERVED]
        cell = corrector.profile[(WIDELY_OBSERVED, 7)]
        assert abs(cell - class_offset) < abs(weight * 100), "a thin cell stays near its class"

    def test_with_no_shape_it_matches_the_arm_it_extends(self):
        """What makes the class arm a fair control: with a flat holiday the two agree.

        Not exactly equal, since a per-cell median over a handful of hours is noisier than
        one over the whole class, but the arm must not invent a shape where none exists.
        """
        days, n = 200, 200 * 24
        hours = np.tile(np.arange(24), days)
        codes = np.zeros(n, dtype=np.int8)
        codes[np.repeat(np.arange(days) % 5 == 0, 24)] = WIDELY_OBSERVED
        actual = np.full(n, 100.0) - np.where(codes > 0, 30.0, 0.0)
        base = self._forecasts(n)

        flat = HolidayClassScaleCorrector(self.LEVELS).fit(base, actual, codes)
        shaped = HolidayHourScaleCorrector(self.LEVELS).fit(base, actual, codes, hours)
        assert np.allclose(shaped.predict(base, codes, hours), flat.predict(base, codes), atol=1e-6)

    def test_an_unseen_cell_falls_back_through_class_to_pooled(self):
        base, actual, codes, hours = self._shaped_data()
        corrector = HolidayHourScaleCorrector(self.LEVELS).fit(base, actual, codes, hours)

        probe = self._forecasts(2)
        adjusted = corrector.predict(
            probe,
            np.array([FEDERAL_ONLY, ORDINARY], dtype=np.int8),
            np.array([12, 12]),
        )
        assert adjusted[0, 1] == pytest.approx(100.0 + corrector.offset)
        assert adjusted[1, 1] == pytest.approx(100.0)

    def test_too_few_holiday_hours_means_no_profile_at_all(self):
        n = 400
        hours = np.tile(np.arange(24), n // 24 + 1)[:n]
        codes = np.zeros(n, dtype=np.int8)
        codes[:10] = WIDELY_OBSERVED
        actual = np.full(n, 100.0) - np.where(codes > 0, 30.0, 0.0)

        corrector = HolidayHourScaleCorrector(self.LEVELS).fit(
            self._forecasts(n), actual, codes, hours
        )
        assert corrector.offset == 0.0
        assert corrector.profile == {}

    def test_ordinary_hours_are_untouched(self):
        base, actual, codes, hours = self._shaped_data()
        corrector = HolidayHourScaleCorrector(self.LEVELS).fit(base, actual, codes, hours)
        adjusted = corrector.predict(base, codes, hours)

        ordinary = codes == ORDINARY
        assert np.allclose(
            adjusted[ordinary], QuantileScaleCorrector.predict(corrector, base)[ordinary]
        )

    def test_mismatched_hours_are_rejected(self):
        with pytest.raises(ValueError, match="local hours"):
            HolidayHourScaleCorrector(self.LEVELS).fit(
                self._forecasts(5), np.ones(5), np.zeros(5, dtype=np.int8), np.zeros(4)
            )


class TestHolidayClassFlags:
    """The split is fixed from private-sector observance, not from any result here."""

    def _day(self, date: str) -> np.ndarray:
        return np.array(
            [np.datetime64(f"{date}T00", "ns") + np.timedelta64(h, "h") for h in range(48)]
        )

    def test_christmas_is_widely_observed(self):
        codes = holiday_class_flags(self._day("2025-12-25"), "America/Chicago")
        assert (codes == WIDELY_OBSERVED).any()
        assert not (codes == FEDERAL_ONLY).any()

    def test_veterans_day_is_federal_only(self):
        codes = holiday_class_flags(self._day("2025-11-11"), "America/Chicago")
        assert (codes == FEDERAL_ONLY).any()
        assert not (codes == WIDELY_OBSERVED).any()

    def test_an_ordinary_week_carries_no_class(self):
        stamps = np.array(
            [np.datetime64("2025-03-10T00", "ns") + np.timedelta64(h, "h") for h in range(72)]
        )
        assert not holiday_class_flags(stamps, "America/Chicago").any()

    def test_the_codes_agree_with_the_plain_holiday_flag(self):
        """Two calendars that disagreed would make the arms incomparable rather than wrong."""
        stamps = np.array(
            [np.datetime64("2025-01-01T00", "ns") + np.timedelta64(h, "h") for h in range(24 * 400)]
        )
        codes = holiday_class_flags(stamps, "America/Los_Angeles")
        assert np.array_equal(codes > 0, holiday_flags(stamps, "America/Los_Angeles"))


class TestHolidayFlags:
    def test_thanksgiving_is_flagged_in_local_time(self):
        stamps = np.array(
            [np.datetime64("2024-11-28T00", "ns") + np.timedelta64(h, "h") for h in range(24)]
        )
        flags = holiday_flags(stamps, "America/Chicago")
        assert flags.any()

    def test_an_ordinary_week_is_not(self):
        stamps = np.array(
            [np.datetime64("2025-03-10T00", "ns") + np.timedelta64(h, "h") for h in range(72)]
        )
        assert not holiday_flags(stamps, "America/Chicago").any()

    def test_the_flag_follows_local_midnight_not_utc(self):
        """In UTC the boundary would fall several hours into the wrong day."""
        # 2024-11-28 07:00 UTC is Thanksgiving in Chicago (01:00) and still the 27th in
        # Los Angeles (23:00).
        stamps = np.array([np.datetime64("2024-11-28T07", "ns")])
        assert not holiday_flags(stamps, "America/Los_Angeles")[0]
        assert holiday_flags(stamps, "America/Chicago")[0]
