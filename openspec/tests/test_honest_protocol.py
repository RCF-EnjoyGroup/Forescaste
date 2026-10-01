"""Tests for the honest evaluation protocol: date-aligned prediction and
leakage-free recursive forecasting."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.models.statistical.ets_model import ETSForecaster
from src.forecasting.models.statistical.sarimax_model import SARIMAXForecaster
from src.forecasting.forecasting.recursive import recursive_forecast
from src.forecasting.preprocessing.features import TemporalFeatureEngineer


@pytest.fixture
def daily_series_df():
    """Synthetic daily series with trend + weekly seasonality (150 days)."""
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    t = np.arange(150)
    values = 1000 + 2 * t + 50 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(42).normal(0, 10, 150)
    return pd.DataFrame({"date": dates, "revenue": values})


class TestSARIMAXDateAlignment:
    def test_predictions_align_with_requested_dates(self, daily_series_df):
        model = SARIMAXForecaster(seasonal_periods=[7], stepwise=True, max_p=2, max_q=2)
        model.fit(daily_series_df, target_col="revenue")

        train_end = daily_series_df["date"].max()

        # Window 1: immediately after training
        w1 = pd.DataFrame({"date": pd.date_range(train_end + pd.Timedelta(days=1), periods=5)})
        # Window 2: 5 days later (simulates a gap between fit and evaluation)
        w2 = pd.DataFrame({"date": pd.date_range(train_end + pd.Timedelta(days=6), periods=5)})

        r1 = model.predict(w1, target_col="revenue")
        r2 = model.predict(w2, target_col="revenue")

        # Reference: full 10-step forecast from the pmdarima model
        full = np.asarray(model._model.predict(n_periods=10))

        assert np.allclose(r1.predictions, full[:5])
        assert np.allclose(r2.predictions, full[5:10])

    def test_raises_for_dates_before_train_end(self, daily_series_df):
        model = SARIMAXForecaster(seasonal_periods=[7], stepwise=True, max_p=2, max_q=2)
        model.fit(daily_series_df, target_col="revenue")
        bad = pd.DataFrame({"date": [daily_series_df["date"].iloc[10]]})
        with pytest.raises(ValueError, match="after the last training date"):
            model.predict(bad, target_col="revenue")


class TestETSDateAlignment:
    def test_predictions_align_with_requested_dates(self, daily_series_df):
        model = ETSForecaster(seasonal_periods=7, auto_select=True)
        model.fit(daily_series_df, target_col="revenue")

        train_end = daily_series_df["date"].max()
        full = np.asarray(model._model.forecast(10))

        w1 = pd.DataFrame({"date": pd.date_range(train_end + pd.Timedelta(days=1), periods=5)})
        w2 = pd.DataFrame({"date": pd.date_range(train_end + pd.Timedelta(days=6), periods=5)})

        r1 = model.predict(w1, target_col="revenue")
        r2 = model.predict(w2, target_col="revenue")

        assert np.allclose(r1.predictions, full[:5])
        assert np.allclose(r2.predictions, full[5:10])


class _Lag1IdentityModel:
    """Stub model: prediction = revenue_lag_1 (records the features it receives)."""

    def __init__(self):
        self.name = "Lag1Identity"
        self.seen_lag1 = []

    def predict(self, df, target_col="revenue", date_col="date", feature_cols=None):
        lag1 = df["revenue_lag_1"].values
        self.seen_lag1.extend(lag1.tolist())
        from src.forecasting.models.base import ForecastResult

        return ForecastResult(predictions=lag1, actuals=None, model_name=self.name)


class TestRecursiveForecast:
    def test_recursive_uses_only_history(self, daily_series_df):
        """The recursion must feed predictions (not future actuals) back as lags."""
        engineer = TemporalFeatureEngineer(
            lags=[1, 7], rolling_windows=[7], rolling_stats=["mean"], expanding_stats=[],
            cyclical_features=["dayofweek"], target_cols=["revenue"],
        )
        model = _Lag1IdentityModel()

        history = daily_series_df.iloc[:100]
        future_dates = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=5)

        preds = recursive_forecast(
            model, history, future_dates,
            target_col="revenue", feature_cols=["revenue_lag_1", "revenue_lag_7"],
            feature_engineer=engineer,
        )

        last_value = history["revenue"].iloc[-1]
        # Identity model: every step must return the previous prediction
        assert len(preds) == 5
        assert np.allclose(preds, last_value)

    def test_recursive_rejects_overlapping_history(self, daily_series_df):
        engineer = TemporalFeatureEngineer(lags=[1], rolling_windows=[], rolling_stats=[],
                                           expanding_stats=[], cyclical_features=[],
                                           target_cols=["revenue"])
        model = _Lag1IdentityModel()
        history = daily_series_df.iloc[:100]
        overlapping = pd.date_range(history["date"].max() - pd.Timedelta(days=2), periods=3)

        with pytest.raises(ValueError, match="before the first forecast date"):
            recursive_forecast(
                model, history, overlapping,
                target_col="revenue", feature_cols=["revenue_lag_1"],
                feature_engineer=engineer,
            )


class TestPatchTSTFallbackAlignment:
    def test_seasonal_naive_fallback_aligned(self, daily_series_df):
        from src.forecasting.models.dl.patchtst_model import PatchTSTForecaster

        model = PatchTSTForecaster(context_len=256, horizon=30)
        # fit with short data -> training sequences empty -> fallback mode
        model.fit(daily_series_df, target_col="revenue")

        train_end = daily_series_df["date"].max()
        w = pd.DataFrame({"date": pd.date_range(train_end + pd.Timedelta(days=1), periods=10)})

        result = model.predict(w, target_col="revenue")

        tail = daily_series_df["revenue"].values[-7:]
        expected = np.array([tail[i % 7] for i in range(10)])
        assert np.allclose(result.predictions, expected)


class TestIncrementalFastPath:
    """The incremental path must be feature-identical to the reference path."""

    def _make_engineers(self, cyclical=("dayofweek", "dayofyear", "month")):
        from src.forecasting.preprocessing.holidays import HolidayFeatureEngineer

        eng = TemporalFeatureEngineer(
            lags=[1, 7, 30], rolling_windows=[7, 30],
            rolling_stats=["mean", "std", "min", "max"],
            expanding_stats=["mean", "std"],
            cyclical_features=list(cyclical), target_cols=["revenue"],
        )
        he = HolidayFeatureEngineer()
        return eng, he

    def _make_history(self, n=120):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        t = np.arange(n)
        revenue = np.linspace(1000, 3000, n) + 200 * np.sin(2 * np.pi * t / 7)
        return pd.DataFrame({"date": dates, "revenue": revenue})

    def test_features_match_reference_step_by_step(self):
        from src.forecasting.forecasting.recursive import _IncrementalRowBuilder, _values_match

        eng, he = self._make_engineers()
        history = self._make_history()

        # Fit the holiday calendar FIRST (recursive_forecast does this via the
        # probe transform before constructing the builder).
        he.transform(history)

        builder = _IncrementalRowBuilder(eng, he, "revenue", history["revenue"].to_numpy())

        hist = history.copy()
        rng = np.random.default_rng(3)
        for step in range(12):
            d = hist["date"].max() + pd.Timedelta(days=1)
            ctx = pd.concat([hist, pd.DataFrame({"date": [d], "revenue": [np.nan]})],
                            ignore_index=True)
            ref = he.transform(eng.transform(ctx))
            columns = [c for c in ref.columns if c not in ("date", "revenue")]
            ref_row = ref.iloc[-1]

            fast = builder.build_row(d, columns)
            for col in columns:
                assert _values_match(fast[col], ref_row[col]), (
                    f"step {step}, col {col!r}: incremental={fast[col]!r} "
                    f"reference={ref_row[col]!r}"
                )

            p = float(rng.uniform(1500, 2500))
            builder.append(p)
            hist = pd.concat([hist, pd.DataFrame({"date": [d], "revenue": [p]})],
                             ignore_index=True)

    def test_predictions_agree_with_reference_path(self):
        from src.forecasting.models.ml import LightGBMForecaster

        eng, he = self._make_engineers()
        history = self._make_history(200)
        future = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=30)

        feat = he.transform(eng.transform(history.copy()))
        feat = feat.dropna()
        feature_cols = [c for c in feat.columns if c not in ("date", "revenue")]

        model = LightGBMForecaster(
            params={"objective": "regression", "n_estimators": 60, "random_state": 0, "verbose": -1}
        )
        model.fit(feat, target_col="revenue", date_col="date", feature_cols=feature_cols)

        fast = recursive_forecast(
            model, history, future, target_col="revenue", date_col="date",
            feature_cols=feature_cols, feature_engineer=eng, holiday_engineer=he,
            use_incremental=True,
        )
        ref = recursive_forecast(
            model, history, future, target_col="revenue", date_col="date",
            feature_cols=feature_cols, feature_engineer=eng, holiday_engineer=he,
            use_incremental=False,
        )
        assert len(fast) == len(ref) == 30
        # Features are identical within float rounding (verified feature-by-feature
        # in the step test). With tree models, a ~1e-12 feature difference can still
        # flip a split threshold and jump to another leaf, so predictions are
        # compared with a leaf-jump tolerant bound rather than exact equality.
        assert np.mean(np.abs(fast - ref)) / np.mean(np.abs(ref)) < 0.01

    def test_unsupported_config_falls_back_silently(self):
        """A non-replicable cyclical feature must degrade to the reference path."""
        from src.forecasting.models.ml import LightGBMForecaster

        # 'is_weekend' is not in the cyclical period map -> incremental raises
        # -> verification fails -> reference path used -> results still correct.
        eng, he = self._make_engineers(cyclical=("dayofweek", "is_weekend"))
        history = self._make_history(150)
        future = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=15)

        feat = he.transform(eng.transform(history.copy())).dropna()
        feature_cols = [c for c in feat.columns if c not in ("date", "revenue")]

        model = LightGBMForecaster(
            params={"objective": "regression", "n_estimators": 30, "random_state": 0, "verbose": -1}
        )
        model.fit(feat, target_col="revenue", date_col="date", feature_cols=feature_cols)

        result = recursive_forecast(
            model, history, future, target_col="revenue", date_col="date",
            feature_cols=feature_cols, feature_engineer=eng, holiday_engineer=he,
            use_incremental=True,
        )
        assert len(result) == 15
        assert not np.isnan(result).any()
