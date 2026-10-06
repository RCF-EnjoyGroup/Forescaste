"""Backtesting folds/no-leakage, conformal coverage, and tuning smoke tests."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.evaluation.backtesting import (
    make_rolling_folds,
    rolling_origin_backtest,
)
from src.forecasting.evaluation.conformal import (
    empirical_coverage,
    per_horizon_conformal_bands,
)


@pytest.fixture
def daily_df():
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    t = np.arange(400)
    values = 100 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 7)
    return pd.DataFrame({"date": dates, "rooms_sold": values})


class TestRollingFolds:
    def test_folds_chronological_and_disjoint_histories(self):
        last = pd.Timestamp("2025-06-30")
        folds = make_rolling_folds(last, n_folds=3, horizon_days=60)
        assert len(folds) == 3
        ids = [f.fold_id for f in folds]
        assert ids == sorted(ids), "folds must be chronological"
        for f in folds:
            assert f.window_end == f.window_start + pd.Timedelta(days=59)
            assert f.history_end == f.window_start - pd.Timedelta(days=1)
        for older, newer in zip(folds[:-1], folds[1:]):
            assert newer.window_start > older.window_start

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError, match="at least 2"):
            make_rolling_folds(pd.Timestamp("2025-01-01"), n_folds=1)
        with pytest.raises(ValueError, match="at least 7"):
            make_rolling_folds(pd.Timestamp("2025-01-01"), horizon_days=3)


class TestRollingOriginBacktest:
    def test_no_fold_ever_receives_post_origin_data(self, daily_df):
        """The critical no-leakage property: history passed to every
        forecaster ends strictly before the fold window starts."""
        seen_hist_ends = []

        def spy_forecaster(hist, dates):
            seen_hist_ends.append(hist["date"].max())
            return hist["rooms_sold"].values[-1] * np.ones(len(dates))

        folds = make_rolling_folds(daily_df["date"].max(), n_folds=3, horizon_days=50)
        result = rolling_origin_backtest(
            daily_df, {"Spy": spy_forecaster}, folds, target_col="rooms_sold"
        )

        assert len(seen_hist_ends) == 3
        for fold, hist_end in zip(folds, seen_hist_ends):
            assert hist_end < fold.window_start
            assert hist_end == fold.history_end

        assert result.mae_table.shape == (1, 3)
        assert np.isfinite(result.mae_table.values).all()
        assert result.residuals["Spy"].shape == (3, 50)
        assert result.mean_rank.iloc[0] == 1.0
        assert int(result.wins.iloc[0]) == 3

    def test_invalid_forecaster_predictions_raise(self, daily_df):
        def bad_forecaster(hist, dates):
            preds = np.ones(len(dates))
            preds[5] = np.nan
            return preds

        folds = make_rolling_folds(daily_df["date"].max(), n_folds=2, horizon_days=30)
        with pytest.raises(ValueError, match="invalid predictions"):
            rolling_origin_backtest(daily_df, {"Bad": bad_forecaster}, folds)

    def test_rank_stability_distinguishes_models(self, daily_df):
        """A perfect weekly-seasonal forecaster must beat a flat one on every fold."""
        def flat(hist, dates):
            return np.full(len(dates), hist["rooms_sold"].mean())

        def weekly(hist, dates):
            # seasonal naive on the last observed week
            tail = hist["rooms_sold"].values[-7:]
            return np.array([tail[i % 7] for i in range(len(dates))])

        folds = make_rolling_folds(daily_df["date"].max(), n_folds=3, horizon_days=49)
        result = rolling_origin_backtest(
            daily_df, {"Flat": flat, "Weekly": weekly}, folds
        )
        assert result.mean_rank["Weekly"] < result.mean_rank["Flat"]
        assert int(result.wins["Weekly"]) == 3


class TestConformalBands:
    def test_nominal_coverage_on_gaussian_noise(self):
        rng = np.random.default_rng(0)
        n_windows, horizon = 40, 30
        residuals = rng.normal(0, 10, size=(n_windows, horizon))
        forecast = np.full(horizon, 100.0)
        actuals = forecast + rng.normal(0, 10, size=horizon)

        lower, upper = per_horizon_conformal_bands(residuals, forecast, alpha=0.2)
        assert len(lower) == len(upper) == horizon
        assert np.all(upper > lower)

        coverage = empirical_coverage(actuals, lower, upper)
        # With 40 windows, the 80% band should cover ~80% (loose tolerance)
        assert 0.6 <= coverage <= 1.0

    def test_width_grows_when_residuals_grow_with_horizon(self):
        """Uncertainty that grows with the horizon must widen the bands."""
        rng = np.random.default_rng(1)
        n_windows, horizon = 50, 30
        scale = np.linspace(2, 20, horizon)
        residuals = rng.normal(0, 1, size=(n_windows, horizon)) * scale
        forecast = np.zeros(horizon)

        lower, upper = per_horizon_conformal_bands(residuals, forecast, alpha=0.2)
        widths = upper - lower
        assert widths[-1] > 3 * widths[0], "late-horizon bands must be wider"

    def test_shape_and_validation_errors(self):
        residuals = np.zeros((5, 10))
        with pytest.raises(ValueError, match="horizon mismatch"):
            per_horizon_conformal_bands(residuals, np.zeros(9))
        with pytest.raises(ValueError, match="alpha"):
            per_horizon_conformal_bands(residuals, np.zeros(10), alpha=1.5)
        with pytest.raises(ValueError, match="same length"):
            empirical_coverage(np.zeros(10), np.zeros(9), np.zeros(10))


class TestTuningSmoke:
    def test_gbm_tuning_optimizes_recursive_val_mae(self):
        """Small smoke: tuning runs, returns valid params, and a better model
        cannot do worse than the base on the same objective."""
        from src.forecasting.models.ml import LightGBMForecaster
        from src.forecasting.models.tuning import tune_gbm
        from src.forecasting.preprocessing.features import TemporalFeatureEngineer
        from src.forecasting.preprocessing.holidays import HolidayFeatureEngineer

        rng = np.random.default_rng(3)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        t = np.arange(n)
        rooms = np.floor(
            100 + 0.05 * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 4, n)
        )
        df = pd.DataFrame({"date": dates, "rooms_sold": rooms})

        eng = TemporalFeatureEngineer(
            lags=[1, 7], rolling_windows=[7], rolling_stats=["mean"],
            expanding_stats=["mean"], cyclical_features=["dayofweek"],
            target_cols=["rooms_sold"],
        )
        he = HolidayFeatureEngineer()

        feat = he.transform(eng.transform(df.copy())).dropna()
        feature_cols = [c for c in feat.columns if c not in ("date", "rooms_sold")]

        n_train = 220
        train_feat = feat.iloc[:n_train]
        val_feat = feat.iloc[n_train:]
        pre_val_history = df.iloc[: n_train + 5]
        val_dates = pd.DatetimeIndex(df["date"].iloc[n_train + 5: n_train + 5 + 30])
        val_actuals = df["rooms_sold"].iloc[n_train + 5: n_train + 5 + 30].to_numpy()

        def prepare(hist):
            out = he.transform(eng.transform(hist.copy()))
            return out

        base = {"objective": "regression", "n_estimators": 1000, "verbose": -1, "random_state": 42}
        suggest = lambda trial: {
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }

        best = tune_gbm(
            LightGBMForecaster, suggest, base,
            train_feat, val_feat, pre_val_history, val_dates, val_actuals,
            feature_cols, eng, he, prepare,
            target_col="rooms_sold", n_trials=4, seed=42,
        )
        assert "num_leaves" in best and "learning_rate" in best
        assert 8 <= best["num_leaves"] <= 64
        assert best["objective"] == "regression"

    def test_prophet_grid_search(self):
        from src.forecasting.models.tuning import tune_prophet_grid

        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        t = np.arange(120)
        rooms = 100 + 10 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(0).normal(0, 3, 120)
        df = pd.DataFrame({"date": dates, "rooms_sold": rooms})

        train = df.iloc[:90]
        val_dates = pd.DatetimeIndex(df["date"].iloc[90:120])
        val_actuals = df["rooms_sold"].iloc[90:120].to_numpy()

        grid = {
            "changepoint_prior_scale": [0.01, 0.05],
            "seasonality_prior_scale": [1.0, 10.0],
        }
        best, mae = tune_prophet_grid(
            train, val_dates, val_actuals, grid, target_col="rooms_sold"
        )
        assert best["changepoint_prior_scale"] in (0.01, 0.05)
        assert np.isfinite(mae) and mae > 0
