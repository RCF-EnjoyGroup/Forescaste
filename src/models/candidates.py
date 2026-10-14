"""Candidatos estadísticos y ML (spec ``forecasting``, tasks 5.3/5.4).

- ETS y SARIMAX (statsmodels) para series con estacionalidad.
- LightGBM, XGBoost y CatBoost sobre features as-of.

Todos comparten la interfaz ``fit(X/y)`` / ``predict(X)`` mínima para que el
benchmark de 7 candidatos (task 9.5) los trate de forma uniforme.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 5.3 — Estadísticos
# --------------------------------------------------------------------------


@dataclass
class ETSModel:
    """Exponential Smoothing (statsmodels) con selección automática."""

    seasonal_periods: int = 7
    damped_trend: bool = True
    trend: str = "add"
    seasonal: str | None = "add"

    def fit(self, y: pd.Series) -> "ETSModel":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = pd.Series(np.asarray(y, dtype=float)).dropna()
        seasonal = self.seasonal if len(y) >= 2 * self.seasonal_periods else None
        self._model = ExponentialSmoothing(
            y,
            trend=self.trend if len(y) > 4 else None,
            seasonal=seasonal,
            seasonal_periods=self.seasonal_periods if seasonal else None,
            damped_trend=self.damped_trend if self.trend else None,
        )
        self._fit = self._model.fit(optimized=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._fit.forecast(horizon), dtype=float)


@dataclass
class SARIMAXModel:
    """SARIMAX (statsmodels) con orden pequeño y stepwise simple."""

    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 7)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "SARIMAXModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = pd.Series(np.asarray(y, dtype=float)).dropna()
        self._fit = SARIMAX(
            y, exog=exog, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        fc = self._fit.get_forecast(steps=horizon, exog=exog)
        return np.asarray(fc.predicted_mean, dtype=float)

    def prediction_interval(self, horizon: int, alpha: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
        fc = self._fit.get_forecast(steps=horizon)
        ci = fc.conf_int(alpha=alpha)
        return np.asarray(ci.iloc[:, 0], dtype=float), np.asarray(ci.iloc[:, 1], dtype=float)


# --------------------------------------------------------------------------
# 5.4 — ML con features as-of
# --------------------------------------------------------------------------


def _split_xy(
    df: pd.DataFrame, target_col: str, feature_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"features ausentes: {missing}")
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    X = X.loc[mask]
    # codificar categóricas como string para todos los boosting
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            X[col] = X[col].astype(str)
    return X, y.loc[mask].to_numpy(dtype=float)


@dataclass
class LightGBMModel:
    params: dict = field(default_factory=lambda: {"objective": "regression", "metric": "mae", "verbose": -1})

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> "LightGBMModel":
        import lightgbm as lgb

        X, y = _split_xy(df, target_col, feature_cols)
        categorical = [c for c in X.columns if X[c].dtype == object]
        self._model = lgb.LGBMModel(**self.params)
        self._model.fit(X, y, categorical_feature=categorical or None)
        self.feature_cols_ = feature_cols
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols_].copy()
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype) == "category":
                X[col] = X[col].astype(str)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass
class XGBoostModel:
    params: dict = field(
        default_factory=lambda: {
            "objective": "reg:squarederror", "eval_metric": "mae",
            "n_estimators": 200, "learning_rate": 0.05, "n_jobs": -1,
            "verbosity": 0,
        }
    )

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> "XGBoostModel":
        from xgboost import XGBRegressor

        X, y = _split_xy(df, target_col, feature_cols)
        # one-hot para categóricas (xgboost no acepta object en versiones nuevas)
        X = pd.get_dummies(X, columns=[c for c in X.columns if X[c].dtype == object], dummy_na=False)
        self._columns = X.columns.tolist()
        self._model = XGBRegressor(**self.params)
        self._model.fit(X, y)
        self.feature_cols_ = feature_cols
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols_].copy()
        X = pd.get_dummies(X, columns=[c for c in X.columns if X[c].dtype == object], dummy_na=False)
        X = X.reindex(columns=self._columns, fill_value=0)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass
class CatBoostModel:
    params: dict = field(
        default_factory=lambda: {
            "loss_function": "MAE", "iterations": 200, "learning_rate": 0.05,
            "verbose": False, "allow_writing_files": False,
        }
    )

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> "CatBoostModel":
        from catboost import CatBoostRegressor

        X, y = _split_xy(df, target_col, feature_cols)
        cat_features = [i for i, c in enumerate(X.columns) if X[c].dtype == object]
        self._model = CatBoostRegressor(**self.params)
        self._model.fit(X, y, cat_features=cat_features, verbose=False)
        self.feature_cols_ = feature_cols
        self._cat_cols = [X.columns[i] for i in cat_features]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols_].copy()
        for col in self._cat_cols:
            X[col] = X[col].astype(str)
        return np.asarray(self._model.predict(X), dtype=float)


CANDIDATES = {
    "seasonal_naive": "src.models.baselines.SeasonalNaive",
    "pickup_curve": "src.models.baselines.PickupCurveModel",
    "ets": "src.models.candidates.ETSModel",
    "sarimax": "src.models.candidates.SARIMAXModel",
    "lightgbm": "src.models.candidates.LightGBMModel",
    "xgboost": "src.models.candidates.XGBoostModel",
    "catboost": "src.models.candidates.CatBoostModel",
}
