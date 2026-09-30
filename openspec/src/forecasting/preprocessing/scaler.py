"""Feature scaler that fits on training data only to prevent leakage."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class FeatureScaler:
    """
    Fit scaler on training data and transform all splits.

    Parameters
    ----------
    method : str
        'standard', 'minmax', or 'robust'.
    columns : list[str], optional
        Columns to scale. If None, scale all numeric columns.
    """

    def __init__(self, method: str = "standard", columns: list[str] | None = None):
        if method not in _SCALERS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(_SCALERS.keys())}")
        self.method = method
        self.columns = columns
        self._scaler = _SCALERS[method]() if method != "standard" else None
        self._fitted_columns: list[str] = []
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """Fit scaler on training data."""
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()
        self._fitted_columns = [c for c in cols if c in df.columns]
        if self._fitted_columns:
            if self.method == "standard":
                for col in self._fitted_columns:
                    mean = df[col].mean()
                    std = df[col].std(ddof=1)
                    if pd.isna(std) or std == 0:
                        std = 1.0
                    self._means[col] = float(mean)
                    self._stds[col] = float(std)
            else:
                self._scaler.fit(df[self._fitted_columns])
            logger.info("Scaler '%s' fitted on %d columns.", self.method, len(self._fitted_columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        df = df.copy()
        if not self._fitted_columns:
            return df

        available = [c for c in self._fitted_columns if c in df.columns]
        if not available:
            return df

        if self.method == "standard":
            for col in available:
                mean = self._means[col]
                std = self._stds[col]
                df[col] = (df[col] - mean) / std
            return df

        df[available] = self._scaler.transform(df[available])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled columns."""
        df = df.copy()
        available = [c for c in self._fitted_columns if c in df.columns]
        if not available:
            return df

        if self.method == "standard":
            for col in available:
                mean = self._means[col]
                std = self._stds[col]
                df[col] = (df[col] * std) + mean
            return df

        df[available] = self._scaler.inverse_transform(df[available])
        return df
