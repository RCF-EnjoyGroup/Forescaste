"""
Temporal feature engineering: calendar, cyclical, lag, rolling, expanding features.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Create temporal features for time series forecasting.

    All lag/rolling/expanding features are shifted by 1 to prevent data leakage.

    Parameters
    ----------
    lags : list[int]
        Lag periods to create.
    rolling_windows : list[int]
        Window sizes for rolling statistics.
    rolling_stats : list[str]
        Stats to compute: 'mean', 'std', 'min', 'max'.
    expanding_stats : list[str]
        Stats for expanding window: 'mean', 'std'.
    cyclical_features : list[str]
        Features to encode cyclically: 'dayofweek', 'dayofyear', 'month'.
    target_cols : list[str]
        Columns to create lag/rolling features for.
    """

    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        rolling_stats: list[str] | None = None,
        expanding_stats: list[str] | None = None,
        cyclical_features: list[str] | None = None,
        target_cols: list[str] | None = None,
    ):
        self.lags = lags or [1, 7, 14, 30, 365]
        self.rolling_windows = rolling_windows or [7, 14, 30, 90]
        self.rolling_stats = rolling_stats or ["mean", "std", "min", "max"]
        self.expanding_stats = expanding_stats or ["mean", "std"]
        self.cyclical_features = cyclical_features or ["dayofweek", "dayofyear", "month"]
        self.target_cols = target_cols or ["revenue", "room_revenue"]

    def fit(self, df: pd.DataFrame) -> "TemporalFeatureEngineer":
        """No-op for feature engineering (stateless)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all temporal features."""
        df = df.copy()
        date_col = "date" if "date" in df.columns else df.index.name

        # If date is in columns, set as index for feature extraction
        if "date" in df.columns:
            df = df.set_index("date")

        df = self._add_calendar_features(df)
        df = self._add_cyclical_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_expanding_features(df)

        # Reset index if we set it
        if date_col == "date":
            df = df.reset_index()

        logger.info("Feature engineering complete: %d columns", len(df.columns))
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        df["year"] = idx.year
        df["quarter"] = idx.quarter
        df["month"] = idx.month
        df["week"] = idx.isocalendar().week.astype(int)
        df["day"] = idx.day
        df["dayofweek"] = idx.dayofweek
        df["dayofyear"] = idx.dayofyear
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["is_month_start"] = idx.is_month_start.astype(int)
        df["is_month_end"] = idx.is_month_end.astype(int)
        df["is_quarter_start"] = idx.is_quarter_start.astype(int)
        df["is_quarter_end"] = idx.is_quarter_end.astype(int)
        return df

    # ------------------------------------------------------------------
    # Cyclical features
    # ------------------------------------------------------------------

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        period_map = {
            "dayofweek": 7,
            "dayofyear": 365,
            "month": 12,
            "week": 52,
            "quarter": 4,
            "day": 31,
        }
        for feat in self.cyclical_features:
            if feat not in df.columns:
                continue
            period = period_map.get(feat, df[feat].max() + 1)
            df[f"{feat}_sin"] = np.sin(2 * np.pi * df[feat] / period)
            df[f"{feat}_cos"] = np.cos(2 * np.pi * df[feat] / period)
        return df

    # ------------------------------------------------------------------
    # Lag features (shifted by 1 to prevent leakage)
    # ------------------------------------------------------------------

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.target_cols:
            if col not in df.columns:
                continue
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    # ------------------------------------------------------------------
    # Rolling features (shifted by 1 to prevent leakage)
    # ------------------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.target_cols:
            if col not in df.columns:
                continue
            for window in self.rolling_windows:
                rolled = df[col].shift(1).rolling(window=window, min_periods=1)
                for stat in self.rolling_stats:
                    if stat == "mean":
                        df[f"{col}_rmean_{window}"] = rolled.mean()
                    elif stat == "std":
                        df[f"{col}_rstd_{window}"] = rolled.std()
                    elif stat == "min":
                        df[f"{col}_rmin_{window}"] = rolled.min()
                    elif stat == "max":
                        df[f"{col}_rmax_{window}"] = rolled.max()
        return df

    # ------------------------------------------------------------------
    # Expanding features (shifted by 1 to prevent leakage)
    # ------------------------------------------------------------------

    def _add_expanding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.target_cols:
            if col not in df.columns:
                continue
            expanded = df[col].shift(1).expanding(min_periods=1)
            for stat in self.expanding_stats:
                if stat == "mean":
                    df[f"{col}_emean"] = expanded.mean()
                elif stat == "std":
                    df[f"{col}_estd"] = expanded.std()
        return df
