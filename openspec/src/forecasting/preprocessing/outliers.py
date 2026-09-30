"""Outlier handler for time series data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OutlierHandler:
    """
    Handle outliers via winsorization, capping, or flagging.

    Parameters
    ----------
    method : str
        Detection method: 'iqr', 'zscore'.
    threshold : float
        IQR multiplier or z-score threshold.
    action : str
        'flag', 'cap', or 'remove'.
    winsorize_limits : tuple[float, float]
        Lower and upper quantile limits for winsorization.
    """

    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        action: str = "flag",
        winsorize_limits: tuple[float, float] = (0.01, 0.99),
    ):
        self.method = method
        self.threshold = threshold
        self.action = action
        self.winsorize_limits = winsorize_limits
        self._bounds: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "OutlierHandler":
        """Learn outlier bounds from training data."""
        cols = columns or [c for c in df.select_dtypes(include=[np.number]).columns]
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if self.method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                self._bounds[col] = (q1 - self.threshold * iqr, q3 + self.threshold * iqr)
            elif self.method == "zscore":
                mean, std = s.mean(), s.std()
                self._bounds[col] = (mean - self.threshold * std, mean + self.threshold * std)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        return self

    def transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Apply outlier treatment."""
        df = df.copy()
        cols = columns or list(self._bounds.keys())

        for col in cols:
            if col not in df.columns or col not in self._bounds:
                continue

            lower, upper = self._bounds[col]
            mask = (df[col] < lower) | (df[col] > upper)
            n_outliers = mask.sum()

            if self.action == "flag":
                df[f"{col}_outlier"] = mask
            elif self.action == "cap":
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif self.action == "remove":
                df = df[~mask]
            else:
                raise ValueError(f"Unknown action: {self.action}")

            if n_outliers > 0:
                logger.info(
                    "Outliers in '%s': %d (%.2f%%) — action: %s",
                    col, n_outliers, n_outliers / len(df) * 100, self.action,
                )

        return df

    def fit_transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        return self.fit(df, columns).transform(df, columns)
