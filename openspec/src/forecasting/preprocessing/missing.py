"""Missing value handler for time series data."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """
    Handle missing values in time series data.

    Parameters
    ----------
    strategy : str
        'ffill_bfill', 'interpolation', or 'dropna'.
    max_gap : int
        Maximum gap length for interpolation (in periods).
    """

    def __init__(self, strategy: str = "ffill_bfill", max_gap: int = 7):
        self.strategy = strategy
        self.max_gap = max_gap
        self._imputed_counts: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        """Learn missing patterns (no-op for simple strategies)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value treatment."""
        df = df.copy()
        self._imputed_counts = {}

        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue

            if self.strategy == "ffill_bfill":
                df[col] = df[col].ffill().bfill()
            elif self.strategy == "interpolation":
                # Interpolate small gaps, ffill/bfill for edges
                df[col] = df[col].interpolate(
                    method="time", limit=self.max_gap, limit_direction="forward"
                ).bfill()
            elif self.strategy == "dropna":
                df = df.dropna(subset=[col])
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            imputed = n_missing - df[col].isna().sum()
            self._imputed_counts[col] = imputed
            if imputed > 0:
                logger.info("Imputed %d missing values in '%s' (%s)", imputed, col, self.strategy)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def imputed_counts(self) -> dict[str, int]:
        return self._imputed_counts.copy()
