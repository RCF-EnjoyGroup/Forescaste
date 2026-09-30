"""Temporal train/validation/test splitter for time series data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    gap_days: int


class TemporalSplitter:
    """
    Split data chronologically into train/val/test.

    Parameters
    ----------
    train_ratio : float
        Proportion of data for training (default 0.70).
    val_ratio : float
        Proportion for validation (default 0.15).
    test_ratio : float
        Proportion for test (default 0.15).
    gap_days : int
        Gap between splits to reduce leakage (default 7).
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        gap_days: int = 7,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gap_days = gap_days

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        group_col: Optional[str] = None,
    ) -> SplitResult:
        """
        Split data chronologically.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        date_col : str
            Date column name.
        group_col : str, optional
            If provided, split per group (hotel_id).

        Returns
        -------
        SplitResult
        """
        df = df.copy()
        if date_col in df.columns:
            df = df.sort_values(date_col)
            dates = pd.to_datetime(df[date_col])
        else:
            dates = pd.to_datetime(df.index)
            df = df.sort_index()

        min_date = dates.min()
        max_date = dates.max()
        total_days = (max_date - min_date).days

        train_end = min_date + pd.Timedelta(days=int(total_days * self.train_ratio))
        val_end = train_end + pd.Timedelta(days=int(total_days * self.val_ratio))

        # Apply gap
        gap = pd.Timedelta(days=self.gap_days)

        if group_col and group_col in df.columns:
            train_parts, val_parts, test_parts = [], [], []
            for _, group in df.groupby(group_col):
                g_dates = pd.to_datetime(group[date_col]) if date_col in group.columns else pd.to_datetime(group.index)
                g_min = g_dates.min()
                g_max = g_dates.max()
                g_total = (g_max - g_min).days

                g_train_end = g_min + pd.Timedelta(days=int(g_total * self.train_ratio))
                g_val_end = g_train_end + pd.Timedelta(days=int(g_total * self.val_ratio))

                train_mask = g_dates <= g_train_end
                val_mask = (g_dates > g_train_end + gap) & (g_dates <= g_val_end)
                test_mask = g_dates > g_val_end + gap

                train_parts.append(group[train_mask])
                val_parts.append(group[val_mask])
                test_parts.append(group[test_mask])

            train = pd.concat(train_parts, ignore_index=True)
            val = pd.concat(val_parts, ignore_index=True)
            test = pd.concat(test_parts, ignore_index=True)
        else:
            if date_col in df.columns:
                dates_dt = pd.to_datetime(df[date_col])
                train = df[dates_dt <= train_end]
                val = df[(dates_dt > train_end + gap) & (dates_dt <= val_end)]
                test = df[dates_dt > val_end + gap]
            else:
                idx = pd.to_datetime(df.index)
                train = df[idx <= train_end]
                val = df[(idx > train_end + gap) & (idx <= val_end)]
                test = df[idx > val_end + gap]

        result = SplitResult(
            train=train.reset_index(drop=True),
            val=val.reset_index(drop=True),
            test=test.reset_index(drop=True),
            train_end=train_end,
            val_end=val_end,
            gap_days=self.gap_days,
        )

        logger.info(
            "Temporal split: train=%d, val=%d, test=%d (gap=%d days)",
            len(result.train), len(result.val), len(result.test), self.gap_days,
        )
        logger.info(
            "Split dates: train_end=%s, val_end=%s",
            result.train_end.date(), result.val_end.date(),
        )
        return result
