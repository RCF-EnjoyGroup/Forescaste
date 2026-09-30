"""
Preprocessing pipeline composing all steps:
missing values, temporal split, outliers, features, holidays, and scaling.

Design (leakage-safe):
    1. Missing values handled on the full chronological series (stateless).
    2. Temporal split computed first (dates only).
    3. Outlier bounds and scaler statistics learned on TRAIN only.
    4. Lag/rolling features are engineered on the full chronological series
       BEFORE splitting: every feature at time t uses only data from t-1 or
       earlier, which is legitimate (past actuals are known at prediction time).
    5. Target columns are NEVER scaled, so all metrics stay in real units ($).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.forecasting.preprocessing.features import TemporalFeatureEngineer
from src.forecasting.preprocessing.holidays import HolidayFeatureEngineer
from src.forecasting.preprocessing.missing import MissingValueHandler
from src.forecasting.preprocessing.outliers import OutlierHandler
from src.forecasting.preprocessing.scaler import FeatureScaler
from src.forecasting.preprocessing.splitter import SplitResult, TemporalSplitter

logger = logging.getLogger(__name__)

DEFAULT_TARGET_COLS = ["revenue", "room_revenue"]


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    missing_strategy : str
        Strategy for missing values.
    outlier_method : str
        Outlier detection method.
    outlier_action : str
        Action on outliers.
    feature_config : dict, optional
        Feature engineering parameters.
    split_config : dict, optional
        Temporal split parameters.
    scale_method : str or None, optional
        Scaling method (None to skip scaling — recommended for tree models).
    target_cols : list[str], optional
        Target columns excluded from scaling.
    """

    def __init__(
        self,
        missing_strategy: str = "ffill_bfill",
        outlier_method: str = "iqr",
        outlier_action: str = "flag",
        feature_config: dict | None = None,
        split_config: dict | None = None,
        scale_method: str | None = None,
        target_cols: list[str] | None = None,
    ):
        feat_cfg = feature_config or {}
        split_cfg = split_config or {}

        self.missing_handler = MissingValueHandler(strategy=missing_strategy)
        self.outlier_handler = OutlierHandler(method=outlier_method, action=outlier_action)
        self.feature_engineer = TemporalFeatureEngineer(**feat_cfg)
        self.holiday_engineer = HolidayFeatureEngineer()
        self.splitter = TemporalSplitter(**split_cfg)
        self.target_cols = target_cols if target_cols is not None else list(DEFAULT_TARGET_COLS)
        self.scaler = FeatureScaler(method=scale_method) if scale_method else None

        self._is_fitted = False

    def run(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        group_col: str | None = None,
    ) -> tuple[SplitResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full leakage-safe pipeline.

        Returns
        -------
        tuple of (SplitResult, train_df, val_df, test_df)
            Splits include engineered features; targets remain in original units.
        """
        # 1. Missing values on the full chronological series (stateless fill)
        df = self.missing_handler.fit_transform(df)

        # 2. Temporal split on dates only (defines the boundaries)
        split = self.splitter.split(df, date_col=date_col, group_col=group_col)

        # 3. Learn outlier bounds from TRAIN only
        numeric_cols = split.train.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != date_col]
        self.outlier_handler.fit(split.train, columns=numeric_cols)

        # 4. Feature engineering on the FULL series (lags use only past data),
        #    then re-split the featured frame with the same date boundaries.
        df_full = self.feature_engineer.transform(df)
        df_full = self.holiday_engineer.transform(df_full)

        _, train, val, test = self._resplit(df_full, split, date_col, group_col)

        # 5. Outlier flags (bounds from train) on every split
        train = self.outlier_handler.transform(train)
        val = self.outlier_handler.transform(val)
        test = self.outlier_handler.transform(test)

        # 6. Optional scaling — never on target columns
        if self.scaler is not None:
            excluded = set(self.target_cols) | {date_col}
            feature_cols = [
                c for c in train.select_dtypes(include=["number"]).columns
                if c not in excluded and not c.endswith("_outlier")
            ]
            self.scaler.fit(train, columns=feature_cols)
            train = self.scaler.transform(train)
            val = self.scaler.transform(val)
            test = self.scaler.transform(test)

        self._is_fitted = True
        logger.info(
            "Pipeline complete: train=%d rows/%d cols, val=%d, test=%d",
            len(train), len(train.columns), len(val), len(test),
        )
        return split, train, val, test

    def _resplit(
        self,
        df_full: pd.DataFrame,
        split: SplitResult,
        date_col: str,
        group_col: Optional[str],
    ) -> tuple[SplitResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Re-apply the same temporal boundaries to the featured dataframe."""
        featured_split = self.splitter.split(df_full, date_col=date_col, group_col=group_col)
        return featured_split, featured_split.train, featured_split.val, featured_split.test
