"""
Recursive multi-step forecasting for feature-based (GBM) models.

This module implements the honest evaluation protocol for ML forecasters:
predictions are generated one step at a time and fed back as lag/rolling
inputs. Test-period actuals are NEVER used as model inputs, so the reported
metrics reflect real multi-step-ahead forecasting performance (the same
conditions as a true future forecast), not optimistic one-step-ahead numbers.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.forecasting.preprocessing.features import TemporalFeatureEngineer
from src.forecasting.preprocessing.holidays import HolidayFeatureEngineer

logger = logging.getLogger(__name__)


def recursive_forecast(
    model,
    history_df: pd.DataFrame,
    future_dates: Iterable,
    target_col: str = "revenue",
    date_col: str = "date",
    feature_cols: Optional[list[str]] = None,
    feature_engineer: Optional[TemporalFeatureEngineer] = None,
    holiday_engineer: Optional[HolidayFeatureEngineer] = None,
) -> np.ndarray:
    """Generate leakage-free multi-step predictions with a fitted GBM model.

    Parameters
    ----------
    model : BaseForecaster
        Fitted model (LightGBM/XGBoost/CatBoost forecaster).
    history_df : pd.DataFrame
        Observed daily series with at least [date_col, target_col]. Must end
        before the first date in `future_dates`.
    future_dates : Iterable
        Dates to predict, in order.
    target_col, date_col : str
        Column names.
    feature_cols : list[str]
        Exact feature columns used at training time.
    feature_engineer : TemporalFeatureEngineer
        Configured exactly as in training (same lags/windows/stats).
    holiday_engineer : HolidayFeatureEngineer, optional
        Holiday feature builder used in training.

    Returns
    -------
    np.ndarray
        One prediction per future date.
    """
    if feature_cols is None:
        raise ValueError("feature_cols is required for recursive forecasting.")
    if feature_engineer is None:
        raise ValueError("feature_engineer is required for recursive forecasting.")

    history = history_df[[date_col, target_col]].copy()
    history[date_col] = pd.to_datetime(history[date_col])
    history = history.sort_values(date_col).dropna(subset=[target_col]).reset_index(drop=True)

    future_dates = pd.to_datetime(list(future_dates))
    if len(future_dates) == 0:
        return np.array([])

    first_date = future_dates[0]
    if history[date_col].max() >= first_date:
        raise ValueError(
            f"History must end before the first forecast date. "
            f"History ends at {history[date_col].max()}, first forecast date {first_date}."
        )

    # Outlier flag columns produced by the pipeline are unknown for future
    # rows; assume "not an outlier" (0), matching the flag semantics.
    outlier_cols = [c for c in feature_cols if c.endswith("_outlier")]

    predictions: list[float] = []
    n_total = len(future_dates)

    for i, d in enumerate(future_dates):
        # Append a placeholder row for the date being predicted
        row = pd.DataFrame({date_col: [d], target_col: [np.nan]})
        context = pd.concat([history, row], ignore_index=True)

        featured = feature_engineer.transform(context)
        if holiday_engineer is not None:
            featured = holiday_engineer.transform(featured)

        features_row = featured.iloc[[-1]].copy()

        # Guarantee every training feature column exists
        for col in outlier_cols:
            if col not in features_row.columns:
                features_row[col] = 0
            else:
                features_row[col] = features_row[col].fillna(0)

        result = model.predict(features_row, target_col, date_col, feature_cols)
        pred = float(np.atleast_1d(result.predictions)[0])
        predictions.append(pred)

        # Feed the prediction back as if it were an observed value
        history = pd.concat(
            [history, pd.DataFrame({date_col: [d], target_col: [pred]})],
            ignore_index=True,
        )

        if (i + 1) % 50 == 0:
            logger.info("Recursive forecast: %d/%d steps", i + 1, n_total)

    logger.info("Recursive forecast complete: %d steps", n_total)
    return np.array(predictions)
