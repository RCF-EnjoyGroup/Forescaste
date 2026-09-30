"""
Future forecast generator: produce forecasts using the best model,
support multi-hotel aggregation, and generate summary reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.forecasting.models.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


@dataclass
class ForecastReport:
    """Container for future forecast report."""
    forecasts: pd.DataFrame
    summary: dict[str, Any]
    assumptions: list[str]
    limitations: list[str]
    recommendations: list[str]


class FutureForecaster:
    """
    Generate future forecasts using a trained model.

    Parameters
    ----------
    model : BaseForecaster
        Trained forecasting model.
    horizon_days : int
        Number of days to forecast.
    confidence_intervals : list[float]
        Confidence levels for prediction intervals.
    """

    def __init__(
        self,
        model: BaseForecaster,
        horizon_days: int = 90,
        confidence_intervals: list[float] | None = None,
    ):
        self.model = model
        self.horizon_days = horizon_days
        self.confidence_intervals = confidence_intervals or [0.80, 0.95]

    def generate_forecast(
        self,
        train_df: pd.DataFrame,
        target_col: str = "revenue",
        date_col: str = "date",
        hotel_col: str = "hotel_id",
        feature_cols: list[str] | None = None,
    ) -> ForecastReport:
        """
        Generate future forecast for the specified horizon.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data (for context).
        target_col : str
            Column to forecast.
        date_col : str
            Date column name.
        hotel_col : str
            Hotel identifier column.
        feature_cols : list[str], optional
            Feature columns for ML models.

        Returns
        -------
        ForecastReport
        """
        # Create future dates
        last_date = pd.to_datetime(train_df[date_col]).max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.horizon_days,
            freq="D",
        )

        # Check if multi-hotel
        has_hotels = hotel_col in train_df.columns and train_df[hotel_col].nunique() > 1

        if has_hotels:
            forecasts = self._multi_hotel_forecast(
                train_df, future_dates, target_col, date_col, hotel_col, feature_cols
            )
        else:
            forecasts = self._single_forecast(
                train_df, future_dates, target_col, date_col, feature_cols
            )

        # Generate summary
        summary = self._generate_summary(forecasts, target_col)
        assumptions = self._list_assumptions()
        limitations = self._list_limitations()
        recommendations = self._list_recommendations()

        return ForecastReport(
            forecasts=forecasts,
            summary=summary,
            assumptions=assumptions,
            limitations=limitations,
            recommendations=recommendations,
        )

    def _single_forecast(
        self, train_df, future_dates, target_col, date_col, feature_cols
    ) -> pd.DataFrame:
        """Generate forecast for a single time series."""
        # Create a DataFrame with future dates
        future_df = pd.DataFrame({date_col: future_dates})

        # For ML models, we need feature columns
        if feature_cols:
            # Forward-fill features from last known values
            last_row = train_df.iloc[-1]
            for col in feature_cols:
                if col in train_df.columns and col != date_col:
                    future_df[col] = last_row[col]

        # Generate predictions
        result = self.model.predict(future_df, target_col, date_col, feature_cols)

        forecast_df = pd.DataFrame({
            date_col: future_dates,
            target_col: result.predictions[:len(future_dates)],
        })

        if result.lower_bound is not None:
            forecast_df[f"{target_col}_lower"] = result.lower_bound[:len(future_dates)]
        if result.upper_bound is not None:
            forecast_df[f"{target_col}_upper"] = result.upper_bound[:len(future_dates)]

        forecast_df["model"] = self.model.name
        return forecast_df

    def _multi_hotel_forecast(
        self, train_df, future_dates, target_col, date_col, hotel_col, feature_cols
    ) -> pd.DataFrame:
        """Generate per-hotel forecasts and aggregated view."""
        all_forecasts = []
        hotels = train_df[hotel_col].unique()

        for hotel in hotels:
            hotel_data = train_df[train_df[hotel_col] == hotel].copy()
            if len(hotel_data) < 30:
                logger.warning("Hotel %s has only %d rows. Skipping.", hotel, len(hotel_data))
                continue

            future_df = pd.DataFrame({date_col: future_dates, hotel_col: hotel})

            if feature_cols:
                last_row = hotel_data.iloc[-1]
                for col in feature_cols:
                    if col in hotel_data.columns and col not in [date_col, hotel_col]:
                        future_df[col] = last_row[col]

            try:
                result = self.model.predict(future_df, target_col, date_col, feature_cols)
                hotel_forecast = pd.DataFrame({
                    date_col: future_dates,
                    hotel_col: hotel,
                    target_col: result.predictions[:len(future_dates)],
                })
                all_forecasts.append(hotel_forecast)
            except Exception as e:
                logger.warning("Forecast failed for hotel %s: %s", hotel, e)

        if not all_forecasts:
            return pd.DataFrame()

        forecasts = pd.concat(all_forecasts, ignore_index=True)

        # Add aggregated row
        agg = forecasts.groupby(date_col)[target_col].sum().reset_index()
        agg[hotel_col] = "AGGREGATED"
        forecasts = pd.concat([forecasts, agg], ignore_index=True)

        return forecasts

    def _generate_summary(self, forecasts: pd.DataFrame, target_col: str) -> dict[str, Any]:
        """Generate forecast summary statistics."""
        if forecasts.empty:
            return {}

        total = forecasts[target_col].sum()
        daily_mean = forecasts[target_col].mean()
        daily_std = forecasts[target_col].std()

        return {
            "total_forecasted": round(float(total), 2),
            "daily_mean": round(float(daily_mean), 2),
            "daily_std": round(float(daily_std), 2),
            "horizon_days": self.horizon_days,
            "model": self.model.name,
        }

    def _list_assumptions(self) -> list[str]:
        """List key forecasting assumptions."""
        return [
            "Historical patterns will continue into the future.",
            "No major external shocks (pandemics, natural disasters).",
            "Hotel operations remain consistent with historical data.",
            "Seasonality and trend components are stable.",
            "No structural breaks in the revenue generating process.",
        ]

    def _list_limitations(self) -> list[str]:
        """List known limitations."""
        return [
            "No external data (weather, events, competitor pricing).",
            "Forecast uncertainty increases with horizon length.",
            "Model trained on historical data may not capture unprecedented events.",
            "Confidence intervals are approximate.",
            "Single-model forecast; ensemble may improve robustness.",
        ]

    def _list_recommendations(self) -> list[str]:
        """List improvement recommendations."""
        return [
            "Incorporate weather data as exogenous regressor.",
            "Add competitor pricing and market demand indicators.",
            "Implement model ensemble (weighted average of top models).",
            "Set up automated model retraining as new data arrives.",
            "Consider hierarchical forecasting for portfolio-level views.",
            "Validate with domain experts for business reasonableness.",
        ]
