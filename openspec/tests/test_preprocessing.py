"""Unit tests for preprocessing pipeline components."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.preprocessing.features import TemporalFeatureEngineer
from src.forecasting.preprocessing.missing import MissingValueHandler
from src.forecasting.preprocessing.outliers import OutlierHandler
from src.forecasting.preprocessing.scaler import FeatureScaler
from src.forecasting.preprocessing.splitter import TemporalSplitter


@pytest.fixture
def time_series_df():
    """Create a time series DataFrame for testing."""
    dates = pd.date_range("2022-01-01", periods=730, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "hotel_id": "HOTEL_001",
        "revenue": rng.uniform(50000, 150000, 730),
        "room_revenue": rng.uniform(30000, 100000, 730),
    })


def test_missing_value_handler_ffill():
    """Test forward-fill then backward-fill imputation."""
    df = pd.DataFrame({"a": [1, np.nan, np.nan, 4, 5], "b": [1, 2, 3, 4, 5]})
    handler = MissingValueHandler(strategy="ffill_bfill")
    result = handler.fit_transform(df)
    assert result["a"].isna().sum() == 0
    assert result["a"].iloc[1] == 1  # forward filled
    assert result["a"].iloc[2] == 1  # forward filled


def test_outlier_handler_flag():
    """Test outlier flagging mode."""
    df = pd.DataFrame({"a": list(range(100)) + [10000]})  # 10000 is an outlier
    handler = OutlierHandler(method="iqr", action="flag")
    result = handler.fit_transform(df, columns=["a"])
    assert "a_outlier" in result.columns
    assert result["a_outlier"].sum() > 0


def test_outlier_handler_cap():
    """Test outlier capping mode."""
    df = pd.DataFrame({"a": list(range(100)) + [10000]})
    handler = OutlierHandler(method="iqr", action="cap")
    result = handler.fit_transform(df, columns=["a"])
    assert result["a"].max() < 10000


def test_temporal_feature_engineer(time_series_df):
    """Test temporal feature creation."""
    engineer = TemporalFeatureEngineer(
        lags=[1, 7],
        rolling_windows=[7],
        rolling_stats=["mean"],
        expanding_stats=["mean"],
        cyclical_features=["dayofweek", "month"],
        target_cols=["revenue"],
    )
    result = engineer.fit_transform(time_series_df)

    # Calendar features
    assert "year" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns

    # Cyclical features
    assert "dayofweek_sin" in result.columns
    assert "dayofweek_cos" in result.columns

    # Lag features
    assert "revenue_lag_1" in result.columns
    assert "revenue_lag_7" in result.columns

    # Rolling features
    assert "revenue_rmean_7" in result.columns

    # Expanding features
    assert "revenue_emean" in result.columns


def test_temporal_splitter(time_series_df):
    """Test temporal splitting."""
    splitter = TemporalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, gap_days=7)
    result = splitter.split(time_series_df, date_col="date")

    assert len(result.train) > 0
    assert len(result.val) > 0
    assert len(result.test) > 0

    # Verify chronological order
    train_max = pd.to_datetime(result.train["date"]).max()
    val_min = pd.to_datetime(result.val["date"]).min()
    assert train_max < val_min


def test_feature_scaler():
    """Test feature scaling."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [100, 200, 300, 400, 500]})
    scaler = FeatureScaler(method="standard")
    result = scaler.fit_transform(df)

    # Standard scaled should have mean ~0, std ~1
    assert abs(result["a"].mean()) < 0.1
    assert abs(result["a"].std() - 1.0) < 0.1


def test_no_leakage_in_features(time_series_df):
    """Verify that lag/rolling features are shifted to prevent leakage."""
    engineer = TemporalFeatureEngineer(lags=[1, 7], target_cols=["revenue"])
    result = engineer.fit_transform(time_series_df)

    # First row should have NaN for lag features (no prior data)
    assert pd.isna(result["revenue_lag_1"].iloc[0])
    assert pd.isna(result["revenue_lag_7"].iloc[0])

    # Second row lag_1 should equal first row's revenue
    assert result["revenue_lag_1"].iloc[1] == time_series_df["revenue"].iloc[0]
