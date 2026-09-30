"""Unit tests for data loading and validation."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.data.loader import DataLoader
from src.forecasting.data.schema import validate_schema


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "hotel_id": "HOTEL_001",
        "revenue": rng.uniform(50000, 150000, 100),
        "room_revenue": rng.uniform(30000, 100000, 100),
        "rooms_sold": rng.integers(20, 150, 100),
        "available_rooms": 150,
        "occupancy_rate": rng.uniform(0.4, 0.95, 100),
        "adr": rng.uniform(80, 250, 100),
        "revpar": rng.uniform(50, 200, 100),
        "cost_center": None,
        "account_code": None,
        "debit_amount": None,
        "credit_amount": None,
    })


def test_load_sample():
    """Test synthetic sample data generation."""
    loader = DataLoader()
    df = loader.load_sample(n_hotels=3, n_days=365, seed=42)
    assert len(df) > 0
    assert df["hotel_id"].nunique() == 3
    assert "revenue" in df.columns
    assert "room_revenue" in df.columns
    assert (df["revenue"] >= 0).all()
    assert (df["room_revenue"] >= 0).all()


def test_validate_schema_valid(sample_df):
    """Test schema validation on valid data."""
    messages = validate_schema(sample_df, strict=False)
    errors = [m for m in messages if "ERROR" in m]
    assert len(errors) == 0


def test_validate_schema_missing_required():
    """Test schema validation with missing required column."""
    df = pd.DataFrame({"date": ["2023-01-01"], "revenue": [1000]})
    messages = validate_schema(df, strict=False)
    warnings = [m for m in messages if "WARNING" in m]
    assert any("hotel_id" in m for m in warnings)


def test_validate_schema_null_columns(sample_df):
    """Test detection of NULL columns from financial table."""
    messages = validate_schema(sample_df, strict=False)
    null_warnings = [m for m in messages if "entirely NULL" in m]
    assert len(null_warnings) > 0


def test_load_sample_deterministic():
    """Test that sample data is deterministic with same seed."""
    loader = DataLoader()
    df1 = loader.load_sample(seed=42)
    df2 = loader.load_sample(seed=42)
    pd.testing.assert_frame_equal(df1, df2)
