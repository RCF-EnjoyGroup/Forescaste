"""Room-nights change: NULL handling, synthetic generator, capacity ceiling,
and forward-known column passthrough in recursive forecasting."""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.forecasting.data import (
    DataLoader,
    aggregate_daily_target,
    apply_capacity_ceiling,
    implied_occupancy,
)
from src.forecasting.data.schema import validate_schema


@pytest.fixture
def raw_with_financial_rows():
    """Operational rows plus financial rows with NULL rooms_sold."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    for d in dates:
        rows.append({"date": d, "hotel_id": "H1", "rooms_sold": 80, "available_rooms": 100,
                     "revenue": 8000.0, "room_revenue": 6000.0})
        # Financial accounting row: NULL operational columns
        rows.append({"date": d, "hotel_id": "H1", "rooms_sold": None, "available_rooms": None,
                     "revenue": 500.0, "room_revenue": 300.0})
    return pd.DataFrame(rows)


class TestNullTargetHandling:
    def test_financial_rows_excluded_before_imputation(self, raw_with_financial_rows):
        daily, report = aggregate_daily_target(
            raw_with_financial_rows, target_col="rooms_sold", sum_cols=["available_rooms"]
        )
        assert len(daily) == 30
        # Every day must aggregate ONLY the operational row (never NULLs/zeros)
        assert (daily["rooms_sold"] == 80).all()
        assert (daily["available_rooms"] == 100).all()
        assert report["n_excluded_null_rows"] == 30
        assert report["n_gap_dates"] == 0

    def test_gap_dates_reindexed_for_contiguity(self, raw_with_financial_rows):
        df = raw_with_financial_rows
        # Drop one operational day -> a calendar gap
        mask = (df["rooms_sold"].isna()) | ~(df["date"] == pd.Timestamp("2024-01-10"))
        df = df[~((df["date"] == pd.Timestamp("2024-01-10")) & df["rooms_sold"].notna())]
        daily, report = aggregate_daily_target(df, target_col="rooms_sold")
        assert len(daily) == 30  # contiguous calendar preserved
        assert daily.loc[daily["date"] == pd.Timestamp("2024-01-10"), "rooms_sold"].isna().iloc[0]
        assert report["n_gap_dates"] == 1
        assert "2024-01-10" in report["gap_dates"]

    def test_missing_target_column_raises(self):
        with pytest.raises(ValueError, match="not present"):
            aggregate_daily_target(pd.DataFrame({"date": [1]}), target_col="rooms_sold")


class TestSchemaValidation:
    def test_missing_target_column_raises_on_load(self, tmp_path):
        csv = tmp_path / "no_target.csv"
        pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "hotel_id": "H1",
            "revenue": 1000.0,
            "room_revenue": 800.0,
        }).to_csv(csv, index=False)
        loader = DataLoader(csv_path=csv)
        with pytest.raises(ValueError, match="rooms_sold"):
            loader.load(target_col="rooms_sold")

    def test_present_target_column_passes(self, raw_with_financial_rows):
        messages = validate_schema(raw_with_financial_rows, required_columns=["rooms_sold"])
        assert not any("rooms_sold" in m and "ERROR" in m for m in messages)


class TestSyntheticGenerator:
    def test_determinism_by_seed(self, tmp_path):
        loader = DataLoader()
        df1 = loader.generate_roomnights_sample(n_hotels=3, n_days=120, seed=7)
        df2 = loader.generate_roomnights_sample(n_hotels=3, n_days=120, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_capacity_bounds_and_integer_room_nights(self):
        loader = DataLoader()
        df = loader.generate_roomnights_sample(n_hotels=3, n_days=180, seed=42)
        assert (df["rooms_sold"].dropna() >= 0).all()
        assert (df["rooms_sold"].dropna() <= df["available_rooms"].dropna()).all()
        assert (df["rooms_sold"].dropna() % 1 == 0).all()

    def test_exact_physical_identities(self):
        loader = DataLoader()
        df = loader.generate_roomnights_sample(n_hotels=3, n_days=365, seed=42)
        ops = df[df["rooms_sold"].notna()]

        np.testing.assert_allclose(
            ops["occupancy_rate"], ops["rooms_sold"] / ops["available_rooms"], rtol=1e-9
        )
        np.testing.assert_allclose(
            ops["room_revenue"], ops["rooms_sold"] * ops["adr"], rtol=1e-9
        )
        np.testing.assert_allclose(
            ops["revpar"], ops["room_revenue"] / ops["available_rooms"], rtol=1e-9
        )
        assert (ops["revenue"] > ops["room_revenue"]).all()  # ancillary share > 0

    def test_realistic_correlations(self):
        loader = DataLoader()
        df = loader.generate_roomnights_sample(n_hotels=4, n_days=365 * 2, seed=42)
        ops = df[df["rooms_sold"].notna()]

        # The occupancy identity is exact WITHIN each hotel. Pearson
        # correlation is ~1.0 except in the renovation hotel, where the two
        # capacity regimes (full vs. halved) split the linear relation into
        # two segments (~0.84) while the identity still holds pointwise —
        # that pointwise exactness is verified in test_exact_physical_identities.
        per_hotel_occ = {
            hid: g["rooms_sold"].corr(g["occupancy_rate"]) for hid, g in ops.groupby("hotel_id")
        }
        assert min(per_hotel_occ.values()) > 0.8, (
            f"occupancy must track the target per hotel (renovation regime break expected): "
            f"{per_hotel_occ}"
        )

        # Demand and price co-move over time WITHIN each hotel (shared
        # seasonality/premiums). Pooled cross-hotel correlation is dominated by
        # hotel size vs. random ADR positioning (Simpson's paradox) and is not
        # asserted.
        per_hotel_adr = {
            hid: g["rooms_sold"].corr(g["adr"]) for hid, g in ops.groupby("hotel_id")
        }
        assert min(per_hotel_adr.values()) > 0.1, (
            f"ADR must co-move positively with demand within every hotel: {per_hotel_adr}"
        )

    def test_financial_rows_present_with_null_target(self):
        loader = DataLoader()
        df = loader.generate_roomnights_sample(n_hotels=2, n_days=120, seed=42)
        null_rows = df["rooms_sold"].isna().sum()
        assert null_rows > 0, "generator must exercise the NULL handling path"
        fin = df[df["rooms_sold"].isna()]
        assert fin["available_rooms"].isna().all()
        assert fin["cost_center"].notna().all()

    def test_renovation_dip_changes_capacity(self):
        loader = DataLoader()
        df = loader.generate_roomnights_sample(n_hotels=3, n_days=365, seed=42)
        hotel2 = df[(df["hotel_id"] == "HOTEL_002") & df["rooms_sold"].notna()]
        caps = hotel2.groupby("date")["available_rooms"].first()
        assert caps.nunique() > 1, "simulated renovation must move the capacity ceiling"


class TestCapacityCeiling:
    def test_forecast_capped_and_reported(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        forecast = pd.Series([80, 95, 120, 60, 110], index=dates, name="rooms_sold")
        capacity = pd.Series([100, 100, 100, 100, 100])

        capped, report = apply_capacity_ceiling(forecast, capacity)

        assert capped.max() <= 100
        np.testing.assert_array_equal(capped.values, [80, 95, 100, 60, 100])
        assert len(report) == 2
        assert set(report["date"]) == {dates[2], dates[4]}
        np.testing.assert_array_equal(report["original_forecast"], [120, 110])

    def test_implied_occupancy(self):
        forecast = pd.Series([90, 50])
        capacity = pd.Series([100, 100])
        occ = implied_occupancy(forecast, capacity)
        np.testing.assert_allclose(occ.values, [0.9, 0.5])


class TestForwardKnownPassthrough:
    def test_available_rooms_passthrough_matches_reference(self):
        from src.forecasting.models.ml import LightGBMForecaster
        from src.forecasting.preprocessing.features import TemporalFeatureEngineer
        from src.forecasting.preprocessing.holidays import HolidayFeatureEngineer
        from src.forecasting.forecasting.recursive import recursive_forecast

        rng = np.random.default_rng(5)
        n = 240
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        capacity = np.where(np.arange(n) < 200, 100.0, 80.0)  # capacity change
        occupancy = 0.6 + 0.1 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.03, n)
        rooms_sold = np.floor(capacity * np.clip(occupancy, 0.1, 0.95)).astype(float)

        df = pd.DataFrame({"date": dates, "rooms_sold": rooms_sold, "available_rooms": capacity})

        eng = TemporalFeatureEngineer(
            lags=[1, 7], rolling_windows=[7], rolling_stats=["mean"],
            expanding_stats=["mean"], cyclical_features=["dayofweek"],
            target_cols=["rooms_sold"],
        )
        he = HolidayFeatureEngineer()
        feat = he.transform(eng.transform(df.copy())).dropna()
        feature_cols = [
            c for c in feat.columns
            if c not in ("date", "rooms_sold") and not c.startswith("room_revenue")
        ]
        assert "available_rooms" in feature_cols

        model = LightGBMForecaster(
            params={"objective": "regression", "n_estimators": 40, "random_state": 0, "verbose": -1}
        )
        model.fit(feat, target_col="rooms_sold", date_col="date", feature_cols=feature_cols)

        history = df.iloc[:200]
        future = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=25)

        fast = recursive_forecast(
            model, history, future, target_col="rooms_sold", date_col="date",
            feature_cols=feature_cols, feature_engineer=eng, holiday_engineer=he,
            use_incremental=True, future_known_cols=["available_rooms"],
        )
        ref = recursive_forecast(
            model, history, future, target_col="rooms_sold", date_col="date",
            feature_cols=feature_cols, feature_engineer=eng, holiday_engineer=he,
            use_incremental=False, future_known_cols=["available_rooms"],
        )
        assert len(fast) == len(ref) == 25
        assert not np.isnan(fast).any()
        assert np.mean(np.abs(fast - ref)) / np.mean(np.abs(ref)) < 0.01


class TestConfigRoomNights:
    def test_config_keys_and_revenue_untouched(self):
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rn = cfg["roomnights"]
        assert rn["target_col"] == "rooms_sold"
        assert rn["available_rooms_col"] == "available_rooms"
        assert rn["features"]["target_cols"] == ["rooms_sold"]
        assert "revenue" in rn["excluded_feature_prefixes"]
        # Shared preprocessing keys unchanged for the revenue pipeline
        assert cfg["preprocessing"]["features"]["lags"] == [1, 7, 14, 30, 365]
        assert cfg["preprocessing"]["split"]["gap_days"] == 7
        assert "csv_path" in cfg["data"]
