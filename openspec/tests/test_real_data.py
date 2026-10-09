"""Real-schema change tests: property mapping, locale parsing, as-of pickup,
capacity fallback, cold-start flagging, and per-date future covariates."""

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from src.forecasting.data.property_map import (
    PropertyMapError,
    apply_property_map,
    canonical_property,
)
from src.forecasting.data.real_sources import (
    parse_locale_float,
    load_real_aggregates,
)
from src.forecasting.data.pickup import (
    build_pickup_features,
    multi_step_pickup_columns,
    portfolio_pickup,
)
from src.forecasting.data.synthetic_real import generate_real_schema_sample

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def rn_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["roomnights_real"]


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("real_sample")
    generate_real_schema_sample(out, n_days=365 * 2, seed=7, future_horizon_days=120)
    return out


@pytest.fixture(scope="module")
def aggregates(sample_dir, rn_config):
    log = {}
    aggs = load_real_aggregates(
        sample_dir / "demand_daily.csv",
        sample_dir / "pickup_by_lead.csv",
        sample_dir / "capacity_daily.csv",
        mapping_config=rn_config,
        exclusions_log=log,
    )
    aggs["_log"] = log
    return aggs


class TestPropertyMap:
    def test_all_observed_ids_resolve(self, rn_config):
        for variant, expected in rn_config["property_map"].items():
            assert canonical_property(variant, rn_config["property_map"]) == expected

    def test_unknown_identifier_raises_loudly(self, rn_config):
        with pytest.raises(PropertyMapError, match="Unknown property"):
            canonical_property("HOTEL_999", rn_config["property_map"])

    def test_apply_map_dataframe(self, rn_config):
        df = pd.DataFrame({"property": ["Hotel Royal Corin", "LAPAS", "CORIN"]})
        out = apply_property_map(df, "property", mapping_config=rn_config)
        assert out["property"].tolist() == ["corin", "lapas", "corin"]

    def test_apply_map_reports_unknowns(self, rn_config):
        df = pd.DataFrame({"property": ["LAPAS", "MISTERY"]})
        with pytest.raises(PropertyMapError, match="MISTERY"):
            apply_property_map(df, "property", mapping_config=rn_config)


class TestLocaleParsing:
    def test_comma_decimals(self):
        parsed, n_bad = parse_locale_float(pd.Series(["329,6", "657,74", "144,07"]))
        np.testing.assert_allclose(parsed.values, [329.6, 657.74, 144.07])
        assert n_bad == 0

    def test_mixed_valid_invalid(self):
        parsed, n_bad = parse_locale_float(pd.Series(["100,5", "abc", None, 42.0]))
        assert parsed.isna().sum() == 2  # 'abc' + None
        assert parsed.iloc[3] == 42.0
        assert n_bad == 1


class TestRealAggregates:
    def test_demand_shape_and_uniqueness(self, aggregates):
        demand = aggregates["demand"]
        assert not demand.duplicated(subset=["property", "stay_date"]).any()
        assert (demand["rooms_sold"] >= 0).all()
        assert demand["room_revenue"].dtype == float
        # locale strings were parsed: revenue == rooms * adr-ish, just positive check
        assert (demand["room_revenue"] > 0).all()

    def test_pickup_long_format(self, aggregates):
        pickup = aggregates["pickup"]
        assert set(["property", "stay_date", "lead_days", "rooms_booked"]) <= set(pickup.columns)
        assert (pickup["lead_days"] > 0).all()

    def test_capacity_fallback_filled(self, aggregates):
        capacity = aggregates["capacity"]
        # the generator creates a 30-day sparse stretch for lapas
        assert aggregates["_log"]["capacity_fallback_days"] >= 25
        lapas = capacity[capacity["property"] == "lapas"]
        assert lapas["available_rooms"].notna().all()

    def test_cold_start_property_detected(self, aggregates):
        demand = aggregates["demand"]
        counts = demand.groupby("property")["stay_date"].nunique()
        assert counts["sjo"] <= 60  # cold-start: minimal realized history
        assert counts["corin"] > 700  # normal property: full history


class TestPickupFeatures:
    def test_exact_and_nearest_earlier_lead(self):
        pickup = pd.DataFrame({
            "property": ["a", "a", "a"],
            "stay_date": [pd.Timestamp("2026-01-10")] * 3,
            "lead_days": [60, 90, 180],
            "rooms_booked": [55, 45, 28],
        })
        feats = build_pickup_features(pickup, [90, 180])
        row = feats.iloc[0]
        assert row["rotb_d90"] == 45   # exact lead present
        assert row["rotb_d180"] == 28  # exact lead present

        # If d90 were missing, d180 (an EARLIER snapshot in time) must be used
        pickup2 = pickup[pickup["lead_days"] != 90]
        feats2 = build_pickup_features(pickup2, [90, 180])
        assert feats2.iloc[0]["rotb_d90"] == 28

    def test_as_of_no_lookahead(self):
        """A snapshot taken AFTER the (stay - lead) moment must never be used.

        Books at lead 30 (snapshot 30 days before stay) must not leak into
        rotb_d90 (the state of the books 90 days out).
        """
        pickup = pd.DataFrame({
            "property": ["a", "a"],
            "stay_date": [pd.Timestamp("2026-01-10")] * 2,
            "lead_days": [30, 90],
            "rooms_booked": [99, 40],  # 99 is a later, larger book
        })
        feats = build_pickup_features(pickup, [90])
        assert feats.iloc[0]["rotb_d90"] == 40  # NOT 99: no look-ahead

    def test_multi_step_column_selection(self):
        cols = multi_step_pickup_columns([7, 14, 30, 60, 90, 180, 365], horizon_days=90)
        assert cols == ["rotb_d90", "rotb_d180", "rotb_d365"]

    def test_books_bounded_by_eventual_demand_on_average(self, aggregates):
        """Books closer to the stay should be >= books further out (pickup)."""
        pickup = aggregates["pickup"]
        by_lead = pickup.groupby("lead_days")["rooms_booked"].mean()
        assert by_lead[30] > by_lead[180] > by_lead[365]

    def test_portfolio_pickup_sums_properties(self):
        pickup = pd.DataFrame({
            "property": ["a", "b"],
            "stay_date": [pd.Timestamp("2026-01-10")] * 2,
            "lead_days": [90, 90],
            "rooms_booked": [40, 60],
        })
        feats = portfolio_pickup(pickup, [90])
        assert feats.iloc[0]["rotb_d90"] == 100


class TestFutureCovariatesRecursion:
    def test_incremental_matches_reference_with_covariates(self):
        from src.forecasting.models.ml import LightGBMForecaster
        from src.forecasting.preprocessing.features import TemporalFeatureEngineer
        from src.forecasting.preprocessing.holidays import HolidayFeatureEngineer
        from src.forecasting.forecasting.recursive import recursive_forecast

        rng = np.random.default_rng(9)
        n = 220
        dates = pd.date_range("2024-06-01", periods=n, freq="D")
        capacity = 100.0
        occ = 0.6 + 0.08 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.03, n)
        rooms = np.floor(capacity * np.clip(occ, 0.2, 0.95))

        # Forward-known per-date covariate: books at d30 (known for future dates)
        books_d30 = np.floor(rooms * 0.75 + rng.normal(0, 2, n)).clip(0)

        df = pd.DataFrame({
            "date": dates, "rooms_sold": rooms,
            "available_rooms": capacity, "rotb_d30": books_d30,
        })

        eng = TemporalFeatureEngineer(
            lags=[1, 7], rolling_windows=[7], rolling_stats=["mean"],
            expanding_stats=["mean"], cyclical_features=["dayofweek"],
            target_cols=["rooms_sold"],
        )
        he = HolidayFeatureEngineer()
        feat = he.transform(eng.transform(df.copy())).dropna()
        feature_cols = [c for c in feat.columns if c not in ("date", "rooms_sold")]
        assert "rotb_d30" in feature_cols and "available_rooms" in feature_cols

        model = LightGBMForecaster(
            params={"objective": "regression", "n_estimators": 40,
                    "random_state": 0, "verbose": -1}
        )
        model.fit(feat, target_col="rooms_sold", date_col="date", feature_cols=feature_cols)

        hist = df.iloc[:190]
        future = pd.date_range(hist["date"].max() + pd.Timedelta(days=1), periods=25)
        # Future books: known per date (drawn independently — forward-known)
        cov = pd.DataFrame({
            "date": future,
            "rotb_d30": np.floor(60 + 5 * np.sin(np.arange(25) * 2 * np.pi / 7)),
        })

        fast = recursive_forecast(
            model, hist, future, target_col="rooms_sold", date_col="date",
            feature_cols=feature_cols, feature_engineer=eng, holiday_engineer=he,
            use_incremental=True, future_known_cols=["available_rooms"],
            future_covariates=cov,
        )
        ref = recursive_forecast(
            model, hist, future, target_col="rooms_sold", date_col="date",
            feature_cols=feature_cols, feature_engineer=eng, holiday_engineer=he,
            use_incremental=False, future_known_cols=["available_rooms"],
            future_covariates=cov,
        )
        assert len(fast) == len(ref) == 25
        assert not np.isnan(fast).any()
        assert np.mean(np.abs(fast - ref)) / np.mean(np.abs(ref)) < 0.01

    def test_missing_covariate_dates_raise(self):
        from src.forecasting.models.ml import LightGBMForecaster
        from src.forecasting.preprocessing.features import TemporalFeatureEngineer
        from src.forecasting.forecasting.recursive import recursive_forecast

        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        rooms = 60 + 5 * np.sin(2 * np.pi * np.arange(120) / 7)
        df = pd.DataFrame({
            "date": dates, "rooms_sold": rooms,
            "available_rooms": 100.0, "rotb_d30": rooms * 0.7,
        })
        eng = TemporalFeatureEngineer(
            lags=[1, 7], rolling_windows=[7], rolling_stats=["mean"],
            expanding_stats=["mean"], cyclical_features=["dayofweek"],
            target_cols=["rooms_sold"],
        )
        feat = eng.transform(df.copy()).dropna()
        feature_cols = [c for c in feat.columns if c not in ("date", "rooms_sold")]
        model = LightGBMForecaster(
            params={"objective": "regression", "n_estimators": 20,
                    "random_state": 0, "verbose": -1}
        )
        model.fit(feat, target_col="rooms_sold", date_col="date", feature_cols=feature_cols)

        hist = df.iloc[:110]
        future = pd.date_range(hist["date"].max() + pd.Timedelta(days=1), periods=5)
        incomplete_cov = pd.DataFrame({"date": future[:3], "rotb_d30": [40, 41, 42]})

        with pytest.raises(ValueError, match="future_covariates missing"):
            recursive_forecast(
                model, hist, future, target_col="rooms_sold", date_col="date",
                feature_cols=feature_cols, feature_engineer=eng,
                use_incremental=True, future_known_cols=["available_rooms"],
                future_covariates=incomplete_cov,
            )


class TestGeneratorDeterminism:
    def test_same_seed_same_output(self, tmp_path):
        a = generate_real_schema_sample(tmp_path / "a", n_days=90, seed=11,
                                        future_horizon_days=30)
        b = generate_real_schema_sample(tmp_path / "b", n_days=90, seed=11,
                                        future_horizon_days=30)
        pd.testing.assert_frame_equal(a["demand"], b["demand"])
        pd.testing.assert_frame_equal(a["pickup"], b["pickup"])
        pd.testing.assert_frame_equal(a["capacity"], b["capacity"])

    def test_export_queries_documented(self):
        from src.forecasting.data import real_sources
        for sql in (real_sources.REAL_DEMAND_SQL, real_sources.REAL_PICKUP_SQL,
                    real_sources.REAL_CAPACITY_SQL):
            assert "GROUP BY" in sql
