"""Tests del cambio improve-long-horizon-forecast.

- Propiedades cerradas: `data.excluded_properties` las excluye de TODA la
  corrida y el manifiesto las declara.
- Curva de pickup v2: condicionada por temporada del stay, con ventana de
  recencia y fallback por escasez — la mejora de largo plazo.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig, load_config
from src.forecasting.core import build_forecasts, fit_candidate
from src.forecasting.runner import run_end_to_end
from src.models.baselines import PickupCurveModel
from src.monitoring.run_manifest import read_manifest


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _closed_config(excluded: list[str], horizon: int = 30) -> EngineConfig:
    base = load_config()
    return EngineConfig(
        random_seed=42,
        forecast_horizon=horizon,
        percentiles=(10, 50, 90),
        retrain_frequency="monthly",
        min_history_days=365,
        sources=base.sources,
        column_hints=base.column_hints,
        property_aliases={},
        excluded_properties=excluded,
        pickup_curve={},
    )


def _demand_3props(n=300, start="2025-01-01", seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    dow = dates.dayofweek
    base = 55 + 12 * (dow >= 5)
    frames = []
    for prop, mult in (("corin", 1.0), ("fiesta", 0.6), ("cerrada", 0.4)):
        frames.append(
            pd.DataFrame(
                {
                    "property": [prop] * n,
                    "stay_date": dates,
                    "rooms_sold": np.clip(base * mult + rng.normal(0, 4, n), 0, 100).round(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _capacity_3props(n=300, start="2025-01-01"):
    dates = pd.date_range(start, periods=n, freq="D")
    rows = []
    for prop in ("corin", "fiesta", "cerrada"):
        rows.append(
            pd.DataFrame({"propiedad": [prop] * n, "fecha": dates, "inventory": [90] * n})
        )
    return pd.concat(rows, ignore_index=True)


def _pickup_3props(demand: pd.DataFrame):
    rows = []
    rng = np.random.default_rng(3)
    for prop, g in demand.groupby("property"):
        final = g.set_index("stay_date")["rooms_sold"]
        for stay in g["stay_date"].iloc[::8]:
            for lead in [30, 14, 7, 1]:
                snap = stay - pd.Timedelta(days=lead)
                if snap > pd.Timestamp("2025-06-30"):
                    continue
                total = float(final.get(stay, 50))
                rows.append(
                    {
                        "hotel": prop,
                        "stay": stay,
                        "snap": snap,
                        "lead": lead,
                        "booked_rn": max(total * (1 - np.exp(-lead / 35)) + rng.normal(0, 1), 0),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. configuración
# --------------------------------------------------------------------------

def test_config_parsea_exclusion_y_curva():
    cfg = load_config()
    assert cfg.excluded_properties == ["LIRAK", "LAPAS"]
    assert cfg.pickup_curve["season_conditional"] is True
    assert cfg.pickup_curve["season_min_lead"] == 30
    assert cfg.pickup_curve["max_history_days"] == 1095
    assert cfg.pickup_curve["min_stays_per_bucket"] == 30


def test_config_sin_claves_defaults_vacios(tmp_path):
    yaml = tmp_path / "config.yaml"
    yaml.write_text(
        "random_seed: 1\nforecast:\n  horizon_days: 30\n"
        "retraining:\n  frequency: monthly\n  min_history_days: 365\n",
        encoding="utf-8",
    )
    cfg = load_config(yaml)
    assert cfg.excluded_properties == []
    assert cfg.pickup_curve == {}


# --------------------------------------------------------------------------
# 2. exclusión de propiedades cerradas
# --------------------------------------------------------------------------

def test_propiedad_cerrada_excluida_de_toda_la_corrida(tmp_path):
    dem = _demand_3props()
    cap = _capacity_3props()
    pick = _pickup_3props(dem)
    cfg = _closed_config(["cerrada"])

    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "excl",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        config=cfg,
        fold_metrics=None,
    )

    # no aparece en el forecast
    assert "cerrada" not in set(result.forecast_table["property"])
    # ni en el benchmark (fold_metrics sintéticos no llevan propiedad, pero la
    # cobertura y el manifiesto deben declarar la exclusión)
    stored = read_manifest(result.manifest_path)
    assert stored["excluded_properties"] == {"cerrada": "propiedad cerrada (configuración)"}
    coverage = stored["evaluation_coverage"]
    assert coverage["excluded_closed"] == {"cerrada": "propiedad cerrada (configuración)"}
    assert coverage["n_properties_portfolio"] == 2
    # el accuracy report también lo declara
    acc = json.loads(
        (tmp_path / "excl" / "forecast_accuracy_report.json").read_text(encoding="utf-8")
    )
    assert acc["evaluation_coverage"]["excluded_closed"]["cerrada"]
    # el portafolio evaluado son las propiedades activas con reales
    assert "cerrada" not in coverage["evaluated"]


def test_exclusion_tras_alias(tmp_path):
    dem = _demand_3props()
    dem.loc[dem["property"] == "cerrada", "property"] = "Hotel Cerrado"
    cap = _capacity_3props()
    cap.loc[cap["propiedad"] == "cerrada", "propiedad"] = "Hotel Cerrado"
    pick = _pickup_3props(_demand_3props())  # alias distinto en pickup
    pick.loc[pick["hotel"] == "cerrada", "hotel"] = "Hotel Cerrado"

    cfg = _closed_config(["cerrada"])
    cfg = EngineConfig(
        **{**cfg.__dict__, "property_aliases": {"Hotel Cerrado": "cerrada"}}
    )
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "alias",
        cutoff=pd.Timestamp("2025-06-30"),
        sample_data=True,
        config=cfg,
    )
    assert "cerrada" not in set(result.forecast_table["property"])
    stored = read_manifest(result.manifest_path)
    assert stored["excluded_properties"]["cerrada"]


# --------------------------------------------------------------------------
# 3. curva de pickup v2: temporada + recencia + fallback
# --------------------------------------------------------------------------

def _season_curves():
    """Curvas sintéticas: en temporada alta el pickup restante a lead 90 es
    80 RN; en baja es 20; en shoulder 50."""
    rows = []
    for stay_s, season_pickup in (
        ("2025-12-20", 80.0), ("2025-12-27", 80.0), ("2026-01-03", 80.0),  # high
        ("2025-07-05", 20.0), ("2025-07-12", 20.0), ("2025-07-19", 20.0),  # low
        ("2025-03-08", 50.0), ("2025-03-15", 50.0), ("2025-03-22", 50.0),  # shoulder
    ):
        for lead in (90, 30):
            rows.append(
                {
                    "property": "corin",
                    "stay_date": pd.Timestamp(stay_s),
                    "lead_days": lead,
                    "pickup_rn": season_pickup,
                }
            )
    return pd.DataFrame(rows)


def _request(stay: str, lead: int = 90, booked: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{"property": "corin", "stay_date": pd.Timestamp(stay), "lead_days": lead,
          "current_booked_rn": booked}]
    )


def test_curva_estacional_solo_a_partir_de_season_min_lead():
    """A leads cortos la curva estacional NO aplica: preserva el pooling."""
    model = PickupCurveModel(
        n_bins=2, season_conditional=True, min_stays_per_bucket=3, season_min_lead=30
    ).fit(_season_curves())

    # lead 90 (≥ umbral): temporada del stay decide
    high = model.predict(_request("2026-12-24", lead=90))["expected_pickup_rn"].iloc[0]
    low = model.predict(_request("2026-07-24", lead=90))["expected_pickup_rn"].iloc[0]
    assert high == pytest.approx(80.0, abs=0.1)
    assert low == pytest.approx(20.0, abs=0.1)

    # lead 7 (< umbral): curva agrupada de la propiedad (50) para ambas fechas
    short_high = model.predict(_request("2026-12-24", lead=7))["expected_pickup_rn"].iloc[0]
    short_low = model.predict(_request("2026-07-24", lead=7))["expected_pickup_rn"].iloc[0]
    assert short_high == pytest.approx(50.0, abs=0.1)
    assert short_low == pytest.approx(50.0, abs=0.1)


def test_curva_condicionada_por_temporada_del_stay():
    model = PickupCurveModel(
        n_bins=2, season_conditional=True, min_stays_per_bucket=3
    ).fit(_season_curves())

    high = model.predict(_request("2026-12-24"))["expected_pickup_rn"].iloc[0]
    low = model.predict(_request("2026-07-24"))["expected_pickup_rn"].iloc[0]
    assert high == pytest.approx(80.0, abs=0.1)
    assert low == pytest.approx(20.0, abs=0.1)

    # sin condicionamiento ambas fechas reciben el promedio anual (50)
    static = PickupCurveModel(n_bins=2).fit(_season_curves())
    s_high = static.predict(_request("2026-12-24"))["expected_pickup_rn"].iloc[0]
    s_low = static.predict(_request("2026-07-24"))["expected_pickup_rn"].iloc[0]
    assert s_high == pytest.approx(50.0, abs=0.1)
    assert s_low == pytest.approx(50.0, abs=0.1)


def test_curva_fallback_por_escasez():
    # min_stays_per_bucket=10: ningún bucket estacional llega → curva de propiedad
    model = PickupCurveModel(
        n_bins=2, season_conditional=True, min_stays_per_bucket=10
    ).fit(_season_curves())
    val = model.predict(_request("2026-12-24"))["expected_pickup_rn"].iloc[0]
    assert val == pytest.approx(50.0, abs=0.1), "debe degradar a la curva de propiedad"
    assert not model.season_curves_, "ningún bucket debió registrarse"


def test_curva_ventana_de_recencia():
    rows = []
    # régimen viejo: pickup 500 (2020); régimen reciente: pickup 10 (2025)
    for stay_s, pickup in (("2020-12-20", 500.0), ("2020-12-27", 500.0),
                           ("2025-12-05", 10.0), ("2025-12-12", 10.0),
                           ("2025-12-20", 10.0)):
        rows.append(
            {"property": "corin", "stay_date": pd.Timestamp(stay_s),
             "lead_days": 90, "pickup_rn": pickup}
        )
    curves = pd.DataFrame(rows)

    recent = PickupCurveModel(n_bins=2, max_history_days=730).fit(curves)
    val = recent.predict(_request("2026-12-24"))["expected_pickup_rn"].iloc[0]
    assert val == pytest.approx(10.0, abs=0.1), "los stays de 2020 no deben promediarse"

    full = PickupCurveModel(n_bins=2).fit(curves)
    val_full = full.predict(_request("2026-12-24"))["expected_pickup_rn"].iloc[0]
    assert val_full > 10.0, "sin ventana, el régimen viejo contamina la curva"


def test_add_demand_features_preserva_indice():
    """Regresión del bug que la exclusión de LIRAK expuso: el mapeo de
    features por merge reseteaba el índice y la suite de leakage (que alinea
    por índice) comparaba filas equivocadas."""
    from src.features.demand import add_demand_features
    from src.validation.leakage import check_feature_whole_dataset

    n = 1000
    dem = pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": pd.date_range("2025-01-01", periods=n),
            "room_nights": np.full(n, 50.0),
        }
    )
    # índice no contiguo: simula filas filtradas (exclusión de propiedades)
    dem = dem.iloc[::2].copy()  # índice [0, 2, ..., 998]
    out, _ = add_demand_features(dem)
    assert list(out.index) == list(dem.index), "el índice original debe preservarse"

    # el check de features-con-dataset-completo debe alinear filas correctamente
    def fn(frame):
        f, _ = add_demand_features(frame)
        return f

    train = out.iloc[:400]
    finding = check_feature_whole_dataset(
        train, dem, feature_fn=fn, feature_cols=["lag_7", "rolling_mean_7"]
    )
    assert finding is None, "features honestas no deben marcarse como leakage"
    n = 200
    dem = pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": pd.date_range("2025-01-01", periods=n),
            "room_nights": np.full(n, 50.0),
        }
    )
    rows = []
    for stay in dem["stay_date"]:
        for lead in (30, 7):
            rows.append(
                {"property": "corin", "stay_date": stay,
                 "snapshot_date": stay - pd.Timedelta(days=lead),
                 "lead_days": lead, "pickup_rn": 25.0}
            )
    pick = pd.DataFrame(rows)
    c = pd.Timestamp("2025-06-30") - pd.Timedelta(days=1)

    fitted = fit_candidate(
        "pickup_curve", dem, pick, c, 30,
        curve_params={"season_conditional": True, "max_history_days": 365,
                      "min_stays_per_bucket": 30, "n_bins": 4},
    )
    assert fitted.artifact.season_conditional is True
    assert fitted.artifact.max_history_days == 365
    assert fitted.artifact.min_stays_per_bucket == 30

    # build_forecasts propaga los mismos parámetros
    grid, fitted_map = build_forecasts(
        {"corin": "pickup_curve"}, dem, pick, c, 30,
        curve_params={"season_conditional": True, "n_bins": 4},
    )
    assert fitted_map["corin"].artifact.season_conditional is True
    assert len(grid) == 30
