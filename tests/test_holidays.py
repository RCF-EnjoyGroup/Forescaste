"""Tests del cambio add-holiday-features.

- Calendario CR+US: fechas movibles (Semana Santa por computus, Thanksgiving,
  Memorial), ventanas de viaje, empates al más cercano, agrupación semántica
  insensible a acentos.
- Curva de pickup: nivel de condicionamiento por ventana de feriado con
  fallback (propiedad, feriado) → temporada → propiedad.
- ML: ``days_to_nearest_holiday`` en la matriz; exógena a cualquier lead.
- Config + e2e: calendario activo declara ``is_holiday`` y el manifiesto
  registra la configuración de feriados.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig, load_config
from src.features.calendar import add_calendar_features, has_holiday_feature
from src.features.holidays import (
    HolidayCalendar,
    build_holiday_calendar,
    holiday_calendar_from_config,
    holiday_key,
)
from src.forecasting.core import ML_FEATURES, _ml_row, build_forecasts, fit_candidate
from src.forecasting.runner import run_end_to_end
from src.models.baselines import PickupCurveModel
from src.monitoring.run_manifest import read_manifest


# --------------------------------------------------------------------------
# 1. calendario
# --------------------------------------------------------------------------

def test_fechas_movibles_resueltas_correctamente():
    cal = build_holiday_calendar(range(2026, 2027), ("CR", "US"), window_days=4)
    # Semana Santa 2026: Jueves Santo 2-abr, Viernes Santo 3-abr (computus)
    assert pd.Timestamp("2026-04-02") in set(cal.dates)
    assert pd.Timestamp("2026-04-03") in set(cal.dates)
    # Thanksgiving 2026: 4.º jueves de noviembre = 26-nov
    assert pd.Timestamp("2026-11-26") in set(cal.dates)
    # Memorial Day 2026: último lunes de mayo = 25-may
    assert pd.Timestamp("2026-05-25") in set(cal.dates)
    # claves semánticas
    assert cal.window_key(pd.Timestamp("2026-04-03")) == "semana_santa"
    assert cal.window_key(pd.Timestamp("2026-11-26")) == "thanksgiving"
    assert cal.window_key(pd.Timestamp("2026-12-25")) == "fin_de_ano"
    assert cal.window_key(pd.Timestamp("2026-01-01")) == "fin_de_ano"


def test_ventanas_y_empate_al_mas_cercano():
    cal = build_holiday_calendar(range(2026, 2027), ("CR", "US"), window_days=4)
    # dentro de la ventana ±4: 30-nov (Thanksgiving 26-nov + 4)
    assert cal.window_key(pd.Timestamp("2026-11-30")) == "thanksgiving"
    # fuera de toda ventana (Columbus 12-oct+4=16; Veterans 11-nov−4=7)
    assert cal.window_key(pd.Timestamp("2026-10-20")) is None
    # días exactos al feriado más cercano
    d = cal.days_to_nearest(pd.Timestamp("2026-11-26"))
    assert d == 0.0
    assert cal.days_to_nearest(pd.Timestamp("2026-11-28")) == 2.0
    assert cal.is_holiday(pd.Timestamp("2026-12-25"))
    assert not cal.is_holiday(pd.Timestamp("2026-12-24"))


def test_holiday_key_agrupacion_semantica():
    assert holiday_key("Jueves Santo") == "semana_santa"
    assert holiday_key("Viernes Santo") == "semana_santa"
    assert holiday_key("Navidad") == "fin_de_ano"
    assert holiday_key("Año Nuevo") == "fin_de_ano"
    assert holiday_key("New Year's Day") == "fin_de_ano"
    assert holiday_key("Anexión del Partido de Nicoya a Costa Rica") == "anexion"
    assert holiday_key("Día de la Independencia") == "independencia_cr"
    assert holiday_key("Independence Day (observed)") == "independencia_us"
    assert holiday_key("Día de la Madre") == "dia_de_la_madre"
    # feriado sin regla → slug estable sin acentos
    assert holiday_key("Día Internacional del Trabajo") == "dia_internacional_del_trabajo"


def test_calendario_vacio_y_cache():
    cal = build_holiday_calendar([], ("CR",), window_days=4)
    assert len(cal.dates) == 0
    assert cal.window_key(pd.Timestamp("2026-12-25")) is None
    assert np.isnan(cal.days_to_nearest(pd.Timestamp("2026-12-25")))
    # cache: misma instancia para los mismos parámetros
    a = build_holiday_calendar(range(2026, 2027), ("CR", "US"), 4)
    b = build_holiday_calendar(range(2026, 2027), ("CR", "US"), 4)
    assert a is b


def test_config_parsea_holidays():
    cfg = load_config()
    assert cfg.holidays["countries"] == ["CR", "US"]
    assert cfg.holidays["window_days"] == 4
    assert cfg.pickup_curve["holiday_conditional"] is True
    assert cfg.pickup_curve["min_stays_per_holiday_bucket"] == 15
    # deshabilitado → None
    off = EngineConfig(
        random_seed=1, forecast_horizon=30, percentiles=(10, 50, 90),
        retrain_frequency="monthly", min_history_days=365, sources={},
        column_hints={}, holidays={"countries": [], "window_days": 4},
    )
    assert holiday_calendar_from_config(off, 2024, 2026) is None


# --------------------------------------------------------------------------
# 2. curva de pickup: nivel feriado
# --------------------------------------------------------------------------

def _holy_week_curves():
    """Curvas sintéticas por grupo limpio:

    - Semana Santa (mar/abr, dentro de la ventana ±4): pickup 90 (6 estancias)
    - Otros shoulder fuera de ventana: pickup 60 (6 estancias)
      → bucket de temporada shoulder = (90×6 + 60×6)/12 = 75
    - Temporada baja (jul): pickup 20
    """
    rows = []
    stays = (
        # semana santa (dentro de la ventana ±4 del Jueves/Viernes Santo)
        [("2026-03-30", 90.0), ("2026-04-02", 90.0), ("2026-04-05", 90.0)]
        + [("2025-04-15", 90.0), ("2025-04-18", 90.0), ("2025-04-20", 90.0)]
        # shoulder normal (marzo/abril/mayo fuera de la ventana)
        + [("2026-03-16", 60.0), ("2026-03-20", 60.0), ("2026-04-27", 60.0),
           ("2026-04-28", 60.0), ("2026-05-08", 60.0), ("2026-05-12", 60.0)]
        # temporada baja
        + [("2026-07-10", 20.0), ("2026-07-15", 20.0), ("2026-07-20", 20.0),
           ("2026-07-25", 20.0), ("2026-07-30", 20.0), ("2026-08-05", 20.0)]
    )
    for stay_s, pickup in stays:
        rows.append(
            {"property": "corin", "stay_date": pd.Timestamp(stay_s),
             "lead_days": 60, "pickup_rn": pickup}
        )
    return pd.DataFrame(rows)


def _req(stay: str, lead: int = 60, booked: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{"property": "corin", "stay_date": pd.Timestamp(stay), "lead_days": lead,
          "current_booked_rn": booked}]
    )


def test_curva_condicionada_por_ventana_de_feriado():
    cal = build_holiday_calendar(range(2025, 2027), ("CR", "US"), window_days=4)
    model = PickupCurveModel(
        n_bins=2, season_conditional=True, holiday_conditional=True,
        min_stays_per_bucket=3, season_min_lead=30, holiday_calendar=cal,
    ).fit(_holy_week_curves())

    # stay DENTRO de la ventana de Semana Santa 2026 (lead ≥ umbral)
    holy = model.predict(_req("2026-04-04"))["expected_pickup_rn"].iloc[0]
    assert holy == pytest.approx(90.0, abs=0.1), "Semana Santa ≠ un marzo cualquiera"
    # stay shoulder FUERA de la ventana → curva de temporada (75 = mezcla 90/60)
    normal = model.predict(_req("2026-04-18"))["expected_pickup_rn"].iloc[0]
    assert normal == pytest.approx(75.0, abs=0.1)
    # stay de temporada baja → 20
    low = model.predict(_req("2026-07-15"))["expected_pickup_rn"].iloc[0]
    assert low == pytest.approx(20.0, abs=0.1)
    # lead corto (< umbral): pooling de la propiedad (~58 = mezcla de todo)
    short = model.predict(_req("2026-04-04", lead=7))["expected_pickup_rn"].iloc[0]
    assert short != pytest.approx(90.0, abs=1.0), "el umbral debe gatear el bucket de feriado"


def test_curva_feriado_fallback_por_bucket_delgado():
    cal = build_holiday_calendar(range(2025, 2027), ("CR", "US"), window_days=4)
    # umbral 10: bloquea el bucket de feriado (6 estancias) pero permite el
    # de temporada shoulder (12) → el lookup degrada feriado → temporada
    model = PickupCurveModel(
        n_bins=2, season_conditional=True, holiday_conditional=True,
        min_stays_per_bucket=10, season_min_lead=30, holiday_calendar=cal,
    ).fit(_holy_week_curves())
    val = model.predict(_req("2026-04-04"))["expected_pickup_rn"].iloc[0]
    assert val == pytest.approx(75.0, abs=0.1)
    assert not model.holiday_curves_


def test_umbral_propio_para_buckets_de_feriado():
    """Las ventanas de feriado (~9 días) no pueden cumplir la barra de las
    temporadas (meses): el umbral de feriado es independiente."""
    cal = build_holiday_calendar(range(2025, 2027), ("CR", "US"), window_days=4)
    # min_stays_per_bucket=50 bloquea TODO (temporada y feriado)…
    blocked = PickupCurveModel(
        n_bins=2, season_conditional=True, holiday_conditional=True,
        min_stays_per_bucket=50, season_min_lead=30, holiday_calendar=cal,
    ).fit(_holy_week_curves())
    val = blocked.predict(_req("2026-04-04"))["expected_pickup_rn"].iloc[0]
    assert val == pytest.approx(56.667, abs=0.1), "debe caer a la curva de propiedad"
    # …pero min_stays_per_holiday_bucket=6 deja activo SOLO el bucket de feriado
    only_holiday = PickupCurveModel(
        n_bins=2, season_conditional=True, holiday_conditional=True,
        min_stays_per_bucket=50, min_stays_per_holiday_bucket=6,
        season_min_lead=30, holiday_calendar=cal,
    ).fit(_holy_week_curves())
    val2 = only_holiday.predict(_req("2026-04-04"))["expected_pickup_rn"].iloc[0]
    assert val2 == pytest.approx(90.0, abs=0.1), "el bucket de feriado aplica pese a la barra de temporadas"
    assert set(k for _, k in only_holiday.holiday_curves_) == {"semana_santa"}


# --------------------------------------------------------------------------
# 3. features ML
# --------------------------------------------------------------------------

def test_ml_feature_dias_a_feriado():
    cal = build_holiday_calendar(range(2025, 2027), ("CR", "US"), window_days=4)
    vals = np.full(400, 50.0)
    row = _ml_row(vals, 399, pd.Timestamp("2026-12-25"), 20.0, cal)
    assert row["days_to_nearest_holiday"] == 0.0
    row_3 = _ml_row(vals, 399, pd.Timestamp("2026-12-28"), 20.0, cal)
    assert row_3["days_to_nearest_holiday"] == 3.0
    # sin calendario → NaN (la columna SIEMPRE está en la matriz)
    row_none = _ml_row(vals, 399, pd.Timestamp("2026-12-25"), 20.0, None)
    assert np.isnan(row_none["days_to_nearest_holiday"])
    assert "days_to_nearest_holiday" in ML_FEATURES


def test_calendario_llega_al_candidato_ml_y_a_la_curva():
    n = 300
    dem = pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": pd.date_range("2025-06-01", periods=n),
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
    c = dem["stay_date"].max()
    cal = build_holiday_calendar(range(2025, 2027), ("CR", "US"), 4)

    fitted = fit_candidate("lightgbm", dem, pick, c, 30, holiday_calendar=cal)
    assert fitted.holiday_calendar is cal
    fitted_pu = fit_candidate(
        "pickup_curve", dem, pick, c, 30,
        curve_params={"season_conditional": True, "holiday_conditional": True,
                      "min_stays_per_bucket": 3, "n_bins": 4},
        holiday_calendar=cal,
    )
    assert fitted_pu.artifact.holiday_calendar is cal

    grid, fitted_map, _blend = build_forecasts(
        {"corin": "pickup_curve"}, dem, pick, c, 30,
        curve_params={"holiday_conditional": True, "n_bins": 4},
        holiday_calendar=cal,
    )
    assert fitted_map["corin"].artifact.holiday_conditional is True


# --------------------------------------------------------------------------
# 4. e2e: is_holiday + manifiesto
# --------------------------------------------------------------------------

def _holiday_config(countries: list[str]) -> EngineConfig:
    base = load_config()
    return EngineConfig(
        random_seed=42, forecast_horizon=30, percentiles=(10, 50, 90),
        retrain_frequency="monthly", min_history_days=365,
        sources=base.sources, column_hints=base.column_hints,
        excluded_properties=[], pickup_curve={},
        holidays={"countries": countries, "window_days": 2},
    )


def test_e2e_con_feriados_declarados(tmp_path):
    from tests.test_long_horizon import (
        _capacity_3props, _demand_3props, _pickup_3props,
    )

    dem = _demand_3props()
    cap = _capacity_3props()
    pick = _pickup_3props(dem)
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "holidays",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        config=_holiday_config(["CR", "US"]),
    )
    assert result.checks.all_passed, (
        f"checks fallidos: {result.checks.failures()} details={result.checks.details}"
    )
    stored = read_manifest(result.manifest_path)
    assert stored["hyperparameters"]["holidays"]["countries"] == ["CR", "US"]
    assert stored["holiday_calendar"]["n_holiday_dates"] > 0
    # el feature report del e2e registra el indicador de feriado presente
    feat = json.loads(
        (tmp_path / "holidays" / "feature_engineering_report.json").read_text(encoding="utf-8")
    )
    assert feat["n_features"] > 0  # sanity
    # con calendario vacío el manifiesto lo declara deshabilitado
    result_off = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "holidays_off",
        cutoff=pd.Timestamp("2025-06-30"),
        sample_data=True,
        config=_holiday_config([]),
    )
    stored_off = read_manifest(result_off.manifest_path)
    assert stored_off["hyperparameters"]["holidays"]["countries"] == []
    assert stored_off["holiday_calendar"] is None


def test_is_holiday_feature_con_calendario():
    cal = build_holiday_calendar(range(2026, 2027), ("CR",), window_days=2)
    df = pd.DataFrame({"stay_date": pd.to_datetime(["2026-12-24", "2026-12-25", "2026-10-05"])})
    out = add_calendar_features(df, "stay_date", holidays=list(cal.dates))
    assert has_holiday_feature(out)
    assert out["is_holiday"].tolist() == [False, True, False]
    out_off = add_calendar_features(df, "stay_date", holidays=None)
    assert not has_holiday_feature(out_off)
