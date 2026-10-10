"""Tests de monitoring (tasks 10.1–10.4)."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    OUTLIER_CATEGORIES,
    check_distribution_drift,
    check_metric_drift,
    check_pickup_curve_drift,
    classified_frame,
    detect_outliers,
    run_drift_suite,
    should_retrain,
)


# --------------------------- 10.1 drift de métricas ----------------------


def test_deterioro_mape_genera_alerta_con_metrica_y_magnitud():
    ref = {"mape": 10.0, "wmape": 0.10, "bias": 0.02}
    cur = {"mape": 13.5, "wmape": 0.13, "bias": 0.03}  # +35% MAPE

    alerts = check_metric_drift(ref, cur)
    mape_alert = next(a for a in alerts if a.metric == "mape")

    assert mape_alert.severity == "alert", "+35% > umbral alert 25%"
    assert "MAPE" in mape_alert.message.upper()
    assert f"{ref['mape']:.4f}" in mape_alert.message
    assert f"{cur['mape']:.4f}" in mape_alert.message
    # magnitud disponible
    assert mape_alert.magnitude == pytest.approx(3.5)
    assert mape_alert.relative_change == pytest.approx(0.35)
    d = mape_alert.to_dict()
    assert {"metric", "severity", "message", "reference", "current", "magnitude"} <= set(d)


def test_deterioro_leve_warning():
    ref = {"mape": 10.0, "wmape": 0.10, "bias": 0.02}
    cur = {"mape": 11.0, "wmape": 0.105, "bias": 0.02}  # +10% MAPE (justo en warning)
    alerts = check_metric_drift(ref, cur)
    mape = next(a for a in alerts if a.metric == "mape")
    assert mape.severity == "warning"


def test_sin_deterioro_sin_alerta():
    ref = {"mape": 10.0, "wmape": 0.10, "bias": 0.02}
    cur = {"mape": 9.0, "wmape": 0.09, "bias": 0.01}  # mejoró
    alerts = check_metric_drift(ref, cur)
    assert alerts == [], f"mejora no debe alertar: {alerts}"


def test_bias_drift_por_puntos():
    ref = {"bias": 0.02}
    cur = {"bias": 0.15}  # +13 puntos de sesgo
    alerts = check_metric_drift(ref, cur)
    bias = next(a for a in alerts if a.metric == "bias")
    assert bias.severity == "alert"  # delta 0.13 >= 0.10
    assert bias.metric == "bias"


def test_metricas_faltantes_no_crashean():
    assert check_metric_drift({"mape": 10.0}, {"wmape": 0.1}) == []
    assert check_metric_drift({"mape": float("nan")}, {"mape": 20.0}) == []


# --------------------------- 10.2 drift de features/curva/demanda -------


def test_drift_distribucion_demanda():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(60, 10, 500))
    cur = pd.Series(rng.normal(85, 10, 500))  # +2.5σ

    alert = check_distribution_drift(ref, cur, "demand_distribution")
    assert alert is not None
    assert alert.metric == "demand_distribution"
    assert alert.severity == "alert"
    assert "2" in alert.message or "σ" in alert.message


def test_sin_drift_distribucion():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(60, 10, 500))
    cur = pd.Series(rng.normal(61, 10, 500))
    assert check_distribution_drift(ref, cur, "demand_distribution") is None


def test_drift_curva_pickup():
    leads = [90, 60, 30, 14, 7, 1]
    ref = pd.DataFrame({"lead_days": leads, "pickup_rn": [40, 30, 20, 12, 6, 2]})
    # la curva actual se desplaza ~50% hacia arriba
    cur = pd.DataFrame({"lead_days": leads, "pickup_rn": [60, 45, 30, 18, 9, 3]})

    alert = check_pickup_curve_drift(ref, cur)
    assert alert is not None
    assert alert.metric == "pickup_curve"
    assert alert.severity == "alert"
    assert alert.relative_change != 0


def test_curva_pickup_sin_drift():
    leads = [90, 60, 30]
    ref = pd.DataFrame({"lead_days": leads, "pickup_rn": [40, 30, 20]})
    cur = pd.DataFrame({"lead_days": leads, "pickup_rn": [40, 30, 20]})
    assert check_pickup_curve_drift(ref, cur) is None


def test_drift_features_y_suite_completa():
    rng = np.random.default_rng(1)
    ref_metrics = {"mape": 10.0, "wmape": 0.10, "bias": 0.02}
    cur_metrics = {"mape": 14.0, "wmape": 0.13, "bias": 0.03}

    ref_feat = pd.DataFrame({"current_booked_rn": rng.normal(50, 5, 200)})
    cur_feat = pd.DataFrame({"current_booked_rn": rng.normal(70, 5, 200)})  # drift

    alerts = run_drift_suite(
        reference_metrics=ref_metrics,
        current_metrics=cur_metrics,
        reference_demand=pd.Series(rng.normal(60, 10, 300)),
        current_demand=pd.Series(rng.normal(60, 10, 300)),  # sin drift
        feature_reference=ref_feat,
        feature_current=cur_feat,
    )

    metrics = {a.metric for a in alerts}
    assert "mape" in metrics          # 10.1
    assert "feature:current_booked_rn" in metrics  # 10.2
    assert "demand_distribution" not in metrics  # sin drift => no alerta
    assert all(a.severity in ("warning", "alert") for a in alerts)


# --------------------------- 10.3 outliers -------------------------------


def _demand_with_spike():
    n = 60
    rng = np.random.default_rng(2)
    values = rng.normal(50, 4, n)
    values[30] = 95.0   # pico aislado (evento)
    values[45] = 20.0   # caída abrupta
    return pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "room_nights": values,
            "capacity": [100.0] * n,
        }
    )


def test_outliers_detectados_y_clasificados():
    df = _demand_with_spike()
    outliers = detect_outliers(df, threshold=3.0)

    assert outliers, "debe detectar el pico"
    for o in outliers:
        assert o.category in OUTLIER_CATEGORIES
        assert o.rationale
        assert o.removed is False, "NUNCA se eliminan automáticamente"
        assert abs(o.zscore) >= 3.0


def test_outlier_evento_real_se_conserva():
    """Un evento real se clasifica y se CONSERVA en el entrenamiento."""
    df = _demand_with_spike()
    outliers = detect_outliers(df, threshold=3.0)
    spike = next(o for o in outliers if o.value == 95.0)
    assert spike.category in ("event", "seasonality", "structural")
    assert spike.removed is False

    # el frame de entrenamiento NO se modifica (nunca se filtran)
    assert len(detect_outliers(df)) >= 1
    assert len(df) == 60, "detect_outliers no debe mutar el dataset"


def test_outlier_error_de_datos_por_capacidad():
    df = _demand_with_spike()
    df.loc[10, "room_nights"] = 150.0  # > capacidad 100
    df.loc[10, "stay_date"] = pd.Timestamp("2026-01-11")
    outliers = detect_outliers(df, threshold=3.0)
    err = next((o for o in outliers if o.value == 150.0), None)
    assert err is not None, "RN > capacidad debe detectarse"
    assert err.category == "data_error"
    assert "capacidad" in err.rationale


def test_outlier_capacidad_reducida_es_estructural():
    df = _demand_with_spike()
    # el día de la caída (20 RN) tiene capacidad reducida: la "anomalía" la
    # explica el inventario, no la demanda => structural (y no es imposible
    # físico: 20 <= 40)
    df.loc[45, "capacity"] = 40.0
    outliers = detect_outliers(df, threshold=3.0)
    o = next(x for x in outliers if x.value == 20.0)
    assert o.category == "structural"
    assert "capacidad reducida" in o.rationale


def test_outlier_frame_y_sin_outliers():
    assert classified_frame([]).empty
    df = _demand_with_spike()
    frame = classified_frame(detect_outliers(df))
    assert {"property", "date", "value", "zscore", "category", "rationale", "removed"} <= set(
        frame.columns
    )
    assert not frame["removed"].any()


def test_outlier_congeladas_sin_outliers():
    df = pd.DataFrame(
        {
            "property": ["corin"] * 20,
            "stay_date": pd.date_range("2026-01-01", periods=20),
            "room_nights": [50.0] * 20,  # sin varianza
            "capacity": [100.0] * 20,
        }
    )
    assert detect_outliers(df) == []


# --------------------------- 10.4 retraining -----------------------------


def test_retraining_programado_se_ejecuta():
    d = should_retrain(
        frequency="weekly",
        new_records=500,
        days_since_last=10,
        min_history_days=365,
        current_history_days=730,
    )
    assert d.should_retrain
    assert "programado" in d.reason
    assert d.frequency == "weekly"


def test_sin_datos_nuevos_suficientes_se_omite():
    d = should_retrain(
        frequency="weekly",
        new_records=0,
        days_since_last=10,
        min_history_days=365,
        current_history_days=730,
    )
    assert not d.should_retrain
    assert "sin información nueva suficiente" in d.reason
    assert d.to_dict()["should_retrain"] is False


def test_frecuencia_aun_no_vencida():
    d = should_retrain(
        frequency="monthly",
        new_records=1000,
        days_since_last=5,
        min_history_days=365,
        current_history_days=730,
    )
    assert not d.should_retrain
    assert "no vencida" in d.reason


def test_historico_insuficiente():
    d = should_retrain(
        frequency="weekly",
        new_records=100,
        days_since_last=30,
        min_history_days=365,
        current_history_days=100,
    )
    assert not d.should_retrain
    assert "histórico insuficiente" in d.reason
    assert "MIN_HISTORY" in d.reason


def test_frecuencia_never():
    d = should_retrain(
        frequency="never",
        new_records=9999,
        days_since_last=1000,
        min_history_days=30,
        current_history_days=1000,
    )
    assert not d.should_retrain
    assert "never" in d.reason


def test_frecuencia_desconocida():
    d = should_retrain(
        frequency="fortnightly_bogus",
        new_records=100,
        days_since_last=100,
        min_history_days=10,
        current_history_days=100,
    )
    assert not d.should_retrain
    assert "desconocida" in d.reason
