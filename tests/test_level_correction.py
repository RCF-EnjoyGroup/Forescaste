"""Corrección de nivel por drift del forecast publicado — openspec
2026-11-05-drift-level-correction.

La señal es el error del forecast que el negocio YA vio (vintage_detail
de la publicación previa): walk-forward puro, sin lookahead. Estas
pruebas fijan el estimador (peso suave por TS, shrinkage, clip), el
aplicador multiplicativo, el gate de modo (solo futuro) y el knob OFF
bit a bit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.core import apply_level_factors, level_correction_factors


def _detalle_con_drift(prop: str = "TEST", sesgo: float = 0.20,
                       n: int = 40, ts_alto: bool = True) -> pd.DataFrame:
    """Errores publicados con sesgo constante: fc = act·(1−sesgo) abajo.

    Con errores proporcionales (misma magnitud relativa) la TS escala con
    n: 40 días de −20% → TS muy por encima de 4 (trigger pleno).
    """
    fechas = pd.date_range("2026-09-01", periods=n, freq="D")
    actual = 100.0
    filas = []
    for d in fechas:
        act = actual + (d.day_of_week % 3)  # variación mínima realista
        filas.append({
            "property": prop, "stay_date": d, "forecast_date": d - pd.Timedelta(days=30),
            "published_forecast_rn": act * (1 - sesgo), "actual_rn": act,
        })
    return pd.DataFrame(filas)


# ---------- 1. estimador ----------


def test_drift_sostenido_dispara_correccion_al_alza():
    detail = _detalle_con_drift(sesgo=0.20)
    factors, report = level_correction_factors(detail)
    assert factors.get("TEST") is not None
    f = factors["TEST"]
    assert 1.05 < f <= 1.20, "un −20% sostenido debe corregir al alza, clipado a +20%"
    fila = report["properties"][0]
    assert fila["weight"] == 1.0 and fila["applied"]


def test_sesgo_chico_pesa_menos_que_sesgo_grande():
    leve = level_correction_factors(_detalle_con_drift(sesgo=0.05))[0].get("TEST")
    fuerte = level_correction_factors(_detalle_con_drift(sesgo=0.20))[0].get("TEST")
    assert leve is not None and fuerte is not None
    assert leve < fuerte, "menos drift → menos corrección"


def test_peso_suave_sin_estado():
    # errores mixtos (unos abajo, otros arriba): la TS queda intermedia
    # → peso parcial, no todo-o-nada
    detail = _detalle_con_drift(n=12)
    mitad = detail.index[len(detail) // 2:]
    detail.loc[mitad, "published_forecast_rn"] = (
        detail.loc[mitad, "actual_rn"] * 1.1333
    )
    _, report = level_correction_factors(detail)
    fila = report["properties"][0]
    assert 0.0 < fila["weight"] < 1.0, "TS intermedia → peso parcial"
    assert fila["applied"]


def test_n_bajo_no_estima():
    detail = _detalle_con_drift(n=5)
    factors, report = level_correction_factors(detail, min_obs=10)
    assert "TEST" not in factors
    assert report["properties"][0]["reason"].startswith("n<")


def test_ventana_recorte_60_dias():
    # 90 dias de drift, ventana 60: solo las ultimas 60 entran
    detail = _detalle_con_drift(n=90)
    _, report = level_correction_factors(detail, window_days=60)
    fila = report["properties"][0]
    assert fila["n_obs"] == 60


def test_clip_y_umbral_inmaterial():
    # sesgo 0.5% → correccion inmaterial → factor 1.0 con razon
    detail = _detalle_con_drift(sesgo=0.005)
    factors, report = level_correction_factors(detail)
    assert "TEST" not in factors
    assert report["properties"][0]["reason"] == "correccion inmaterial (<0.5%)"
    # clip: sesgo enorme no pasa de ±max_rel
    detail_big = _detalle_con_drift(sesgo=0.60)
    factors_big, _ = level_correction_factors(detail_big, max_rel=0.20)
    assert factors_big["TEST"] == pytest.approx(1.20)


def test_sin_detalle_devuelve_reporte_honesto():
    factors, report = level_correction_factors(None)
    assert factors == {} and "note" in report
    factors2, report2 = level_correction_factors(pd.DataFrame())
    assert factors2 == {} and "note" in report2


# ---------- 2. aplicador ----------


def test_aplicador_multiplicativo_y_columna():
    hz = pd.DataFrame({
        "property": ["A", "B"], "lead_days": [10, 10],
        "raw_forecast_rn": [100.0, 200.0],
    })
    out = apply_level_factors(hz, {"A": 1.10})
    assert float(out.loc[0, "raw_forecast_rn"]) == pytest.approx(110.0)
    assert float(out.loc[1, "raw_forecast_rn"]) == pytest.approx(200.0)
    assert out["level_correction_factor"].tolist() == [1.10, 1.0]


def test_aplicador_sin_factores_devuelve_intacto():
    hz = pd.DataFrame({"property": ["A"], "raw_forecast_rn": [100.0]})
    out = apply_level_factors(hz, None)
    assert "level_correction_factor" not in out.columns
    assert float(out.loc[0, "raw_forecast_rn"]) == 100.0


# ---------- 3. composición con la corrección por banda ----------


def test_nivel_y_banda_componen_multiplicativos():
    from src.forecasting.core import apply_bias_factors

    hz = pd.DataFrame({
        "property": ["A"], "lead_days": [10],
        "stay_date": pd.to_datetime(["2026-11-10"]),
        "raw_forecast_rn": [100.0],
    })
    hz1 = apply_level_factors(hz, {"A": 1.10})
    hz2 = apply_bias_factors(hz1, {("A", "7-13"): 0.98})
    assert float(hz2.loc[0, "raw_forecast_rn"]) == pytest.approx(107.8)
    assert float(hz2.loc[0, "level_correction_factor"]) == pytest.approx(1.10)
    assert float(hz2.loc[0, "bias_correction_factor"]) == pytest.approx(0.98)
