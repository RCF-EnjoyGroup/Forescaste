"""Tests del knob openspec 2026-11-09-band-calibration.

La banda P10-P90 es conformal por construcción sobre los folds del año
previo al corte, pero esa muestra no es intercambiable con la ventana
publicada (verificado sobre real-20261106-1323: cobertura observada
66.3-95.6% por propiedad vs nominal 80%, k necesario 0.64-1.25 con signo
no predecible desde los folds). El knob escala cada semilado por el factor
de su propiedad — aprendido del REGISTRO de corridas evaluables, nunca de
los folds — con shrink por tamaño de pool, clip y gate de observaciones.

Estos tests fijan:
1. knob ausente/`None`/`enabled: false` → banda bit a bit como estaba;
2. knob activo → cada semilado escala exactamente  1+(k-1)*n/(n+shrink)
   (con clip) cuando el pool alcanza el gate;
3. pool insuficiente → k=1 con nota explícita (cold-start no se calibra);
4. sobre una muestra etiquetada (ventana con error más ancho que la
   calibración) el factor mejora la cobertura observada — el what-if
   in-sample que justifica el knob, no una promesa.
"""

import numpy as np
import pandas as pd

from src.forecasting.constraints import build_intervals

BANDA = "30-59"  # lead 45 dias cae aca


def _escenario(prop: str = "corin", n: int = 150, seed: int = 7, sd: float = 10.0):
    rng = np.random.default_rng(seed)
    residuos = rng.normal(0.0, sd, n)
    df = pd.DataFrame(
        {
            "property": [prop] * n,
            "lead_days": [45] * n,
            "final_forecast_rn": [100.0] * n,
            "capacity": [400.0] * n,
        }
    )
    bands = {(prop, BANDA): pd.Series(residuos)}
    return df, bands, residuos


def test_knob_off_deja_la_banda_bit_a_bit():
    """None, {} y enabled:false son el comportamiento historico."""
    df, bands, _ = _escenario()
    base = build_intervals(df, model_name="m", residual_bands=bands, property_col="property")
    for cal in (None, {}, {"enabled": False, "k_by_property": {"corin": 1.8}}):
        otra = build_intervals(
            df, model_name="m", residual_bands=bands, property_col="property",
            band_calibration=cal,
        )
        pd.testing.assert_frame_equal(base.frame, otra.frame)
        assert otra.note == base.note


def test_knob_on_aplica_k_con_shrink_y_clip():
    """k_eff = 1+(k-1)*n/(n+shrinkage), recortado al clip, por propiedad."""
    df, bands, _ = _escenario(n=150)
    base = build_intervals(df, model_name="m", residual_bands=bands, property_col="property")
    cal = {"enabled": True, "k_by_property": {"corin": 1.4},
           "shrinkage_obs": 30, "clip": [0.6, 1.5], "min_obs_property": 120}
    res = build_intervals(
        df, model_name="m", residual_bands=bands, property_col="property",
        band_calibration=cal,
    )
    n, shrink = 150, 30
    k_eff = 1 + (1.4 - 1) * (n / (n + shrink))  # 1.3333...
    lo_base = (base.frame["p50_forecast"] - base.frame["p10_forecast"]).to_numpy()
    hi_base = (base.frame["p90_forecast"] - base.frame["p50_forecast"]).to_numpy()
    lo_new = (res.frame["p50_forecast"] - res.frame["p10_forecast"]).to_numpy()
    hi_new = (res.frame["p90_forecast"] - res.frame["p50_forecast"]).to_numpy()
    assert np.allclose(lo_new / lo_base, k_eff)
    assert np.allclose(hi_new / hi_base, k_eff)
    assert "band_calibration aplicada" in res.note


def test_pool_insuficiente_no_calibra():
    """Pool < min_obs_property → k=1 y nota explícita (sin calibración a ciegas)."""
    df, bands, _ = _escenario(n=60)  # 60 < 120
    cal = {"enabled": True, "k_by_property": {"corin": 1.4},
           "shrinkage_obs": 30, "clip": [0.6, 1.5], "min_obs_property": 120}
    base = build_intervals(df, model_name="m", residual_bands=bands, property_col="property")
    res = build_intervals(
        df, model_name="m", residual_bands=bands, property_col="property",
        band_calibration=cal,
    )
    pd.testing.assert_frame_equal(base.frame, res.frame)
    assert "band_calibration omitida por pool insuficiente" in res.note


def test_knob_mejora_cobertura_en_ventana_mas_ancha():
    """What-if in-sample etiquetado: la ventana publicada tiene error más
    ancho que el año de calibración; el factor acerca la cobertura al
    nominal. Mismo mecanismo medido en real-20261106-1323 (FIESTA/SJOSL)."""
    rng = np.random.default_rng(11)
    n = 200
    cal_residuos = rng.normal(0.0, 8.0, 300)          # año de calibración: sd 8
    ventana_error = rng.normal(0.0, 20.0, n)          # ventana publicada: sd 20
    df = pd.DataFrame(
        {
            "property": ["fiesta"] * n,
            "lead_days": [45] * n,
            "final_forecast_rn": [100.0] * n,
            "capacity": [400.0] * n,
            "_actual": 100.0 + ventana_error,
        }
    )
    bands = {("fiesta", BANDA): pd.Series(cal_residuos)}
    cal = {"enabled": True, "k_by_property": {"fiesta": 1.5},
           "shrinkage_obs": 30, "clip": [0.6, 1.5], "min_obs_property": 120}

    def _cobertura(res_frame):
        p10, p90 = res_frame["p10_forecast"], res_frame["p90_forecast"]
        return float(((df["_actual"] >= p10) & (df["_actual"] <= p90)).mean())

    base = build_intervals(df, model_name="m", residual_bands=bands, property_col="property")
    calib = build_intervals(
        df, model_name="m", residual_bands=bands, property_col="property",
        band_calibration=cal,
    )
    cov_base = _cobertura(base.frame)
    cov_cal = _cobertura(calib.frame)
    assert cov_cal > cov_base + 0.05, f"{cov_base:.2%} -> {cov_cal:.2%}"
