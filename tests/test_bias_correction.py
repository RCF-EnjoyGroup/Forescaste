"""Tests de la corrección de bias por (propiedad, banda) y la banda conformal.

Contexto (corrida de verificación real-20261016-0714):

- el backtest operativo documenta sobre-pronóstico sistemático a leads
  largos (bias +23% en 60-89, +30% en 90+) — se corrige en el PUNTO con
  evidencia de folds, shrinkage por tamaño de muestra;
- la banda P10-P90 de-mediaaba los residuos (q10−q50, q90−q50) bajo el
  supuesto «el sesgo vive en el punto»: falsado por la cobertura empírica
  (CORIN 67%, FIESTA 68% vs 80% nominal). Con cuantiles completos
  (−q90, −q10) la banda contiene el 80% de la muestra de calibración POR
  CONSTRUCCIÓN, sesgo incluido;
- la histéresis del campeón retiene al incumbente salvo ventaja material.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.selection import select_per_property
from src.forecasting.constraints import build_intervals
from src.forecasting.core import (
    apply_bias_factors,
    lead_bias_factors,
    operative_residual_bands,
)


def _fold_predictions_bias(
    prop="corin", n_dias=90, n_folds=2, seed=3, over_forecast_long=0.30
):
    """Predicciones sintéticas del campeón con sobre-pronóstico a lead largo."""
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(n_folds):
        base = pd.Timestamp("2026-01-01") + pd.Timedelta(days=90 * fold)
        dates = pd.date_range(base + pd.Timedelta(days=1), periods=n_dias)
        for d in dates:
            lead = (d - base).days
            actual = 100.0 + 10 * np.sin(2 * np.pi * d.day_of_week / 7)
            bias = 0.05 if lead <= 30 else over_forecast_long
            forecast = actual * (1.0 + bias) + rng.normal(0, 3)
            rows.append({"property": prop, "fold": fold, "model": "pickup_curve",
                         "stay_date": d, "lead_days": lead,
                         "forecast": forecast, "actual": actual})
    return pd.DataFrame(rows)


# ------------------------------------------------------- lead_bias_factors


def test_sobre_pronóstico_largo_produce_factor_menor_que_uno():
    preds = _fold_predictions_bias()
    factors, report = lead_bias_factors(
        preds, {"corin": "pickup_curve"}, {}, pd.DataFrame()
    )
    assert factors[("corin", "60-89")] < 1.0, "sobre-pronóstico → corrección a la baja"
    assert factors[("corin", "30-59")] < 1.0
    assert report["n_corrected"] >= 2
    applied = [c for c in report["cells"] if c["applied"]]
    assert all(c["rel_bias"] > 0 for c in applied)


def test_bias_chico_con_shrinkage_no_corrige_de_mas():
    """Con bias leve y muestra chica, el shrinkage mantiene la corrección mínima."""
    preds = _fold_predictions_bias(over_forecast_long=0.02)
    factors, report = lead_bias_factors(
        preds, {"corin": "pickup_curve"}, {}, pd.DataFrame()
    )
    # celdas de lead LARGO (bias crudo 2%): la corrección shrinked queda
    # por debajo del bias crudo + ruido de estimación
    for band in ("30-59", "60-89"):
        f = factors.get(("corin", band))
        if f is not None and f < 1.0:
            assert 0.97 <= f < 1.0, f"corrección {f} fuera del rango del bias 2%"


def test_celda_con_pocos_residuos_no_corrige():
    preds = _fold_predictions_bias(n_dias=10, n_folds=1)  # celdas con n << min_obs
    factors, report = lead_bias_factors(
        preds, {"corin": "pickup_curve"}, {}, pd.DataFrame(), min_obs=20
    )
    assert factors == {}, "celdas con < min_obs no corrigen a ciegas"
    assert all(c["factor"] == 1.0 for c in report["cells"])


def test_clip_del_factor_a_max_rel():
    """Un bias del +80% (otro régimen, no bias) se clipea a ±30%."""
    preds = _fold_predictions_bias(over_forecast_long=0.80)
    factors, _ = lead_bias_factors(
        preds, {"corin": "pickup_curve"}, {}, pd.DataFrame(),
        min_obs=10, shrinkage=0,  # sin shrinkage para ver el clip crudo
    )
    assert factors[("corin", "60-89")] == pytest.approx(0.70)


def test_sesgo_inestable_entre_folds_no_corrige():
    """El gate de consistencia: sesgo que gira entre folds es régimen, no bias.

    Caso real (VILLAS 30-59): biases por fold +25/−17/−7/+55% — la media
    (+6%) la domina un solo fold. Corregir con eso es corregir ruido.
    """
    rng = np.random.default_rng(9)
    rows = []
    fold_biases = [0.50, -0.20, -0.20, 0.45]  # consistencia 2/4 = 50%
    for fold, bias in enumerate(fold_biases):
        base = pd.Timestamp("2026-01-01") + pd.Timedelta(days=90 * fold)
        dates = pd.date_range(base + pd.Timedelta(days=1), periods=90)
        for d in dates:
            lead = (d - base).days
            actual = 100.0
            forecast = actual * (1.0 + (bias if lead > 30 else 0.0)) + rng.normal(0, 1)
            rows.append({"property": "corin", "fold": fold, "model": "pickup_curve",
                         "stay_date": d, "lead_days": lead,
                         "forecast": forecast, "actual": actual})
    preds = pd.DataFrame(rows)
    factors, report = lead_bias_factors(
        preds, {"corin": "pickup_curve"}, {}, pd.DataFrame(),
        shrinkage=0,  # aislar el efecto del gate
    )
    assert ("corin", "60-89") not in factors or factors[("corin", "60-89")] == 1.0
    cell = next(c for c in report["cells"] if c["lead_band"] == "60-89")
    assert cell["applied"] is False
    assert "inestable" in cell["reason"]


def test_sesgo_consistente_entre_folds_si_corrige():
    """Mismo sesgo en todos los folds (estable) → la corrección SÍ aplica."""
    preds = _fold_predictions_bias(n_folds=4)
    factors, report = lead_bias_factors(
        preds, {"corin": "pickup_curve"}, {}, pd.DataFrame(), shrinkage=0
    )
    cell = next(c for c in report["cells"] if c["lead_band"] == "60-89")
    assert cell["applied"] is True
    assert cell["consistency"] == 1.0
    assert factors[("corin", "60-89")] < 1.0


def test_sin_predicciones_deja_todo_igual():
    factors, report = lead_bias_factors(None, {}, {}, pd.DataFrame())
    assert factors == {}
    assert "sin predicciones" in report["note"]


# ------------------------------------------------------ apply_bias_factors


def test_aplica_factor_por_banda_y_registra_columna():
    preds = _fold_predictions_bias()
    factors, _ = lead_bias_factors(
        preds, {"corin": "pickup_curve"}, {}, pd.DataFrame()
    )
    frame = pd.DataFrame({
        "property": ["corin"] * 3,
        "stay_date": pd.date_range("2026-11-01", periods=3),
        "lead_days": [5, 45, 75],
        "raw_forecast_rn": [100.0, 100.0, 100.0],
    })
    out = apply_bias_factors(frame, factors)
    assert "bias_correction_factor" in out.columns
    # lead 5 (banda 3-6): sin evidencia suficiente → factor 1.0
    assert out.loc[0, "raw_forecast_rn"] == pytest.approx(100.0)
    # leads largos: corregidos a la baja
    assert out.loc[1, "raw_forecast_rn"] < 100.0
    assert out.loc[2, "raw_forecast_rn"] < 100.0
    assert (out["bias_correction_factor"] <= 1.0).all()


def test_residual_bands_con_correccion_reflejan_el_pipeline_publicado():
    """Los residuos de la banda son los del forecast CORREGIDO."""
    preds = _fold_predictions_bias()
    models = {"corin": "pickup_curve"}
    factors, _ = lead_bias_factors(preds, models, {}, pd.DataFrame())

    raw_bands = operative_residual_bands(preds, models, {}, pd.DataFrame())
    corr_bands = operative_residual_bands(
        preds, models, {}, pd.DataFrame(), bias_factors=factors
    )
    key = ("corin", "60-89")
    f = factors[key]
    # residual corregido = forecast·f − actual != forecast − actual
    assert not np.allclose(raw_bands[key], corr_bands[key])
    assert float(np.mean(corr_bands[key])) < float(np.mean(raw_bands[key]))


# ------------------------------------------------- banda conformal (P0-D)


def _coverage_on_calibration(points: np.ndarray, residuals: np.ndarray) -> float:
    """Cobertura de la banda sobre la propia muestra de calibración."""
    df = pd.DataFrame({
        "property": ["corin"] * len(points),
        "lead_days": [45] * len(points),
        "final_forecast_rn": points,
        "capacity": [np.inf] * len(points),
    })
    res = build_intervals(
        df, model_name="x", residuals=pd.Series(residuals),
        point_col="final_forecast_rn",
    )
    f = res.frame
    actual = points - residuals
    return float(np.mean((actual >= f["p10_forecast"]) & (actual <= f["p90_forecast"])))


def test_banda_conformal_cubre_nominal_con_residuos_sesgados():
    """El caso CORIN/FIESTA: residuos con sesgo sistemático (1σ).

    Con cuantiles de-mediaos la banda anclada al punto perdía la masa
    desplazada por el sesgo (~60% de cobertura con sesgo de 1σ); con
    cuantiles completos la banda se desplaza con el sesgo y cubre el 80%
    nominal POR CONSTRUCCIÓN. Sesgo moderado (q10 < 0 < q90) para no
    activar la proyección monótona del contrato p10<=p50<=p90.
    """
    rng = np.random.default_rng(42)
    n = 2000
    residuals = rng.normal(5.0, 5.0, n)  # sobre-pronóstico sistemático +1σ
    points = 50.0 + rng.normal(0, 3, n)
    cov = _coverage_on_calibration(points, residuals)
    assert 0.77 <= cov <= 0.83, f"cobertura conformal {cov:.2f} debe ser ~0.80"


def test_banda_conformal_residuos_sesgados_se_desplaza_con_el_sesgo():
    rng = np.random.default_rng(42)
    n = 500
    residuals = rng.normal(10.0, 5.0, n)
    points = np.full(n, 50.0)
    df = pd.DataFrame({
        "property": ["corin"] * n, "lead_days": [45] * n,
        "final_forecast_rn": points, "capacity": [np.inf] * n,
    })
    res = build_intervals(
        df, model_name="x", residuals=pd.Series(residuals),
        point_col="final_forecast_rn",
    )
    f = res.frame
    # la banda honesta vive DEBAJO del punto (el modelo sobre-pronostica):
    # p90 < p50 = punto — proyectada al contrato monótono, p90 == p50
    assert (f["p10_forecast"] < f["p50_forecast"]).all()
    assert (f["p90_forecast"] <= f["p50_forecast"] + 1e-9).all()
    res.validate_ordering()  # p10 <= p50 <= p90 se mantiene


def test_banda_simétrica_sin_sesgo_equivalente_al_comportamiento_previo():
    """Sin sesgo la banda conformal coincide con la de-mediaada (regresión)."""
    rng = np.random.default_rng(7)
    n = 2000
    residuals = rng.normal(0.0, 8.0, n)
    points = 50.0 + rng.normal(0, 3, n)
    cov = _coverage_on_calibration(points, residuals)
    assert 0.75 <= cov <= 0.85


# ------------------------------------------ histéresis del campeón (P1)


def _fold_metrics_dos_modelos(wmape_a: float, wmape_b: float, prop="corin"):
    rows = []
    for fold in range(3):
        rows.append({"model": "a", "fold": fold, "property": prop,
                     "mape": wmape_a * 100, "wmape": wmape_a, "mae": 5.0,
                     "rmse": 6.0, "bias": 0.01})
        rows.append({"model": "b", "fold": fold, "property": prop,
                     "mape": wmape_b * 100, "wmape": wmape_b, "mae": 5.0,
                     "rmse": 6.0, "bias": 0.01})
    return pd.DataFrame(rows)


def test_incumbente_retenido_sin_ventaja_material():
    """Challenger 1% mejor (dentro del 5%) → el incumbente NO cambia."""
    fm = _fold_metrics_dos_modelos(wmape_a=0.100, wmape_b=0.099)
    sel = select_per_property(
        fm, global_champion="a", incumbent={"corin": "a"}
    )
    assert sel.by_property["corin"] == "a"
    assert "incumbente retenido" in sel.property_fallback["corin"]


def test_incumbente_reemplazado_con_ventaja_material():
    """Challenger 10% mejor → SÍ cambia (la histéresis no es un candado)."""
    fm = _fold_metrics_dos_modelos(wmape_a=0.100, wmape_b=0.090)
    sel = select_per_property(
        fm, global_champion="a", incumbent={"corin": "a"}
    )
    assert sel.by_property["corin"] == "b"


def test_incumbente_ausente_de_candidatos_se_ignora():
    """El incumbente que ya no compite (p. ej. sin folds) no bloquea."""
    fm = _fold_metrics_dos_modelos(wmape_a=0.100, wmape_b=0.099)
    sel = select_per_property(
        fm, global_champion="a", incumbent={"corin": "modelo_retirado"}
    )
    # sin incumbente válido aplica la lógica global: b no supera el margen
    # sobre el campeón global a → retención del global
    assert sel.by_property["corin"] == "a"


def test_sin_incumbente_comportamiento_previo():
    """Sin registro previo la selección es exactamente la de siempre."""
    fm = _fold_metrics_dos_modelos(wmape_a=0.100, wmape_b=0.099)
    sel = select_per_property(fm, global_champion="a")
    assert sel.by_property["corin"] == "a"
    assert "override retenida" in sel.property_fallback["corin"]
