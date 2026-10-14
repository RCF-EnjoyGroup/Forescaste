"""Tests del cambio per-property-intervals.

Las bandas P10-P90 se calibran con los residuos del forecast OPERATIVO
(campeón + blend + perfil DOW) por (propiedad, banda de lead): el pool
del portafolio daba anchuras idénticas para hoteles de 54 y 408
habitaciones. Los folds ya computan las predicciones de todos los
candidatos — `real_fold_metrics` las retiene y el ensamble operativo las
reutiliza sin re-entrenar nada.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.core import operative_residual_bands
from src.forecasting.constraints import build_intervals
from src.forecasting.run_real import real_fold_metrics


def _fold_predictions(prop="corin", n_dias=90, n_folds=2, seed=3):
    """Predicciones sintéticas: dos modelos, un campeón y un reemplazo."""
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(n_folds):
        base = pd.Timestamp("2026-01-01") + pd.Timedelta(days=90 * fold)
        dates = pd.date_range(base + pd.Timedelta(days=1), periods=n_dias)
        for d in dates:
            lead = (d - base).days
            actual = 100 + 10 * np.sin(2 * np.pi * d.dayofweek / 7) + rng.normal(0, 3)
            rows.append({"property": prop, "fold": fold, "model": "lightgbm",
                         "stay_date": d, "lead_days": lead,
                         "forecast": actual + rng.normal(0, 10), "actual": actual})
            rows.append({"property": prop, "fold": fold, "model": "sarimax",
                         "stay_date": d, "lead_days": lead,
                         "forecast": actual + rng.normal(0, 4), "actual": actual})
    return pd.DataFrame(rows)


# ---------------------------------------------------------- retención folds


def test_real_fold_metrics_retiene_predicciones():
    """fold_predictions reproduce por celda lo que las métricas agregan."""
    n = 200
    dem = pd.DataFrame({
        "property": ["corin"] * n,
        "stay_date": pd.date_range("2025-01-01", periods=n),
        "room_nights": 50.0 + np.arange(n) * 0.05,
    })
    pick = pd.DataFrame({
        "property": ["corin"] * n,
        "stay_date": pd.date_range("2025-01-01", periods=n),
        "snapshot_date": pd.date_range("2025-01-01", periods=n) - pd.Timedelta(days=7),
        "lead_days": [7] * n,
        "pickup_rn": [10.0] * n,
    })
    folds, _, preds = real_fold_metrics(dem, pick, pd.Timestamp("2025-06-30"), n_folds=2)
    assert not preds.empty
    assert {"property", "fold", "model", "stay_date", "lead_days", "forecast", "actual"} <= set(preds.columns)
    # una fila por (fold, model, fecha): 2 folds x 7 modelos x 90 dias
    por_celda = preds.groupby(["fold", "model", "stay_date"]).size()
    assert (por_celda == 1).all(), "cada (fold, modelo, fecha) aparece una vez"
    assert preds["actual"].notna().all(), "el real viaja con la predicción"


# ------------------------------------------------------- ensamble operativo


def test_ensamble_operativo_campeon_puro():
    """Sin blend: los residuos son exactamente forecast - actual del campeón."""
    preds = _fold_predictions()
    bands = operative_residual_bands(
        preds, {"corin": "lightgbm"}, {}, pd.DataFrame(),
    )
    assert set(bands), "debe emitir residuos por banda"
    base = preds[(preds["model"] == "lightgbm")]
    esperado = (base["forecast"] - base["actual"]).abs()
    emitido = pd.concat([b.abs() for b in bands.values()])
    # misma multiset de |residuos| (agrupados por banda, sin perdida)
    assert len(emitido) == len(esperado)
    assert sorted(emitido.round(6)) == sorted(esperado.round(6))


def test_ensamble_operativo_aplica_blend_y_perfil():
    """Con blend: leads > split usan el reemplazo; con perfil DOW, se multiplica."""
    preds = _fold_predictions()
    blend = {"corin": {"applied": True, "long_model": "sarimax",
                       "blend_split": 30, "long_pattern": "flat"}}
    bands = operative_residual_bands(
        preds, {"corin": "lightgbm"}, blend, pd.DataFrame(),
    )
    # reconstruir a mano: leads > 30 con sarimax
    from src.evaluation.breakdowns import lead_band
    base = preds[(preds["model"] == "lightgbm")]
    alt = preds[(preds["model"] == "sarimax")].set_index(["fold", "stay_date"])
    op = base.set_index(["fold", "stay_date"]).copy()
    largos = op["lead_days"] > 30
    op.loc[largos, "forecast"] = alt.loc[largos, "forecast"]
    residuos = (op["forecast"] - op["actual"]).dropna()
    emitidos = pd.concat([b for b in bands.values()], ignore_index=True)
    assert len(emitidos) == len(residuos)
    assert sorted(emitidos.round(6)) == sorted(residuos.round(6))


def test_ensamble_operativo_sin_predicciones():
    vacio = operative_residual_bands(None, {}, {}, pd.DataFrame())
    assert vacio == {}
    vacio2 = operative_residual_bands(pd.DataFrame(), {}, {}, pd.DataFrame())
    assert vacio2 == {}


# ------------------------------------------------- build_intervals propiedad


def _frame_intervalos(prop_valores):
    rows = []
    for prop, fc in prop_valores:
        for lead in (1, 45, 75):
            rows.append({"property": prop, "stay_date": pd.Timestamp("2026-07-10"),
                         "lead_days": lead, "final_forecast_rn": fc, "capacity": 500.0})
    return pd.DataFrame(rows)


def test_intervalos_por_propiedad_escalan_a_cada_hotel():
    """La propiedad chica recibe banda chica y la grande banda grande."""
    rng = np.random.default_rng(5)
    chica = pd.Series(rng.normal(0, 2, 200))     # errores de hotel de 54 hab
    grande = pd.Series(rng.normal(0, 40, 200))  # errores de hotel de 408 hab
    bands = {
        ("chica", "30-59"): chica, ("chica", "60-89"): chica,
        ("chica", "1-2"): chica.iloc[:10],
        ("grande", "30-59"): grande, ("grande", "60-89"): grande,
        ("grande", "1-2"): grande.iloc[:10],
    }
    df = _frame_intervalos([("chica", 30.0), ("grande", 300.0)])
    res = build_intervals(
        df, model_name="x", residual_bands=bands, property_col="property",
        point_col="final_forecast_rn",
    )
    f = res.frame
    ancho = f["p90_forecast"] - f["p10_forecast"]
    a_chica = ancho[f["property"] == "chica"].mean()
    a_grande = ancho[f["property"] == "grande"].mean()
    assert a_grande > 3 * a_chica, "la banda de la propiedad grande debe ser mucho más ancha"
    assert a_chica < 15, "la banda chica no hereda la escala del pool"
    assert res.has_intervals


def test_intervalos_escalera_de_fallback():
    """Celda con < min residuos cae al pool de la propiedad, no al global."""
    chica = pd.Series(np.abs(np.random.default_rng(1).normal(0, 2, 200)))
    grande = pd.Series(np.abs(np.random.default_rng(2).normal(0, 40, 200)))
    bands = {
        ("chica", "30-59"): chica, ("chica", "60-89"): chica.iloc[:5],  # < 20
        ("grande", "30-59"): grande, ("grande", "60-89"): grande,
    }
    df = _frame_intervalos([("chica", 30.0), ("grande", 300.0)])
    res = build_intervals(
        df, model_name="x", residual_bands=bands, property_col="property",
        point_col="final_forecast_rn",
    )
    # la celda chica 60-89 hereda el pool de la PROPIEDAD chica (escala chica)
    f = res.frame
    ancho_chica_6089 = (f["p90_forecast"] - f["p10_forecast"])[
        (f["property"] == "chica") & (f["lead_days"] == 75)
    ].iloc[0]
    ancho_grande_6089 = (f["p90_forecast"] - f["p10_forecast"])[
        (f["property"] == "grande") & (f["lead_days"] == 75)
    ].iloc[0]
    assert ancho_chica_6089 < 20, "la celda chica con pocos residuos NO hereda la escala grande"
    assert ancho_grande_6089 > 40
    assert "escalera" in (res.note or "")


def test_intervalos_sin_property_col_comportamiento_actual():
    """Claves de banda planas (sin tuplas): sin property_col, comporta igual que hoy."""
    chica = pd.Series(np.random.default_rng(1).normal(0, 2, 200))
    bands = {"30-59": chica, "60-89": chica}
    df = _frame_intervalos([("chica", 30.0)])
    res = build_intervals(
        df, model_name="x", residual_bands=bands, point_col="final_forecast_rn",
    )
    assert res.has_intervals
    f = res.frame
    assert (f["p10_forecast"] <= f["p50_forecast"]).all()
    assert (f["p50_forecast"] <= f["p90_forecast"]).all()
