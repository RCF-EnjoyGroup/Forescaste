"""Tests del cambio long-horizon-folds.

Los folds de selección cubren el horizonte OPERATIVO completo
(``benchmark.fold_horizon_days``, 90) y tesian el año anterior al corte; la
corrección HAC de Diebold-Mariano usa el largo real de la ventana de fold
derivado del panel de errores (antes un fijo de 30, ciego al régimen largo).
"""

import numpy as np
import pandas as pd

import src.reporting.benchmark as bench_mod
from src.config import load_config
from src.evaluation.selection import select_champion
from src.forecasting.runner import _dm_horizon_from_panel
from src.forecasting.run_real import (
    FOLD_HORIZON,
    FOLD_SPACING,
    fold_cutoffs_for,
    real_fold_metrics,
)
from src.reporting.benchmark import run_benchmark


# --------------------------------------------------------------------------
# 1. configuración
# --------------------------------------------------------------------------


def test_config_benchmark_es_el_horizonte_operativo():
    cfg = load_config()
    assert cfg.benchmark["fold_horizon_days"] == FOLD_HORIZON == 90
    assert cfg.benchmark["n_folds"] == 4


def test_config_benchmark_defaults_sin_seccion(tmp_path):
    yaml = tmp_path / "config.yaml"
    yaml.write_text("random_seed: 1\nforecast:\n  horizon_days: 90\n", encoding="utf-8")
    cfg = load_config(yaml)
    assert cfg.benchmark == {"fold_horizon_days": 90, "n_folds": 4}


# --------------------------------------------------------------------------
# 2. tiling de las ventanas de fold
# --------------------------------------------------------------------------


def test_folds_tesian_el_anio_previo_sin_solapes():
    """4 ventanas de 90d contiguas: 360d de cobertura y terminan en el corte."""
    cutoff = pd.Timestamp("2026-07-08")
    cuts = fold_cutoffs_for(cutoff, n_folds=4)
    assert len(cuts) == 4
    windows = [
        (c + pd.Timedelta(days=1), c + pd.Timedelta(days=FOLD_HORIZON)) for c in cuts
    ]
    # contiguas, sin solape entre folds
    for (_, end_prev), (start_next, _) in zip(windows, windows[1:]):
        assert start_next == end_prev + pd.Timedelta(days=1)
    # el último fold termina en el corte → NO solapa la ventana de reporte
    # (cutoff, cutoff + horizonte]
    assert windows[-1][1] == cutoff
    # cobertura estacional: tesian 360 días (≥ 10 meses del año)
    span = (windows[-1][1] - windows[0][0]).days + 1
    assert span == 4 * FOLD_SPACING == 360
    assert span >= 300, "los folds deben cubrir ≥ 10 meses del año"


def test_real_fold_metrics_90_fechas_por_fold_en_el_panel():
    """El panel de errores trae el horizonte COMPLETO de cada fold (90 pasos)."""
    from src.data.schema import SourceKind, detect_schema
    from src.config import load_config as _load
    from src.forecasting.core import _normalize_demand, _normalize_pickup
    from tests.test_remediation import _demand_2props, _pickup_2props

    cfg = _load()
    # histórico lo largo para que las 4 ventanas de 90d caigan dentro del dato
    dem_raw = _demand_2props(n=1000, start="2023-01-01")
    pick_raw = _pickup_2props(dem_raw)
    dem = _normalize_demand(dem_raw, detect_schema(dem_raw, SourceKind.DEMAND, cfg.column_hints))
    pick = _normalize_pickup(pick_raw, detect_schema(pick_raw, SourceKind.PICKUP, cfg.column_hints))

    folds, panel, _ = real_fold_metrics(dem, pick, pd.Timestamp("2025-06-30"))
    per_fold = panel.groupby("fold")["stay_date"].nunique()
    assert not per_fold.empty
    # cada fold evaluado cubre el horizonte operativo completo
    assert (per_fold == FOLD_HORIZON).all(), per_fold.to_dict()
    # y el DM usa ese largo como corrección HAC
    assert _dm_horizon_from_panel(panel) == FOLD_HORIZON


# --------------------------------------------------------------------------
# 3. dm_horizon derivado del panel de errores
# --------------------------------------------------------------------------


def _panel(steps_by_fold: dict[int, int]) -> pd.DataFrame:
    rows = []
    for fold, steps in steps_by_fold.items():
        for d in pd.date_range("2026-01-01", periods=steps, freq="D"):
            rows.append(
                {"model": "a", "fold": fold, "stay_date": d, "actual": 50.0, "forecast": 50.0}
            )
    return pd.DataFrame(rows)


def test_dm_horizon_es_la_mediana_de_las_ventanas():
    assert _dm_horizon_from_panel(_panel({0: 90, 1: 90, 2: 90, 3: 90})) == 90
    # un fold recortado (histórico insuficiente) no arruina la mediana
    assert _dm_horizon_from_panel(_panel({0: 90, 1: 30})) == 60


def test_dm_horizon_sin_panel_evaluable_es_30():
    assert _dm_horizon_from_panel(None) == 30
    assert _dm_horizon_from_panel(pd.DataFrame()) == 30
    assert _dm_horizon_from_panel(pd.DataFrame({"model": [], "stay_date": []})) == 30


def test_run_benchmark_threadea_dm_horizon_a_la_seleccion(monkeypatch):
    real = bench_mod.select_champion
    seen: dict = {}

    def spy(fold_metrics, **kw):
        seen.update(kw)
        return real(fold_metrics, **kw)

    monkeypatch.setattr(bench_mod, "select_champion", spy)

    folds = pd.DataFrame(
        {
            "model": ["a", "b", "a", "b"],
            "fold": [0, 0, 1, 1],
            "mape": [5.0, 6.0, 5.0, 6.0],
            "wmape": [0.10, 0.12, 0.10, 0.12],
            "mae": [3.0, 3.6, 3.0, 3.6],
            "rmse": [4.0, 4.8, 4.0, 4.8],
            "bias": [0.1, 0.1, 0.1, 0.1],
        }
    )
    run_benchmark(folds, dm_horizon=90)
    assert seen["dm_horizon"] == 90
    # default conservador si no se pasa
    run_benchmark(folds)
    assert seen["dm_horizon"] == 30
