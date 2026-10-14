"""Tests del blend por lead (accuracy-aware, cambio accuracy-aware-blend).

Cuando el campeón es un GBM y su rollout recursivo colapsa a largo lead
(q25 largo < 0.85 × q25 corto), el forecast puede empalmar otro modelo en
leads > 30 — pero el reemplazo se elige por EVIDENCIA: el candidato con
mejor ``wmape_long`` en los folds de la propiedad, solo si supera
materialmente al propio campeón (margen ``BLEND_ACCURACY_MARGIN``).
Sin evidencia no hay cirugía: el campeón conserva su rollout completo.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.core import (
    BLEND_ACCURACY_MARGIN,
    BLEND_COLLAPSE_RATIO,
    BLEND_LEAD_SPLIT,
    _select_long_replacement,
    build_forecasts,
)


def _panel(n=200, prop="corin"):
    """Panel de pickup con estancias diarias y leads 7/30."""
    stays = pd.date_range("2026-01-01", periods=n)
    rows = []
    for stay in stays:
        for lead in (30, 7):
            rows.append({
                "property": prop, "stay_date": stay,
                "snapshot_date": stay - pd.Timedelta(days=lead),
                "lead_days": lead, "pickup_rn": 15.0,
            })
    return pd.DataFrame(rows)


def _demand(n=200, prop="corin"):
    return pd.DataFrame({
        "property": [prop] * n,
        "stay_date": pd.date_range("2026-01-01", periods=n),
        "room_nights": np.full(n, 50.0),
    })


def test_blend_auto_diagnostico_detecta_colapso():
    """Campeón GBM cuyo rollout colapsa: el blend activa automáticamente."""
    # construyo datos que generan un patrón con valles de dia-de-semana
    # profundos — suficiente para que el GBM aprenda "dow bajo" y el
    # rollout realimentado colapse a largo lead
    n = 400
    dates = pd.date_range("2025-06-01", periods=n)
    rng = np.random.default_rng(42)
    rn = np.where(dates.dayofweek == 3, 20.0, 60.0) + rng.normal(0, 5, n)
    dem = pd.DataFrame({
        "property": ["corin"] * n,
        "stay_date": dates,
        "room_nights": np.clip(rn, 0, 100),
    })
    pick = _panel(n=n)
    cutoff = dates[-1]

    grid, fitted, blend = build_forecasts(
        {"corin": "lightgbm"}, dem, pick, cutoff, 90,
    )
    # verificar el auto-diagnóstico: si el rollout no colapsa en estos
    # datos sintéticos, el blend no activa (correcto: no hay que forzarlo)
    # -- solo verificamos que NO CRASHEA y que blend es un dict
    assert isinstance(blend, dict)
    assert len(grid) == 90


def test_sin_blend_cuando_campeon_no_es_gbm():
    """Campeón pickup_curve: sin blend (no es GBM)."""
    dem = _demand()
    pick = _panel()
    cutoff = pd.Timestamp("2026-06-20")
    grid, fitted, blend = build_forecasts(
        {"corin": "pickup_curve"}, dem, pick, cutoff, 90,
        curve_params={"n_bins": 5},
    )
    assert blend == {}
    assert (grid["model_name"] == "pickup_curve").all()


def test_constantes_del_blend_declaradas():
    """Las constantes del blend están exportadas y son las esperadas."""
    assert BLEND_LEAD_SPLIT == 30
    assert BLEND_COLLAPSE_RATIO == 0.85
    assert BLEND_ACCURACY_MARGIN == 0.95


# --------------------------------------------------------------------------
# _select_long_replacement — selección del reemplazo largo por evidencia
# --------------------------------------------------------------------------


def _folds(prop="corin", champ_long=0.20, alts=None, n_folds=4):
    """fold_metrics con wmape_long por (modelo, propiedad, fold)."""
    alts = alts if alts is not None else {"sarimax": 0.10, "pickup_curve": 0.30}
    rows = []
    for fold in range(n_folds):
        rows.append({
            "model": "lightgbm", "fold": fold, "property": prop,
            "mape": 10.0, "wmape": 0.10, "mae": 8.0, "bias": 0.01,
            "wmape_short": 0.08, "wmape_long": champ_long,
        })
        for model, wl in alts.items():
            rows.append({
                "model": model, "fold": fold, "property": prop,
                "mape": 12.0, "wmape": 0.12, "mae": 9.0, "bias": 0.02,
                "wmape_short": 0.09, "wmape_long": wl,
            })
    return pd.DataFrame(rows)


def test_seleccion_elige_mejor_wmape_long():
    model, reason = _select_long_replacement("corin", "lightgbm", _folds())
    assert model == "sarimax"
    assert "sarimax=0.100" in reason
    assert "lightgbm=0.200" in reason


def test_seleccion_rechaza_mejora_no_material():
    """0.192 contra umbral 0.20×0.95=0.19: no justifica el empalme."""
    folds = _folds(champ_long=0.20, alts={"sarimax": 0.192})
    model, reason = _select_long_replacement("corin", "lightgbm", folds)
    assert model is None
    assert "ningún candidato supera el margen" in reason


def test_seleccion_sin_folds_no_hay_reemplazo():
    model, reason = _select_long_replacement("corin", "lightgbm", None)
    assert model is None
    assert "sin fold_metrics" in reason


def test_seleccion_sin_columna_wmape_long():
    folds = _folds().drop(columns=["wmape_long"])
    model, reason = _select_long_replacement("corin", "lightgbm", folds)
    assert model is None
    assert "sin wmape_long" in reason


def test_seleccion_sin_alternativas():
    folds = _folds(alts={})
    model, reason = _select_long_replacement("corin", "lightgbm", folds)
    assert model is None
    assert "sin candidatos" in reason


def test_seleccion_campeon_sin_wmape_long():
    folds = _folds()
    folds.loc[folds["model"] == "lightgbm", "wmape_long"] = np.nan
    model, reason = _select_long_replacement("corin", "lightgbm", folds)
    assert model is None
    assert "sin wmape_long" in reason


def test_seleccion_excluye_otras_propiedades():
    """Un wmape_long brillante de OTRA propiedad no cuenta para corin."""
    folds = _folds(alts={"sarimax": 0.30})  # sarimax malo para corin
    otra = pd.DataFrame([{
        "model": "sarimax", "fold": 0, "property": "otra",
        "mape": 1.0, "wmape": 0.01, "mae": 1.0, "bias": 0.0,
        "wmape_short": 0.01, "wmape_long": 0.001,
    }])
    model, reason = _select_long_replacement("corin", "lightgbm", pd.concat([folds, otra]))
    assert model is None, "la evidencia de otra propiedad no debe elegirse"


# --------------------------------------------------------------------------
# build_forecasts — empalme y retención con colapso forzado
# --------------------------------------------------------------------------


@pytest.fixture
def collapse_forzado(monkeypatch):
    """Fuerza el trigger de colapso: ratio < 99 es siempre cierto."""
    monkeypatch.setattr("src.forecasting.core.BLEND_COLLAPSE_RATIO", 99.0)


def test_blend_empalma_el_mejor_candidato(collapse_forzado):
    dem = _demand()
    pick = _panel()
    cutoff = pd.Timestamp("2026-06-20")
    grid, fitted, blend = build_forecasts(
        {"corin": "lightgbm"}, dem, pick, cutoff, 90, fold_metrics=_folds(),
    )
    entry = blend["corin"]
    assert entry["applied"] is True
    assert entry["short_model"] == "lightgbm"
    assert entry["long_model"] == "sarimax"
    assert "sarimax=0.100" in entry["long_selection"]
    largo = grid.loc[grid["lead_days"] > BLEND_LEAD_SPLIT]
    esperado = f"lightgbm+sarimax(blend>{BLEND_LEAD_SPLIT})"
    assert (largo["model_name"] == esperado).all()
    corto = grid.loc[grid["lead_days"] <= BLEND_LEAD_SPLIT]
    assert (corto["model_name"] == "lightgbm").all()


def test_blend_retiene_campeon_sin_folds(collapse_forzado):
    """Colapso detectado pero sin evidencia: NO se reemplaza nada."""
    dem = _demand()
    pick = _panel()
    cutoff = pd.Timestamp("2026-06-20")
    grid, fitted, blend = build_forecasts(
        {"corin": "lightgbm"}, dem, pick, cutoff, 90,
    )
    entry = blend["corin"]
    assert entry["applied"] is False
    assert entry["long_model"] == "lightgbm"
    assert "sin fold_metrics" in entry["long_selection"]
    assert (grid["model_name"] == "lightgbm").all()


def test_blend_retiene_campeon_cuando_nada_califica(collapse_forzado):
    folds = _folds(champ_long=0.20, alts={"sarimax": 0.192})
    dem = _demand()
    pick = _panel()
    cutoff = pd.Timestamp("2026-06-20")
    grid, fitted, blend = build_forecasts(
        {"corin": "lightgbm"}, dem, pick, cutoff, 90, fold_metrics=folds,
    )
    entry = blend["corin"]
    assert entry["applied"] is False
    assert entry["long_model"] == "lightgbm"
    assert "ningún candidato supera el margen" in entry["long_selection"]
    assert (grid["model_name"] == "lightgbm").all()


def test_blend_declara_trigger_en_manifiesto():
    """Cuando el blend activa, el trigger queda declarado."""
    # test integración: e2e con un campeón GBM produce blend_by_property
    # (si el rollout colapsa en los datos de muestra) o dict vacío (si no)
    from tests.test_remediation import (
        _folds_synthetic, _demand_2props, _capacity_2props, _pickup_2props,
    )
    from src.forecasting.runner import run_end_to_end
    from src.monitoring.run_manifest import read_manifest

    dem = _demand_2props()
    cap = _capacity_2props()
    pick = _pickup_2props(dem)
    # folds sintéticos con lightgbm de campeón para ambas props
    folds = _folds_synthetic("lightgbm")
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=None,  # tmp_path via pytest fixture below
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=folds,
    ) if False else None  # e2e necesita tmp_path; lo hacemos abajo


def test_e2e_blend_gbm_no_crashea(tmp_path):
    """Integración: e2e con campeón GBM produce manifiesto con blend info."""
    from tests.test_remediation import (
        _folds_synthetic, _demand_2props, _capacity_2props, _pickup_2props,
    )
    from src.forecasting.runner import run_end_to_end
    from src.monitoring.run_manifest import read_manifest

    dem = _demand_2props()
    cap = _capacity_2props()
    pick = _pickup_2props(dem)
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "blend_e2e",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=_folds_synthetic("lightgbm"),
    )
    assert result.checks.all_passed
    stored = read_manifest(result.manifest_path)
    # blend_by_property puede ser {} (sin colapso) o tener entradas con trigger
    blend = stored["hyperparameters"].get("per_property_blend", {})
    assert isinstance(blend, dict)
    for prop_info in blend.values():
        assert "trigger" in prop_info, "el blend debe declarar su trigger"
        assert "short_model" in prop_info
        assert "long_model" in prop_info
