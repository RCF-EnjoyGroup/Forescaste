"""Tests del cambio dow-profile-long-leads.

Los reemplazos planos del blend (stats) reciben el patrón DOW de la
propiedad sobre su nivel: el nivel viene de la evidencia de folds, el
patrón de la historia reciente as-of. Guardas ante historia insuficiente
y dow ausente.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.core import (
    BLEND_LEAD_SPLIT,
    DOW_MIN_OBS_PER_DOW,
    DOW_PROFILE_DAYS,
    _dow_profile,
    build_forecasts,
    fit_candidate,
    predict_candidate,
)


def _panel(n=200, prop="corin"):
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


def _demand_patron(n=200, prop="corin"):
    """Jueves valle (20) vs resto (60) + ruido leve."""
    dates = pd.date_range("2026-01-01", periods=n)
    rng = np.random.default_rng(7)
    rn = np.where(dates.dayofweek == 3, 20.0, 60.0) + rng.normal(0, 2, n)
    return pd.DataFrame({
        "property": [prop] * n,
        "stay_date": dates,
        "room_nights": np.clip(rn, 0, 100),
    })


def _folds_stats_best(prop="corin", n_folds=4):
    """lightgbm campeón con wmape_long 0.20; sarimax (stats) mejor largo 0.10."""
    rows = []
    for fold in range(n_folds):
        rows.append({
            "model": "lightgbm", "fold": fold, "property": prop,
            "mape": 10.0, "wmape": 0.10, "mae": 8.0, "bias": 0.01,
            "wmape_short": 0.08, "wmape_long": 0.20,
        })
        for model, wl in (("sarimax", 0.10), ("pickup_curve", 0.30), ("ets", 0.25)):
            rows.append({
                "model": model, "fold": fold, "property": prop,
                "mape": 12.0, "wmape": 0.12, "mae": 9.0, "bias": 0.02,
                "wmape_short": 0.09, "wmape_long": wl,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def collapse_forzado(monkeypatch):
    monkeypatch.setattr("src.forecasting.core.BLEND_COLLAPSE_RATIO", 99.0)


# ---------------------------------------------------------------- perfil


def test_perfil_dow_normalizado_y_con_patron():
    dem = _demand_patron()
    cutoff = dem["stay_date"].max()
    profile = _dow_profile(dem, cutoff)
    assert profile is not None
    # normalizado: media de los 7 factores = 1
    assert profile.mean() == pytest.approx(1.0, abs=1e-9)
    # jueves valle en la historia -> factor < 1; resto > 1
    assert profile.loc[3] < 0.8, "el jueves valle debe tener factor bajo"
    assert profile.drop(index=3).min() > 1.0
    assert set(profile.index) == set(range(7))


def test_perfil_dow_solo_historia_asof():
    """El perfil no puede leer reales posteriores al corte."""
    dem = _demand_patron()
    cutoff = dem["stay_date"].max() - pd.Timedelta(days=90)
    # contaminar el futuro (posteriores al corte) con un patron distinto
    futuro = dem["stay_date"] > cutoff
    dem.loc[futuro, "room_nights"] = np.where(
        dem.loc[futuro, "stay_date"].dt.dayofweek == 2, 5.0, 90.0
    )
    profile = _dow_profile(dem, cutoff)
    assert profile is not None
    # miercoles NO contaminado por el futuro (factor normal, no 5/90)
    assert 0.7 < profile.loc[2] < 1.3


def test_perfil_dow_guardas():
    dem = _demand_patron()
    cutoff = dem["stay_date"].max()
    # historia insuficiente
    assert _dow_profile(dem.iloc[:10], cutoff) is None
    assert _dow_profile(None, cutoff) is None
    # dow ausente: 20 dias con un solo dow faltante no pasa la guarda
    fechas = pd.date_range("2026-01-05", periods=6, freq="W-MON")  # solo lunes
    dem_lunes = pd.DataFrame({
        "property": ["corin"] * 6,
        "stay_date": fechas,
        "room_nights": [50.0] * 6,
    })
    assert _dow_profile(dem_lunes, fechas[-1]) is None
    # constantes exportadas
    assert DOW_PROFILE_DAYS == 84
    assert DOW_MIN_OBS_PER_DOW == 3


# ------------------------------------------------- empalme con patron


def test_blend_empalma_stats_con_perfil_dow(collapse_forzado):
    """sarimax plano largo → el segmento empalmado lleva el patron DOW."""
    dem = _demand_patron()
    pick = _panel()
    cutoff = dem["stay_date"].max()
    grid, fitted, blend = build_forecasts(
        {"corin": "lightgbm"}, dem, pick, cutoff, 90,
        fold_metrics=_folds_stats_best(),
    )
    entry = blend["corin"]
    assert entry["applied"] is True
    assert entry["long_model"] == "sarimax"
    assert entry["long_pattern"] == f"dow_profile({DOW_PROFILE_DAYS}d)"

    # reconstruir la expectativa: sarimax largo x perfil, a mano
    fitted_sx = fit_candidate("sarimax", dem, pick.loc[pick["property"] == "corin"], cutoff, 90)
    pred_sx = predict_candidate(fitted_sx, pick.loc[pick["property"] == "corin"], pd.date_range(cutoff + pd.Timedelta(days=1), periods=90))
    pred_sx["lead_days"] = (pred_sx["stay_date"] - cutoff).dt.days
    profile = _dow_profile(dem, cutoff)
    esperado = pred_sx.set_index("stay_date").copy()
    esperado["factor"] = esperado.index.dayofweek
    esperado["factor"] = esperado["factor"].map(profile)
    esperado["raw"] = esperado["raw_forecast_rn"] * esperado["factor"]

    g = grid.set_index("stay_date")
    largo = g[g["lead_days"] > BLEND_LEAD_SPLIT]
    exp_largo = esperado[esperado["lead_days"] > BLEND_LEAD_SPLIT]
    comparado = largo.join(exp_largo["raw"], rsuffix="_esp")
    assert np.allclose(
        comparado["raw_forecast_rn"], comparado["raw"], atol=1e-6
    ), "el segmento largo debe ser sarimax x perfil"

    # el patron es visible: los jueves del horizonte quedan por debajo del resto
    jue = largo[largo.index.dayofweek == 3]["raw_forecast_rn"].mean()
    resto = largo[largo.index.dayofweek != 3]["raw_forecast_rn"].mean()
    assert jue < resto, "el jueves valle de la historia debe verse en el horizonte"


def test_blend_stats_sin_perfil_cuando_guarda_falla(collapse_forzado, monkeypatch):
    """Guarda del perfil rota → segmento plano, declarado en el manifiesto."""
    dem = _demand_patron()
    pick = _panel()
    cutoff = dem["stay_date"].max()
    monkeypatch.setattr("src.forecasting.core._dow_profile", lambda d, c: None)
    grid, fitted, blend = build_forecasts(
        {"corin": "lightgbm"}, dem, pick, cutoff, 90,
        fold_metrics=_folds_stats_best(),
    )
    assert blend["corin"]["applied"] is True
    assert blend["corin"]["long_model"] == "sarimax"
    assert blend["corin"]["long_pattern"] == "flat"
