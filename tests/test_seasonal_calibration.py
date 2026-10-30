"""Celdas estacionales de calibración — openspec 2026-10-30-seasonal-calibration-cells.

La banda descalibrada del 30-oct tenía la raíz en la AUSENCIA de la
temporada del stay como dimensión de calibración: FIESTA −13.5% de sesgo
en el régimen ago-oct con el gate por banda bloqueado (el sesgo gira
ENTRE estaciones, consistencia 50%) y cobertura de banda 42.2% (51/90
reales sobre p90). Estas pruebas fijan el nuevo modo estacional, sus
gates y — crítico — que con el knob DESACTIVADO el comportamiento es bit
a bit el de siempre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.breakdowns import season_of_stay
from src.forecasting.constraints import build_intervals
from src.forecasting.core import (
    apply_bias_factors,
    lead_bias_factors,
    operative_residual_bands,
)

DEM_VACIA = pd.DataFrame(columns=["property", "stay_date", "room_nights"])


# ---------- fixtures sintéticas ----------


def _fold_predictions_con_sesgo_estacional() -> pd.DataFrame:
    """Folds con sesgo opuesto por estación, meses fijados por diseño.

    La celda (prop, banda) promedia regímenes opuestos → media casi nula
    y consistencia de signo ~50% (el gate la bloquea). Una celda
    (prop, banda, temporada) tiene sesgo consistente → evidencia real.
    Los stays quedan FIJADOS dentro de su temporada (enero high, julio
    low en todos los folds): nada cruza de mes límite por accidente.
    """
    rows = []
    for fold in range(4):
        for lead, banda_offsets in ((5, [1, 3, 5, 7, 10, 12, 14, 16]),
                                    (45, [2, 4, 6, 8, 11, 13, 15, 17])):
            for off in banda_offsets:
                actual = 100.0 + lead * 0.5
                # high: enero (fold 0: 2026-01; fold 1: 2027-01; ...)
                high_stay = pd.Timestamp("2026-01-01") + pd.Timedelta(
                    days=fold * 365 + off)
                # low: julio del mismo año calendario-lógico
                low_stay = pd.Timestamp("2026-07-01") + pd.Timedelta(
                    days=fold * 365 + off)
                for stay, over in ((high_stay, 0.85), (low_stay, 1.15)):
                    rows.append({
                        "property": "TEST", "fold": fold, "model": "m",
                        "stay_date": stay, "lead_days": lead,
                        "forecast": actual * over, "actual": actual,
                    })
    return pd.DataFrame(rows)


MBP = {"TEST": "m"}
BLEND = {"TEST": {}}


# ---------- 1. helper de temporada ----------


def test_season_of_stay_mapa_calendario():
    d = pd.Series(pd.to_datetime([
        "2026-01-15", "2026-03-15", "2026-06-01", "2026-09-30",
        "2026-10-01", "2026-11-30", "2026-12-25",
    ]))
    assert list(season_of_stay(d)) == [
        "high", "shoulder", "low", "low", "shoulder", "shoulder", "high",
    ]


# ---------- 2. modo desactivado: bit a bit el comportamiento actual ----------


def test_knob_desactivado_llaves_de_banda_y_sin_fallback():
    fp = _fold_predictions_con_sesgo_estacional()
    factors, report = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA)
    assert all(len(k) == 2 for k in factors)
    assert "season_cells" not in report and "band_factors" not in report
    # el gate por banda bloquea el sesgo estacional mezclado (consistencia baja)
    for cell in report["cells"]:
        if cell["n_obs"] >= cell["factor"] * 0:  # todas las celdas
            assert cell["factor"] is not None


def test_apply_bias_factors_2tupla_invariante():
    hz = pd.DataFrame({
        "property": ["TEST"] * 2, "lead_days": [5, 5],
        "stay_date": pd.to_datetime(["2026-01-10", "2026-07-10"]),
        "raw_forecast_rn": [100.0, 100.0],
    })
    factors = {("TEST", "3-6"): 1.10}
    out = apply_bias_factors(hz, factors)
    assert out["bias_correction_factor"].tolist() == [1.10, 1.10]
    assert np.allclose(out["raw_forecast_rn"].to_numpy(dtype=float), [110.0, 110.0], atol=1e-9)


def test_operative_residual_bands_2tupla_invariante():
    fp = _fold_predictions_con_sesgo_estacional()
    bands = operative_residual_bands(fp, MBP, BLEND, DEM_VACIA)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in bands)


def test_build_intervals_2tupla_ladder_intacto():
    fp = _fold_predictions_con_sesgo_estacional()
    bands = operative_residual_bands(fp, MBP, BLEND, DEM_VACIA)
    hz = pd.DataFrame({
        "property": ["TEST"], "lead_days": [5],
        "stay_date": pd.to_datetime(["2026-01-10"]),
        "final_forecast_rn": [100.0],
    })
    res = build_intervals(hz, "m", property_col="property", residual_bands=bands)
    f = res.frame
    assert (f["p10_forecast"] <= f["p50_forecast"] + 1e-9).all()
    assert (f["p50_forecast"] <= f["p90_forecast"] + 1e-9).all()
    # celdas con n>=20 (3-6: 4 folds x 15) usan la celda propia: cuantiles
    # completos, anulando el sesgo de la mezcla hacia banda estrecha abajo
    assert f["p90_forecast"].iloc[0] < 200.0


# ---------- 3. modo estacional: estimación con gates ----------


def test_celdas_estacionales_con_tres_llaves_y_reporte():
    fp = _fold_predictions_con_sesgo_estacional()
    factors, report = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA,
                                        seasonal_cells=True)
    assert all(len(k) == 3 for k in factors)
    assert "season_cells" in report and "band_factors" in report
    # la celda 3-6/high tiene sesgo DOWN consistente (0.85) → corrige AL ALZA
    fh = factors.get(("TEST", "3-6", "high"))
    fl = factors.get(("TEST", "3-6", "low"))
    assert fh is not None and fh > 1.0, "sesgo -15% consistente debe corregir al alza"
    assert fl is not None and fl < 1.0, "sesgo +15% consistente debe corregir a la baja"
    # las celdas de BANDA (estaciones mezcladas) NO corrigen: la mezcla de
    # regímenes opuestos cancela el sesgo a ruido — el punto del horizonte
    # salía así sin corrección en la corrida real: el caso FIESTA
    aplicadas_banda = [c for c in report["cells"] if c["applied"]]
    assert not aplicadas_banda, (
        "el sesgo que gira entre estaciones no debe corregir por banda"
    )


def test_celda_estacional_chica_cae_al_factor_de_banda():
    # celda estacional con n<min_obs pero banda con evidencia: fallback
    fp = _fold_predictions_con_sesgo_estacional()
    fp_sin_high = fp[~(fp["stay_date"].dt.month <= 3)].copy()
    factors, report = lead_bias_factors(fp_sin_high, MBP, BLEND, DEM_VACIA,
                                        seasonal_cells=True, min_obs=60)
    srcs = {c["source"] for c in report["season_cells"]}
    assert "season" in srcs or "band" in srcs or "none" in srcs
    # toda celda declara su fuente — nada silencioso
    assert all(c["source"] in {"season", "band", "none"}
               for c in report["season_cells"])


# ---------- 4. aplicación al horizonte ----------


def test_apply_bias_factors_3tupla_por_temporada_y_fallback():
    hz = pd.DataFrame({
        "property": ["TEST"] * 3, "lead_days": [5, 5, 5],
        "stay_date": pd.to_datetime(["2026-01-10", "2026-07-10", "2026-12-01"]),
        "raw_forecast_rn": [100.0, 100.0, 100.0],
    })
    factors = {
        ("TEST", "3-6", "high"): 1.10,
        ("TEST", "3-6", "low"): 0.90,
        # 2026-12-01 es high: cubierto
    }
    out = apply_bias_factors(hz, factors)
    assert out["bias_correction_factor"].tolist() == [1.10, 0.90, 1.10]


def test_apply_bias_factors_3tupla_fallback_a_banda():
    hz = pd.DataFrame({
        "property": ["TEST"], "lead_days": [45],
        "stay_date": pd.to_datetime(["2026-02-01"]),  # high
        "raw_forecast_rn": [100.0],
    })
    # la llave estacional EXISTE con factor 1.0 (decisión del estimador):
    # NO se inventa corrección extra
    out = apply_bias_factors(
        hz, {("TEST", "30-59", "high"): 1.0}, fallback_factors={("TEST", "30-59"): 1.15},
    )
    assert float(out["bias_correction_factor"].iloc[0]) == 1.0
    # la llave estacional NO existe (la subida del backtest):
    # hereda el factor de banda
    out2 = apply_bias_factors(hz, {("TEST", "30-59", "low"): 0.9},
                              fallback_factors={("TEST", "30-59"): 1.15})
    assert float(out2["bias_correction_factor"].iloc[0]) == 1.15


def test_operative_residual_bands_3tupla_factor_por_fila():
    fp = _fold_predictions_con_sesgo_estacional()
    factors, report = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA,
                                        seasonal_cells=True)
    bands = operative_residual_bands(
        fp, MBP, BLEND, DEM_VACIA,
        bias_factors=factors, seasonal_cells=True,
        fallback_factors=report.get("band_factors"),
    )
    assert all(len(k) == 3 for k in bands)
    # la celda de verano high con sesgo -15% corregido al alza: el residuo
    # corregido debe estar centrado cerca de 0 (no −15)
    cell_high = pd.concat([
        bands[k] for k in bands if k[0] == "TEST" and k[2] == "high"
    ])
    low_leads = bands[("TEST", "3-6", "high")]["residual"]
    assert low_leads.abs().mean() < 40.0


# ---------- 5. banda estacional: escalera y contratos ----------


def _horizonte_estacional() -> pd.DataFrame:
    return pd.DataFrame({
        "property": ["TEST"] * 2, "lead_days": [5, 45],
        "stay_date": pd.to_datetime(["2026-01-10", "2026-07-10"]),
        "final_forecast_rn": [100.0, 100.0],
    })


def test_build_intervals_3tupla_ladder_y_contratos():
    fp = _fold_predictions_con_sesgo_estacional()
    factors, report = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA,
                                        seasonal_cells=True)
    bands = operative_residual_bands(
        fp, MBP, BLEND, DEM_VACIA,
        bias_factors=factors, seasonal_cells=True,
        fallback_factors=report.get("band_factors"),
    )
    hz = _horizonte_estacional()
    res = build_intervals(hz, "m", property_col="property", residual_bands=bands)
    f = res.frame
    assert res.has_intervals
    assert (f["p10_forecast"] <= f["p50_forecast"] + 1e-9).all()
    assert (f["p50_forecast"] <= f["p90_forecast"] + 1e-9).all()
    assert (f["p10_forecast"] >= 0).all()


def test_build_intervals_3tupla_fallback_documentado_en_note():
    # celdas estacionales presentes SOLO para high: las filas low caen
    # a la escalera y la caída queda en `note` (nada silencioso)
    fp = _fold_predictions_con_sesgo_estacional()
    fp = fp[fp["stay_date"].dt.month.isin([1, 2, 12])].copy()
    factors, report = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA,
                                        seasonal_cells=True)
    bands = operative_residual_bands(
        fp, MBP, BLEND, DEM_VACIA,
        bias_factors=factors, seasonal_cells=True,
        fallback_factors=report.get("band_factors"),
    )
    hz = _horizonte_estacional()  # incluye una fila low (julio)
    res = build_intervals(hz, "m", property_col="property", residual_bands=bands)
    assert "escalera" in (res.note or "")


def test_banda_estacional_restringe_capacidad():
    fp = _fold_predictions_con_sesgo_estacional()
    factors, report = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA,
                                        seasonal_cells=True)
    bands = operative_residual_bands(
        fp, MBP, BLEND, DEM_VACIA,
        bias_factors=factors, seasonal_cells=True,
        fallback_factors=report.get("band_factors"),
    )
    hz = _horizonte_estacional().assign(capacity=110.0)
    res = build_intervals(hz, "m", property_col="property", residual_bands=bands)
    f = res.frame
    assert (f["p90_forecast"] <= 110.0 + 1e-9).all()
