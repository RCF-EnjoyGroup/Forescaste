"""Selección operativa anclada del Excel de vintages — openspec
2026-11-10-monthly-report-operative-models.

El mapa de modelos de cada corte sale del registro entre corridas (la
selección operativa vigente a esa fecha), no de un diccionario fijo;
los cortes previos al motor caen al registro más antiguo y la ausencia
de registro al snapshot histórico — todo declarado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.monthly_vintage_report import (  # noqa: E402
    MODELOS_FALLBACK,
    _modelos_por_corte,
)


def _rec(run_id: str, cutoff: str, ts: str, **modelos: str) -> dict:
    return {
        "run_id": run_id, "mode": "futuro", "cutoff": cutoff,
        "timestamp": ts, "model_by_property": dict(modelos),
    }


def _registro(*records: dict) -> list[dict]:
    return list(records)


def test_anclaje_toma_la_seleccion_vigente_al_corte():
    recs = _registro(
        _rec("f-oct09", "2026-10-06", "2026-10-09T10:00:00+00:00", MARINA="catboost"),
        _rec("f-oct20", "2026-10-19", "2026-10-20T10:00:00+00:00", MARINA="lightgbm"),
        _rec("f-nov05", "2026-11-05", "2026-11-05T10:00:00+00:00", MARINA="sarimax"),
    )
    # corte 15-oct: la última con cutoff <= 15-oct es la del 9-oct
    # (la del 20-oct tiene corte 19-oct: aún no operaba el 15-oct)
    modelos, fuente = _modelos_por_corte(recs, pd.Timestamp("2026-10-15"))
    assert modelos["MARINA"] == "catboost"
    assert "registro:" in fuente and "f-oct09" in fuente
    # corte 25-oct: ya con corte <= es la del 20-oct (lightgbm)
    modelos_25, _ = _modelos_por_corte(recs, pd.Timestamp("2026-10-25"))
    assert modelos_25["MARINA"] == "lightgbm"
    # corte 15-nov: ya existe la del 5-nov
    modelos_n, fuente_n = _modelos_por_corte(recs, pd.Timestamp("2026-11-15"))
    assert modelos_n["MARINA"] == "sarimax" and "f-nov05" in fuente_n


def test_cortes_premotor_caen_al_registro_mas_antiguo_declarado():
    recs = _registro(
        _rec("f-oct09", "2026-10-06", "2026-10-09T10:00:00+00:00", MARINA="catboost"),
        _rec("f-oct20", "2026-10-19", "2026-10-20T10:00:00+00:00", MARINA="lightgbm"),
    )
    modelos, fuente = _modelos_por_corte(recs, pd.Timestamp("2026-05-15"))
    assert modelos["MARINA"] == "catboost", "el corte antiguo se reconstruye con la selección más cercana"
    assert fuente.startswith("registro-mas-antiguo:") and "f-oct09" in fuente


def test_sin_registro_cae_al_fallback_declarado():
    modelos, fuente = _modelos_por_corte([], pd.Timestamp("2026-11-15"))
    assert modelos == MODELOS_FALLBACK
    assert fuente == "fallback-fijo"


def test_registro_sin_futuros_cae_al_fallback():
    recs = [
        _rec("r-real", "2026-11-04", "2026-11-04T10:00:00+00:00", MARINA="lightgbm")
    ]
    recs[0]["mode"] = "real"
    modelos, fuente = _modelos_por_corte(recs, pd.Timestamp("2026-11-15"))
    assert modelos == MODELOS_FALLBACK and fuente == "fallback-fijo"


def test_fallback_reproduce_la_seleccion_del_26_oct():
    assert MODELOS_FALLBACK["MARINA"] == "catboost"
    assert MODELOS_FALLBACK["FIESTA"] == "seasonal_naive"
    # y coincide con el registro futuro más antiguo real → cortes
    # existentes bit a bit (ver test de regeneración)
