"""Tests del cambio openspec 2026-11-10-kpi-volume-truth.

El MAPE global del reporte mezcla propiedades chicas y cold-start
(provisionales por diseño): en real-20261106-1323 dos propiedades con
4.7% de las room nights explican 62.85% del MAPE global. El cambio
PUBLICA el contexto junto al número (el global no se toca).

Fijan:
1. corrida con cold-start → global_excl_cold_start presente, distinto del
   global, con la razón de cada exclusión;
2. corrida sin cold-start → el bloque NO aparece (nada inventado);
3. la nota de auditoría del reporte visual declara la exclusión.
"""

import json

import numpy as np
import pandas as pd

from src.forecasting.runner import run_end_to_end
from src.reporting.visual_report import _fmt_metrics

CUTOFF = pd.Timestamp("2026-01-01")


def _prop(name: str, n_dias: int, start: str, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_dias, freq="D")
    dow = dates.day_of_week
    base = 55 + 12 * (dow >= 5)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_dias) / 365.25)
    return pd.DataFrame(
        {
            "property": [name] * n_dias,
            "stay_date": dates,
            "rooms_sold": np.clip(base + seasonal + rng.normal(0, 5, n_dias), 0, 100).round(),
        }
    )


def _pickup(dem: pd.DataFrame):
    rows = []
    rng = np.random.default_rng(12)
    final = dem.set_index(["property", "stay_date"])["rooms_sold"]
    for (prop, stay), total in final.items():
        if stay.day != 10:
            continue
        for lead in [90, 60, 30, 14, 7, 3, 1]:
            snap = stay - pd.Timedelta(days=lead)
            if snap > CUTOFF:
                continue
            booked = total * (1 - np.exp(-lead / 35)) + rng.normal(0, 1.5)
            rows.append(
                {
                    "propiedad": prop,
                    "fecha": stay,
                    "lead": lead,
                    "reservas": max(booked, 0.0),
                }
            )
    return pd.DataFrame(rows)


def _correr(properties, run_dir):
    dem = pd.concat(properties, ignore_index=True)
    cap = pd.DataFrame(
        {
            "hotel": dem["property"],
            "dia": dem["stay_date"],
            "cuartos": [90] * len(dem),
        }
    )
    pick = _pickup(dem)
    return run_end_to_end(
        demand=dem,
        capacity=cap,
        pickup=pick,
        run_dir=run_dir,
        cutoff=CUTOFF,
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
    )


def _accuracy(run_dir):
    return json.loads((run_dir / "forecast_accuracy_report.json").read_text(encoding="utf-8"))


def test_con_cold_start_publica_el_contexto(tmp_path):
    vieja = _prop("corin", 500, "2025-01-01", seed=11)     # 366d >= 365d
    nueva = _prop("nuevo", 320, "2025-03-01", seed=7)       # 307d < 365d: cold start
    result = _correr([vieja, nueva], tmp_path / "run_cold")
    rep = _accuracy(tmp_path / "run_cold")

    assert "global_excl_cold_start" in rep, "falta el bloque excluyendo cold-start"
    assert "nuevo" in rep["excluded_cold_start"], "la razón de la exclusión debe explicitarse"
    glo, exc = rep["global"], rep["global_excl_cold_start"]
    assert exc["mape"] != glo["mape"] or exc["n_obs"] != glo["n_obs"]
    assert exc["n_obs"] < glo["n_obs"], "el bloque excluido debe tener menos observaciones"
    # el global publicado NO cambia: sigue siendo el global
    assert result.metrics["mape"] == glo["mape"]


def test_sin_cold_start_no_inventa_el_bloque(tmp_path):
    vieja = _prop("corin", 395, "2024-06-01", seed=11)
    _correr([vieja], tmp_path / "run_solo")
    rep = _accuracy(tmp_path / "run_solo")
    assert "global_excl_cold_start" not in rep, "sin cold-start no debe aparecer el bloque"
    assert "excluded_cold_start" not in rep


def test_nota_de_auditoria_declara_la_exclusion():
    html = _fmt_metrics(
        {"mape": 34.2, "wmape": 0.1369, "n_obs": 539},
        metrics_excl_cold_start={"mape": 19.95, "wmape": 0.17, "bias": -0.02},
        excluded_cold_start={"nuevo": "120d < 365d de histórico"},
    )
    assert "Excluyendo cold-start" in html
    assert "19.9%" in html and "nuevo" in html
    # sin exclusión, la nota no aparece
    html_solo = _fmt_metrics({"mape": 34.2})
    assert "Excluyendo cold-start" not in html_solo
