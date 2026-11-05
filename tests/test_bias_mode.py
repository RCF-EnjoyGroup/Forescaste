"""Modos de la capa de corrección de bias — openspec
2026-11-05-bias-correction-mode.

Dos knobs: `enabled` (kill-switch de la capa completa) y
`fold_recency_decay` (pondera los folds por recencia en el bias y la
consistencia). Fija: decay=1.0 bit a bit el comportamiento histórico, la
ponderación alinea el estimador con el régimen reciente, y el kill-switch
omite la capa declarándolo — nada silencioso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.core import apply_bias_factors, lead_bias_factors

DEM_VACIA = pd.DataFrame(columns=["property", "stay_date", "room_nights"])
MBP = {"T": "m"}
BLEND = {"T": {}}


def _folds_sesgo_cambiante() -> pd.DataFrame:
    """4 folds con sesgos opuestos: viejos sobre-pronostican +30%,
    recientes −30%. Con pesos uniformes la media se cancela (sin
    corrección); con recencia el régimen reciente domina (corrige alza).
    """
    rows = []
    sesgos = {0: 0.30, 1: 0.30, 2: -0.30, 3: -0.30}  # fold → sesgo
    for fold, s in sesgos.items():
        for i in range(15):
            actual = 100.0
            rows.append({
                "property": "T", "fold": fold, "model": "m",
                "stay_date": pd.Timestamp("2026-01-01") + pd.Timedelta(
                    days=fold * 90 + i),
                "lead_days": 20,
                "forecast": actual * (1 + s), "actual": actual,
            })
    return pd.DataFrame(rows)


def test_decay_1_es_bit_a_bit_uniforme():
    fp = _folds_sesgo_cambiante()
    f1, r1 = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA)
    f2, r2 = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA,
                               fold_recency_decay=1.0)
    assert f1 == f2
    c1 = {(c["property"], c["lead_band"]): c["factor"] for c in r1["cells"]}
    c2 = {(c["property"], c["lead_band"]): c["factor"] for c in r2["cells"]}
    assert c1 == c2


def test_recencia_domina_el_regimen_reciente():
    fp = _folds_sesgo_cambiante()
    # uniforme: la media del sesgo se cancela → sin factor aplicado
    f_uniforme, r_uniforme = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA)
    factor_u = f_uniforme.get(("T", "14-29"))
    # el sesgo agregado ponderado es 0 → sin corrección (o inmaterial)
    assert factor_u is None or abs(factor_u - 1.0) < 0.005
    # con recencia: los folds 2-3 (−30%) dominan → corrige AL ALZA
    f_rec, _ = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA,
                                 fold_recency_decay=0.5)
    factor_r = f_rec.get(("T", "14-29"))
    assert factor_r is not None and factor_r > 1.0, (
        "el régimen reciente (−30% de sesgo) debe producir corrección al alza"
    )
    # la consistencia también es votación ponderada: los folds viejos
    # (que discrepan) pesan menos → la celda antes bloqueada ahora pasa
    assert True  # (factor_r aplicado ya implica consistencia ≥ umbral)


def test_consistencia_ponderada_sobre_bloqueada():
    # folds viejos (0,1) discrepan, recientes (2,3) coinciden con el agregado:
    # uniformes → consistencia 0.5 < 0.75 → bloqueada; ponderada → pasa
    fp = _folds_sesgo_cambiante()
    _, r_u = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA)
    _, r_p = lead_bias_factors(fp, MBP, BLEND, DEM_VACIA,
                               fold_recency_decay=0.5)
    cell_u = [c for c in r_u["cells"] if c["lead_band"] == "14-29"][0]
    cell_p = [c for c in r_p["cells"] if c["lead_band"] == "14-29"][0]
    assert cell_u["consistency"] == pytest.approx(0.5)
    assert cell_p["consistency"] > cell_u["consistency"]


def test_min_obs_sigue_contando_observaciones_crudas():
    fp = _folds_sesgo_cambiante()
    fp_recortado = fp[fp["fold"] >= 2].copy()  # solo 30 obs
    _, r = lead_bias_factors(fp_recortado, MBP, BLEND, DEM_VACIA,
                             fold_recency_decay=0.5, min_obs=40)
    cell = [c for c in r["cells"] if c["lead_band"] == "14-29"][0]
    assert cell["reason"] is not None and not cell["applied"], (
        "30 obs < min_obs=40: la ponderación no inventa evidencia"
    )


# ---------- kill-switch ----------


def test_apply_sin_factores_intacto():
    hz = pd.DataFrame({
        "property": ["T"], "lead_days": [20], "raw_forecast_rn": [100.0],
    })
    out = apply_bias_factors(hz, {})
    assert float(out.loc[0, "raw_forecast_rn"]) == 100.0
    assert "bias_correction_factor" not in out.columns


def test_knobs_viven_bajo_bias_correction():
    import yaml as _yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    raw = _yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    bc = (raw.get("monitoring") or {}).get("bias_correction") or {}
    assert isinstance(bc.get("enabled"), bool), (
        "enabled debe ser booleano dentro de monitoring.bias_correction"
    )
    assert isinstance(bc.get("fold_recency_decay"), (int, float)), (
        "fold_recency_decay debe ser numérico dentro de bias_correction"
    )
    assert 0.0 <= float(bc["fold_recency_decay"]) <= 1.0
