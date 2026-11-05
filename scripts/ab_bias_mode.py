# -*- coding: utf-8 -*-
"""A/B 3 brazos — modos de la capa de corrección de bias (openspec
2026-11-05-bias-correction-mode).

Brazo A: baseline capa ON folds uniformes (run existente).
Brazo B: capa OFF (kill-switch).
Brazo C: capa ON con fold_recency_decay=0.7.
Misma data (reales al 3-nov) en los tres. Comparación pareada sobre el
horizonte de verificación (powerbi/forecast.csv) y cobertura del
forecast_accuracy_report.

Criterios de la proposal:
1. el sesgo global se acerca a 0;
2. |sesgo| de MARINA, SJOSL, CORIN y FIESTA mejora;
3. el WMAPE de ninguna propiedad empeora >2pp;
4. la cobertura de banda no cae >5pp en ninguna propiedad.

Uso:  python -X utf8 scripts/ab_bias_mode.py <run_A> <run_B> <run_C>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FOCUS = ("MARINA", "SJOSL", "CORIN", "FIESTA")


def _cargar(run_dir: str) -> tuple[pd.DataFrame, dict]:
    fc = pd.read_csv(Path(run_dir) / "powerbi" / "forecast.csv",
                     encoding="utf-8")
    fc["stay_date"] = pd.to_datetime(fc["stay_date"], format="mixed")
    acc = json.load(open(Path(run_dir) / "forecast_accuracy_report.json",
                         encoding="utf-8"))
    return fc, acc


def _resumen(fc: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for prop, g in fc.groupby("property", observed=True):
        err = g["final_forecast_rn"] - g["actual_rn"]
        filas.append({
            "property": prop,
            "wmape": float(err.abs().sum() / g["actual_rn"].sum()),
            "bias": float(err.sum() / g["actual_rn"].sum()),
        })
    return pd.DataFrame(filas).set_index("property")


def _cobertura(acc: dict) -> dict:
    out = {}
    for row in acc.get("interval_coverage", []):
        if isinstance(row, dict) and row.get("property"):
            out[row["property"]] = float(row.get("coverage") or 0.0)
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    fa, fb, fc_ = (_cargar(sys.argv[i]) for i in (1, 2, 3))
    A, B, C = _resumen(fa[0]), _resumen(fb[0]), _resumen(fc_[0])
    cov = {"A": _cobertura(fa[1]), "B": _cobertura(fb[1]), "C": _cobertura(fc_[1])}

    tabla = A.join(B, lsuffix="_A", rsuffix="_B").join(C.add_suffix("_C"))
    print("== por propiedad (sesgo | WMAPE) ==")
    out = tabla.copy()
    for c in out.columns:
        out[c] = (out[c] * 100).round(1)
    print(out.to_string())
    print()

    wg = {}
    fcdfs = {"A": fa[0], "B": fb[0], "C": fc_[0]}
    for nombre, df in (("A", A), ("B", B), ("C", C)):
        # wmape global ponderado por RN reales
        fcdf = fcdfs[nombre]
        err = (fcdf["final_forecast_rn"] - fcdf["actual_rn"]).abs().sum()
        wg[nombre] = float(err / fcdf["actual_rn"].sum())
    print(f"WMAPE global:  A {wg['A']:.1%} | B {wg['B']:.1%} | C {wg['C']:.1%}")
    print(f"|sesgo| medio (4 hoteles foco): "
          f"A {np.mean([abs(A.loc[p,'bias']) for p in FOCUS]):.1%} | "
          f"B {np.mean([abs(B.loc[p,'bias']) for p in FOCUS]):.1%} | "
          f"C {np.mean([abs(C.loc[p,'bias']) for p in FOCUS]):.1%}")
    print()

    def gana(df: pd.DataFrame, base: pd.DataFrame, cov_key: dict,
             cov_base: dict):
        c1 = abs(float(df["bias"].sum())) < abs(float(base["bias"].sum()))
        c2 = all(
            abs(float(df.loc[p, "bias"])) < abs(float(base.loc[p, "bias"]))
            for p in FOCUS
        )
        c3 = bool(((df["wmape"] - base["wmape"]) <= 0.02).all())
        c4 = all(
            cov_key.get(p, 0.0) >= cov_base.get(p, 0.0) - 0.05
            for p in df.index
        )
        return c1 and c2 and c3 and c4, c1, c2, c3, c4

    print("== veredicto por brazo (vs baseline A) ==")
    resultados = {}
    for nombre, df, cov_key, cov_base in (
        ("B (capa OFF)", B, cov["B"], cov["A"]),
        ("C (recencia 0.7)", C, cov["C"], cov["A"]),
    ):
        ok, c1, c2, c3, c4 = gana(df, A, cov_key, cov_base)
        resultados[nombre] = ok
        print(f"  {nombre}: sesgo global mejor={c1} | foco mejora={c2} | "
              f"wmape estable={c3} | cobertura estable={c4} -> "
              f"{'CUMPLE' if ok else 'NO CUMPLE'}")
    decision = any(resultados.values())
    print("\n  DECISION:", "activar el brazo ganador como default"
          if decision else "DEJAR defaults actuales (capa ON uniforme) — "
          "sin evidencia suficiente para cambiar")
    return 0 if decision else 1


if __name__ == "__main__":
    raise SystemExit(main())
