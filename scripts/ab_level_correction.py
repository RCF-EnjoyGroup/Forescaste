# -*- coding: utf-8 -*-
"""A/B replay — corrección de nivel por drift (openspec 2026-11-05-drift-level-correction).

Walk-forward sobre el registro de corridas futuras: para cada par
(publicación anterior → publicación siguiente, en orden de tiempo), el
factor de nivel se estima SOLO con errores ≤ corte de la siguiente y se
aplica a su horizonte publicado; se puntúa contra los reales
materializados (data vigente — la comparación es pareada).

Criterios de la proposal:
1. WMAPE global del conjunto de añadas mejora (>=2% relativo);
2. FIESTA: |sesgo| mejora >=5pp hacia 0;
3. ninguna propiedad empeora WMAPE >2pp;
4. la cobertura de banda no empeora >5pp en ninguna propiedad.

Uso:  python -X utf8 scripts/ab_level_correction.py
Salida: 0 = cumple (activar default), 1 = no cumple.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.accuracy_tracking import (  # noqa: E402
    load_published_forecast,
    vintage_detail,
)
from src.forecasting.core import level_correction_factors  # noqa: E402


def _cargar_reales() -> pd.DataFrame:
    d = pd.read_csv("data/demand_daily.csv", encoding="utf-8-sig")
    d["stay_date"] = pd.to_datetime(d["stay_date"], format="mixed")
    d = d[~d["property"].astype(str).isin(["LIRAK", "LAPAS"])]
    return d.rename(columns={})[["property", "stay_date", "rooms_sold"]].rename(
        columns={"rooms_sold": "actual"}
    )


def main() -> int:
    import json

    registry = json.load(open("outputs/monitoring/run_registry.json", encoding="utf-8"))
    futuros = sorted(
        (r for r in registry.get("records") or []
         if str(r.get("mode")) == "futuro" and r.get("run_dir")),
        key=lambda r: str(r.get("timestamp") or ""),
    )
    if len(futuros) < 2:
        print("registro sin suficientes corridas futuras")
        return 2
    reales = _cargar_reales()

    filas: list[pd.DataFrame] = []
    pares = 0
    CORTE_MIN = pd.Timestamp("2026-10-27")  # post-corrección de alcance MARINA
    for i in range(1, len(futuros)):
        prev, curr = futuros[i - 1], futuros[i]
        corte_curr = pd.Timestamp(str(curr.get("cutoff") or ""))
        if corte_curr < CORTE_MIN:
            continue  # publicaciones pre-corrección: mezclan alcances (MARINA)
        pub_prev = load_published_forecast(str(prev.get("run_dir") or ""))
        pub_curr = load_published_forecast(str(curr.get("run_dir") or ""))
        if pub_prev is None or pub_curr is None:
            continue
        corte = corte_curr
        detail = vintage_detail(pub_prev, reales.rename(
            columns={"actual": "room_nights"}), through=corte)
        factors, _ = level_correction_factors(detail)
        pares += 1
        # horizonte de la corrida actual con reales materializados
        hz = pub_curr.copy()
        hz["stay_date"] = pd.to_datetime(hz["stay_date"], errors="coerce")
        hz = hz[hz["stay_date"] <= reales["stay_date"].max()]
        hz = hz.merge(
            reales, on=["property", "stay_date"], how="inner"
        )
        if hz.empty:
            continue
        hz["p50_base"] = pd.to_numeric(
            hz.get("final_forecast_rn", hz.get("p50_forecast")),
            errors="coerce",
        )
        hz = hz.dropna(subset=["p50_base", "actual"])
        hz["factor"] = hz["property"].astype(str).map(
            lambda p: float(factors.get(str(p), 1.0))
        )
        hz["p50_corr"] = hz["p50_base"] * hz["factor"]
        n_corr = int((hz["factor"] != 1.0).sum())
        print("  par %s -> %s: factores=%s | corregidas=%d/%d" % (
            prev.get("run_id", "?"), curr.get("run_id", "?"),
            {k: round(v, 3) for k, v in factors.items()} or "{}",
            n_corr, len(hz)))
        for col in ("p10_forecast", "p90_forecast"):
            if col in hz.columns:
                hz[col + "_base"] = pd.to_numeric(hz[col], errors="coerce")
                hz[col + "_corr"] = hz[col + "_base"] * hz["factor"]
        filas.append(hz)

    if not filas:
        print("sin pares evaluables")
        return 2
    todo = pd.concat(filas, ignore_index=True)
    print(f"pares walk-forward evaluados: {pares} | estancias puntuadas: {len(todo)}")
    print()

    filas_res = []
    for prop, g in todo.groupby("property", observed=True):
        act = g["actual"].to_numpy(dtype=float)
        base, corr = g["p50_base"].to_numpy(dtype=float), g["p50_corr"].to_numpy(dtype=float)

        def _stats(punto: np.ndarray) -> tuple[float, float]:
            wmape = float(np.abs(punto - act).sum() / np.abs(act).sum())
            bias = float((punto - act).sum() / np.abs(act).sum())
            return wmape, bias

        w_b, b_b = _stats(base)
        w_c, b_c = _stats(corr)
        cov_b = cov_c = float("nan")
        if "p10_forecast_base" in g.columns:
            dentro_b = (act >= g["p10_forecast_base"]) & (act <= g["p90_forecast_base"])
            dentro_c = (act >= g["p10_forecast_corr"]) & (act <= g["p90_forecast_corr"])
            cov_b, cov_c = float(dentro_b.mean()), float(dentro_c.mean())
        filas_res.append({
            "property": prop, "n": len(g),
            "wmape_base": w_b, "wmape_corr": w_c,
            "bias_base": b_b, "bias_corr": b_c,
            "cov_base": cov_b, "cov_corr": cov_c,
        })
    res = pd.DataFrame(filas_res).set_index("property")

    print("== por propiedad: baseline vs con corrección de nivel ==")
    out = res.copy()
    for c in ("wmape_base", "wmape_corr", "bias_base", "bias_corr",
              "cov_base", "cov_corr"):
        out[c] = (out[c] * 100).round(1)
    print(out.to_string())
    print()

    act = todo["actual"].to_numpy(dtype=float)
    wg_b = float(np.abs(todo["p50_base"] - act).sum() / np.abs(act).sum())
    wg_c = float(np.abs(todo["p50_corr"] - act).sum() / np.abs(act).sum())
    print(f"WMAPE global: base {wg_b:.1%} -> corregido {wg_c:.1%} "
          f"(mejora {(wg_b - wg_c) / wg_b:+.1%})")

    c1 = wg_c <= wg_b * 0.98
    f = res.loc["FIESTA"]
    c2 = abs(float(f["bias_corr"])) <= abs(float(f["bias_base"])) - 0.05
    c3 = bool(((res["wmape_corr"] - res["wmape_base"]) <= 0.02).all())
    if res["cov_base"].notna().all():
        c4 = bool(((res["cov_corr"] - res["cov_base"]) >= -0.05).all())
    else:
        c4 = True

    print("\n== veredicto (criterios de la proposal) ==")
    print(f"  1. WMAPE global mejora >=2%: {'CUMPLE' if c1 else 'NO CUMPLE'}")
    print(f"  2. FIESTA |sesgo| mejora >=5pp: {abs(f['bias_base']):.1%} -> "
          f"{abs(f['bias_corr']):.1%}: {'CUMPLE' if c2 else 'NO CUMPLE'}")
    print(f"  3. nadie empeora WMAPE >2pp: {'CUMPLE' if c3 else 'NO CUMPLE'}")
    print(f"  4. cobertura no cae >5pp: {'CUMPLE' if c4 else 'NO CUMPLE'}")
    decision = c1 and c2 and c3 and c4
    print("\n  DECISION:", "ACTIVAR default level_correction=true"
          if decision else "DEJAR default off — sin evidencia suficiente")
    return 0 if decision else 1


if __name__ == "__main__":
    raise SystemExit(main())
