# -*- coding: utf-8 -*-
"""A/B — celdas estacionales de calibración (openspec 2026-10-30-seasonal-calibration-cells).

Compara la corrida BASELINE (knob off, `real-20261030-1147`) contra una
re-corrida con knob ON (misma data, misma seed — el resto del pipeline es
idéntico por diseño) y evalúa los criterios de aceptación de la proposal:

1. FIESTA mejora materialmente la cobertura (>= 65% desde 42.2%);
2. ninguna propiedad pierde cobertura de forma material (>= -5pp);
3. WMAPE del punto no empeora.

Uso:  python -X utf8 scripts/ab_seasonal_cells.py <run_off_dir> <run_on_dir>
Salida: 0 = cumple los criterios (activar default), 1 = no cumple.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.breakdowns import lead_band  # noqa: E402

FIESTA_MIN = 0.65   # criterio 1: cobertura objetivo para FIESTA
MAX_DEGRADE = -0.05  # criterio 2: pérdida máxima tolerada en otras props
WMAPE_TOL = 1.10    # criterio 3: tolerancia de deterioro del punto


def _cargar(run_dir: Path) -> pd.DataFrame:
    fut = pd.read_csv(run_dir / "future_forecast_dataset.csv", encoding="utf-8")
    fut["stay_date"] = pd.to_datetime(fut["stay_date"])

    d = pd.read_csv("data/demand_daily.csv")
    d["stay_date"] = pd.to_datetime(d["stay_date"], format="mixed")
    d["property"] = d["property"].replace({"Hotel Royal Corin": "CORIN"})
    d = d[~d["property"].isin(["LIRAK", "LAPAS"])]
    act = d.groupby(["property", "stay_date"], as_index=False)["rooms_sold"]
    act = act.sum().rename(columns={"rooms_sold": "actual"})

    ev = fut.merge(act, on=["property", "stay_date"], how="inner")
    ev["banda"] = ev["lead_days"].map(lead_band)
    ev["in_band"] = (
        (ev["actual"] >= ev["p10_forecast"]) & (ev["actual"] <= ev["p90_forecast"])
    )
    return ev


def _tabla(ev: pd.DataFrame) -> pd.DataFrame:
    cob = ev.groupby("property").agg(
        cobertura=("in_band", "mean"), n=("in_band", "size")
    )
    err = ev.assign(err_abs=(ev["final_forecast_rn"] - ev["actual"]).abs())
    we = err.groupby("property").agg(err_abs=("err_abs", "sum"),
                                     real=("actual", "sum"))
    we["wmape"] = we["err_abs"] / we["real"]
    return cob.join(we["wmape"])


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base_dir = Path(sys.argv[1])
    on_dir = Path(sys.argv[2])
    for d in (base_dir, on_dir):
        if not (d / "future_forecast_dataset.csv").exists():
            print(f"no existe future_forecast_dataset.csv en {d}")
            return 2
    base, on = _cargar(base_dir), _cargar(on_dir)
    comp = _tabla(base).join(_tabla(on), how="outer", lsuffix="_off", rsuffix="_on")
    comp["delta_cobertura"] = comp["cobertura_on"] - comp["cobertura_off"]

    print("\n== COBERTURA (horizonte con reales, out-of-sample) ==")
    print(
        comp[["cobertura_off", "cobertura_on", "delta_cobertura", "n_on"]]
        .assign(cobertura_off=lambda d_: (d_["cobertura_off"] * 100).round(1),
                cobertura_on=lambda d_: (d_["cobertura_on"] * 100).round(1),
                delta_cobertura=lambda d_: (d_["delta_cobertura"] * 100).round(1))
        .to_string()
    )
    print("\n== WMAPE del punto por propiedad (%) ==")
    print(
        comp[["wmape_off", "wmape_on"]]
        .assign(wmape_off=lambda d_: (d_["wmape_off"] * 100).round(1),
                wmape_on=lambda d_: (d_["wmape_on"] * 100).round(1))
        .to_string()
    )

    f_off = float(comp.loc["FIESTA", "cobertura_off"])
    f_on = float(comp.loc["FIESTA", "cobertura_on"])
    c1 = f_on >= FIESTA_MIN
    c2 = bool((comp.drop(index="FIESTA")["delta_cobertura"] >= MAX_DEGRADE).all())
    c3 = bool((comp["wmape_on"] <= comp["wmape_off"] * WMAPE_TOL).all())

    print("\n== veredicto (criterios de la proposal) ==")
    print(f"  1. FIESTA {f_off:.1%} -> {f_on:.1%} (umbral {FIESTA_MIN:.0%}): "
          f"{'CUMPLE' if c1 else 'NO CUMPLE'}")
    print(f"  2. sin deterioro material en otras (>= {MAX_DEGRADE:+.0%}): "
          f"{'CUMPLE' if c2 else 'NO CUMPLE'}")
    print(f"  3. WMAPE del punto no empeora (tolerancia {WMAPE_TOL:.0%}): "
          f"{'CUMPLE' if c3 else 'NO CUMPLE'}")
    decision = c1 and c2 and c3
    print("\n  DECISION:", "ACTIVAR default seasonal_cells=true"
          if decision else "DEJAR default off — sin evidencia suficiente")
    return 0 if decision else 1


if __name__ == "__main__":
    raise SystemExit(main())
