# -*- coding: utf-8 -*-
"""Diagnóstico: regenerar el reporte visual de una corrida desde sus
artefactos y reportar por qué se dibujan (o no) las tarjetas que
dependen de reales (Dispersión, Forecast vs Real).

Uso: python -X utf8 scripts/debug_report_cards.py <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import visual_report as vr  # noqa: E402


def main() -> int:
    run_dir = Path(sys.argv[1])
    fc = pd.read_csv(run_dir / "future_forecast_dataset.csv", encoding="utf-8")
    pb = pd.read_csv(run_dir / "powerbi" / "forecast.csv", encoding="utf-8")
    fc["stay_date"] = pd.to_datetime(fc["stay_date"], errors="coerce")
    pb["stay_date"] = pd.to_datetime(pb["stay_date"], errors="coerce")

    # reconstruir el actual_series exactamente como el runner: merge
    # (property, stay_date) left desde forecast_final
    act = pb[["property", "stay_date", "actual_rn"]].drop_duplicates(
        subset=["property", "stay_date"]
    ).rename(columns={"actual_rn": "actual"})
    joined = fc.reset_index(drop=True).merge(
        act, on=["property", "stay_date"], how="left"
    )
    actual = joined["actual"]
    print("filas forecast:", len(fc), "| filas actual:", len(actual),
          "| actual no-nan:", int(actual.notna().sum()))

    # replica del chequeo interno del generador
    f2 = vr._with_lead_band(fc)
    print("len(_with_lead_band(fc)):", len(f2),
          "| chequeo longitud pasaria:", len(actual) == len(f2))

    # reproducir las dos tarjetas dependientes de reales
    c1 = vr._chart_forecast_vs_actual(f2, actual)
    c2 = vr._scatter_forecast_vs_actual(f2, actual)
    print("chart vs actual:", "OK" if c1 else "None")
    print("scatter:", "OK" if c2 else "None")

    import json
    mf = json.load(open(run_dir / "run_manifest.json", encoding="utf-8"))
    out = vr.generate_visual_report(
        run_dir,
        fc,
        demand=pd.read_csv("data/demand_daily.csv", encoding="utf-8-sig"),
        metrics=mf.get("metrics"),
        kpis=mf.get("kpis"),
        actual=actual,
        property_breakdown=pd.DataFrame(
            json.load(open(run_dir / "forecast_accuracy_report.json",
                           encoding="utf-8"))["by_property"]),
        property_weekly_breakdown=pd.DataFrame(
            json.load(open(run_dir / "forecast_accuracy_report.json",
                           encoding="utf-8")).get("by_property_weekly") or []),
        model_by_property=mf.get("model_by_property"),
        backtest_report=mf.get("backtest") or {},
        champion=mf.get("champion"),
        significance=mf.get("champion_significance"),
        interval_coverage=json.load(open(
            run_dir / "forecast_accuracy_report.json", encoding="utf-8")
        )["interval_coverage"],
        cold_start=mf.get("cold_start_properties"),
        drift_alerts=(mf.get("monitoring") or {}).get("drift_alerts"),
        monitoring_state=(mf.get("monitoring") or {}).get("retraining_state"),
        bias_correction=mf.get("bias_correction"),
        vintage_tracking=mf.get("vintage_tracking"),
        forecast_revision=mf.get("forecast_revision"),
        lead_band_breakdown=json.load(open(
            run_dir / "forecast_accuracy_report.json", encoding="utf-8"
        ))["by_lead_band"],
        track_record=(mf.get("vintage_tracking") or {}).get("track_record"),
        run_id=str(run_dir.name),
        run_mode="verificacion",
    )
    print("reporte regenerado:", out)
    t = out.read_text(encoding="utf-8")
    print("tarjetas vs real en el HTML regenerado:",
          "Forecast vs Real" in t, "| Dispersion:", "Dispersi" in t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
