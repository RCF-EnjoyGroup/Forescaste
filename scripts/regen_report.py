"""Regenera report.html de una corrida existente sin re-correr el pipeline.

Útil cuando se mejora la PRESENTACIÓN del reporte (visual_report.py): los
10 entregables de la corrida ya contienen todo lo necesario (métricas,
desgloses, cobertura, backtest, manifiesto), así que la vista se reconstruye
en segundos con los mismos datos — misma lógica de presentación que usa el
runner, sin re-entrenar nada.

Uso:
    python scripts/regen_report.py outputs/runs/real-20261020-0746
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# proyecto en sys.path: el script vive en scripts/ pero importa src/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.reporting.visual_report import generate_visual_report  # noqa: E402


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_demand() -> pd.DataFrame:
    """Demanda normalizada (solo para la sección EDA y los reales)."""
    from src.config import load_config
    from src.data.schema import SourceKind, detect_schema
    from src.forecasting.core import _apply_aliases, _normalize_demand

    cfg = load_config()
    demand = pd.read_csv(cfg.source_paths["demand"])
    schema = detect_schema(demand, SourceKind.DEMAND, cfg.column_hints)
    demand = _apply_aliases(demand, schema, cfg.property_aliases)
    excluded = [str(p) for p in cfg.excluded_properties]
    demand = demand.loc[~demand[schema.property_col].astype(str).isin(excluded)]
    return _normalize_demand(demand, schema)


def _benchmark_table(run_dir: Path) -> pd.DataFrame | None:
    """Reconstruye la tabla del benchmark desde model_benchmark.md.

    Incluye ``is_champion`` (última columna del markdown): el realce del
    campeón en la gráfica depende de ella — o del parámetro explícito de
    ``generate_visual_report`` como fallback.
    """
    md = run_dir / "model_benchmark.md"
    if not md.exists():
        return None
    rows: list[dict] = []
    for line in md.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8 or cells[0] in {"model", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        try:
            rows.append({
                "model": cells[0],
                "mape": float(cells[1]), "wmape": float(cells[2]),
                "mae": float(cells[3]), "rmse": float(cells[4]),
                "bias": float(cells[5]), "stability": float(cells[6]),
                "is_champion": cells[7].lower() == "true",
            })
        except ValueError:
            continue
    return pd.DataFrame(rows) if rows else None


def main(run_dir: str) -> int:
    run_dir = Path(run_dir)
    man = _load_json(run_dir / "run_manifest.json")
    acc = _load_json(run_dir / "forecast_accuracy_report.json")
    backtest = _load_json(run_dir / "backtesting_report.json")

    forecast = pd.read_csv(run_dir / "future_forecast_dataset.csv")
    forecast["stay_date"] = pd.to_datetime(forecast["stay_date"])

    dem = _load_demand()
    actual = None
    merged = forecast.merge(
        dem[["property", "stay_date", "room_nights"]],
        on=["property", "stay_date"], how="inner",
    )
    if not merged.empty:
        actual = merged.sort_values(["property", "stay_date"])["room_nights"]

    hyper = man.get("hyperparameters") or {}
    mon = man.get("monitoring") or {}
    cov = acc.get("interval_coverage") or None
    # corrida futura: la banda vive sobre el horizonte sin reales — la
    # nota "no medible" se presenta tal cual en el panel de validación
    if isinstance(cov, dict) and not (cov.get("global") or cov.get("note")):
        cov = None

    # panel de interpretación y validación (auto-explicado)
    vintage = man.get("vintage_tracking") or None
    vintage = dict(vintage) if isinstance(vintage, dict) else None
    if vintage and "detail_csv" not in vintage:
        # corridas previas a la mejora: el CSV backfilled se detecta en disco
        detail_path = run_dir / "vintage_tracking_detail.csv"
        if detail_path.exists():
            vintage["detail_csv"] = detail_path.name
            with open(detail_path, encoding="utf-8") as fh:
                vintage["n_detail_rows"] = sum(1 for _ in fh) - 1
    out = generate_visual_report(
        run_dir,
        forecast,
        demand=dem,
        benchmark=_benchmark_table(run_dir),
        metrics=man.get("metrics") or None,
        kpis=man.get("kpis") or None,
        actual=actual,
        property_breakdown=(
            pd.DataFrame(acc["by_property"]) if acc.get("by_property") else None
        ),
        property_weekly_breakdown=(
            pd.DataFrame(acc["by_property_weekly"])
            if acc.get("by_property_weekly") else None
        ),
        model_by_property=hyper.get("per_property") or None,
        backtest_report=backtest,
        champion=hyper.get("model"),
        significance=man.get("champion_significance") or None,
        interval_coverage=cov if isinstance(cov, dict) and (
            cov.get("global") or cov.get("note")
        ) else None,
        cold_start=man.get("cold_start_properties") or None,
        drift_alerts=mon.get("drift_alerts") or None,
        monitoring_state=mon.get("retraining_state") or None,
        bias_correction=man.get("bias_correction") or None,
        vintage_tracking=vintage,
        forecast_revision=man.get("forecast_revision") or None,
        run_id=run_dir.name,
        run_mode=(
            "futuro" if run_dir.name.startswith("futuro")
            else ("real" if run_dir.name.startswith("real") else None)
        ),
    )
    print(f"reporte regenerado: {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
