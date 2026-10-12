"""Model Benchmark + entregables del proyecto (spec ``reporting``, 9.5/9.6).

- 9.5 tabla comparativa de los 7 candidatos con MAPE/WMAPE/MAE/RMSE/Bias/
  estabilidad y campeón declarado con evidencia out-of-sample.
- 9.6 generación de los 10 entregables en una corrida completa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import accuracy_band, compute_metrics
from src.evaluation.selection import SelectionResult, select_champion

CANDIDATE_ORDER = [
    "seasonal_naive",
    "pickup_curve",
    "ets",
    "sarimax",
    "lightgbm",
    "xgboost",
    "catboost",
]

# spec reporting §35 — los 10 entregables
DELIVERABLES = [
    "data_quality_report",
    "eda_report",
    "feature_engineering_report",
    "model_benchmark",
    "backtesting_report",
    "forecast_accuracy_report",
    "future_forecast_dataset",
    "model_explainability_report",
    "powerbi_dataset",
    "technical_documentation",
]


@dataclass
class BenchmarkResult:
    """Tabla comparativa + selección de campeón con evidencia out-of-sample."""

    table: pd.DataFrame
    selection: SelectionResult
    candidate_order: list[str] = field(default_factory=lambda: list(CANDIDATE_ORDER))
    source: str = "real_backtest"  # real_backtest | synthetic_placeholder

    @property
    def champion(self) -> str:
        return self.selection.champion

    @property
    def full_matrix(self) -> pd.DataFrame:
        """Tabla con TODOS los candidatos (rellenando ausentes con NaN)."""
        out = pd.DataFrame({"model": self.candidate_order})
        return out.merge(self.table, on="model", how="left")


def run_benchmark(
    fold_metrics: pd.DataFrame,
    candidate_order: list[str] | None = None,
    error_panel: pd.DataFrame | None = None,
    source: str = "real_backtest",
    dm_horizon: int = 30,
) -> BenchmarkResult:
    """9.5 — Evalúa los candidatos con métricas por fold (out-of-sample).

    ``error_panel`` habilita la prueba de significancia Diebold-Mariano del
    campeón contra el subcampeón (empate estadístico → desempate por
    parsimonia). ``dm_horizon`` es el largo REAL de la ventana de fold (pasos
    hacia adelante): la corrección HAC debe alinearse con él, porque los
    errores multi-paso de una ventana larga están correlacionados y un
    horizonte fijo demasiado corto inflaría la significancia. ``source``
    etiqueta el origen de la evidencia; un benchmark sintético de reemplazo
    se declara en el entregable.
    """
    order = candidate_order or CANDIDATE_ORDER
    selection = select_champion(
        fold_metrics, error_panel=error_panel, dm_horizon=int(dm_horizon)
    )

    table = selection.table()
    # asegurar presencia de las 6 columnas del spec (MAPE/WMAPE/MAE/RMSE/Bias + Stability)
    required = ("mape", "wmape", "mae", "rmse", "bias", "stability")
    missing = [c for c in required if c not in fold_metrics.columns and c not in table.columns]
    if missing:
        raise ValueError(
            f"fold_metrics sin métricas requeridas por el spec del benchmark: {missing}"
        )
    for col in required:
        if col not in table.columns:
            table[col] = np.nan

    # columna 'accuracy band' interpretativa (referencias Lighthouse)
    table["accuracy_band"] = table["mape"].map(accuracy_band)

    # estabilidad como rango entre ventanas (spec: columna Stability)
    if "fold" in fold_metrics.columns:
        spread = fold_metrics.groupby("model")["wmape"].agg(lambda s: float(s.max() - s.min()))
        table["stability_range"] = table["model"].map(spread)

    ordered = pd.DataFrame({"model": order}).merge(table, on="model", how="left")
    ordered["is_champion"] = ordered["model"] == selection.champion
    return BenchmarkResult(
        table=ordered, selection=selection, candidate_order=order, source=source
    )


def benchmark_text(result: BenchmarkResult) -> str:
    """Representación tabular legible (Markdown) para el reporte."""
    cols = ["model", "mape", "wmape", "mae", "rmse", "bias", "stability", "is_champion"]
    cols = [c for c in cols if c in result.table.columns]
    t = result.table[cols].copy()
    for c in cols[1:]:
        if c != "is_champion":
            t[c] = t[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    records = t.to_dict("records")
    rows = ["| " + " | ".join(str(rec[c]) for c in cols) + " |" for rec in records]
    out = "\n".join([header, sep, *rows]) + f"\n\n**Campeón**: {result.champion}\n"
    if result.source == "synthetic_placeholder":
        out += (
            "\n> ⚠️ **EVIDENCIA SINTÉTICA**: este benchmark proviene de datos de "
            "reemplazo, no de un backtest real. El campeón declarado no tiene "
            "evidencia out-of-sample.\n"
        )
    sig = result.selection.significance
    if sig is not None:
        verdict = (
            "ventaja significativa" if sig.get("significant")
            else "empate estadístico (desempate por parsimonia)"
        )
        out += (
            f"\n**Significancia (Diebold-Mariano)**: campeón vs "
            f"'{sig.get('runner_up')}' → DM={sig.get('dm'):.3f}, "
            f"p={sig.get('pvalue'):.4f} ({verdict}, n={sig.get('n_obs')}).\n"
        )
    out += f"\n_{result.selection.reason}_\n"
    return out


# --------------------------------------------------------------------------
# 9.6 — entregables
# --------------------------------------------------------------------------


class DeliverablesError(RuntimeError):
    pass


def generate_deliverables(
    run_dir: str | Path,
    *,
    data_quality_report: dict | None = None,
    eda_report: dict | None = None,
    feature_report: dict | None = None,
    benchmark: BenchmarkResult | None = None,
    backtest_report: dict | None = None,
    accuracy_report: dict | None = None,
    forecast_table: pd.DataFrame | None = None,
    explainability: pd.DataFrame | None = None,
    powerbi: dict[str, pd.DataFrame] | None = None,
    technical_doc: str | None = None,
) -> dict[str, Path]:
    """9.6 — Escribe los 10 entregables; falla si falta alguno.

    El spec exige que la corrida completa produzca los 10: si un entregable no
    se provée, lanza :class:`DeliverablesError` (no se publica a medias).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    provided = {
        "data_quality_report": data_quality_report,
        "eda_report": eda_report,
        "feature_engineering_report": feature_report,
        "model_benchmark": benchmark,
        "backtesting_report": backtest_report,
        "forecast_accuracy_report": accuracy_report,
        "future_forecast_dataset": forecast_table,
        "model_explainability_report": explainability,
        "powerbi_dataset": powerbi,
        "technical_documentation": technical_doc,
    }
    missing = [k for k, v in provided.items() if v is None]
    if missing:
        raise DeliverablesError(
            f"corrida incompleta: faltan entregables {missing} "
            "(spec reporting: los 10 deben producirse en una corrida completa)"
        )

    import json

    paths: dict[str, Path] = {}

    def _write_json(name: str, payload) -> None:
        p = run_dir / f"{name}.json"
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        paths[name] = p

    _write_json("data_quality_report", data_quality_report)
    _write_json("eda_report", eda_report)
    _write_json("feature_engineering_report", feature_report)

    bench_path = run_dir / "model_benchmark.md"
    bench_path.write_text(benchmark_text(benchmark), encoding="utf-8")
    paths["model_benchmark"] = bench_path

    _write_json("backtesting_report", backtest_report)
    _write_json("forecast_accuracy_report", accuracy_report)

    fc_path = run_dir / "future_forecast_dataset.csv"
    forecast_table.to_csv(fc_path, index=False)
    paths["future_forecast_dataset"] = fc_path

    ex_path = run_dir / "model_explainability_report.csv"
    explainability.to_csv(ex_path, index=False)
    paths["model_explainability_report"] = ex_path

    pbi_dir = run_dir / "powerbi"
    pbi_dir.mkdir(exist_ok=True)
    for area, frame in powerbi.items():
        frame.to_csv(pbi_dir / f"{area}.csv", index=False)
    paths["powerbi_dataset"] = pbi_dir

    doc_path = run_dir / "technical_documentation.md"
    doc_path.write_text(technical_doc, encoding="utf-8")
    paths["technical_documentation"] = doc_path

    # verificación: los 10 existen y no están vacíos
    not_written = [k for k in DELIVERABLES if not paths.get(k) or not paths[k].exists()]
    if not_written:
        raise DeliverablesError(f"entregables no escritos: {not_written}")
    return paths
