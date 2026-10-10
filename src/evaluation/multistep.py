"""Evaluación multi-step con snapshots congelados (spec ``evaluation``, task 6.6).

Protocolo (obligatorio): congelar los datos disponibles en cada corte T,
generar el forecast SIN acceso a datos posteriores a T, y comparar contra los
Room Nights reales posteriormente observados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.features.asof import AsOfBuilder, enforce_cutoff


@dataclass
class SnapshotEvaluation:
    """Resultado de un snapshot congelado."""

    cutoff: pd.Timestamp
    predictions: pd.DataFrame
    actuals: pd.DataFrame
    data_fingerprint: str
    history_max_observed: pd.Timestamp | None = None

    def merged(self) -> pd.DataFrame:
        """Forecast vs actual por (property, stay_date)."""
        p = self.predictions.rename(columns={"forecast_rn": "forecast"})
        a = self.actuals.rename(columns={"room_nights": "actual"})
        keys = [c for c in ("property", "stay_date") if c in p.columns and c in a.columns]
        return p.merge(a, on=keys, how="inner", suffixes=("_f", "_a"))


@dataclass
class MultiStepResult:
    snapshots: list[SnapshotEvaluation] = field(default_factory=list)

    @property
    def n_snapshots(self) -> int:
        return len(self.snapshots)

    def all_merged(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame()
        return pd.concat([s.merged() for s in self.snapshots], ignore_index=True)


def evaluate_at_cutoff(
    *,
    cutoff: pd.Timestamp,
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    forecast_fn: Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame],
    date_col: str = "stay_date",
    value_col: str = "room_nights",
    property_col: str = "property",
) -> SnapshotEvaluation:
    """Congela datos en ``cutoff``, predice y alinea contra reales observados.

    Parameters
    ----------
    forecast_fn:
        ``(history_asof, cutoff) -> DataFrame`` con ``forecast_rn``. Debe
        trabajar solo con ``history_asof`` (ya recortado ≤ cutoff).
    """
    cutoff = pd.Timestamp(cutoff)
    builder = AsOfBuilder(date_col=date_col, cutoff=cutoff)
    history_asof = builder.frame(history, "history@snapshot")
    enforce_cutoff(history_asof, date_col, cutoff, "history@snapshot")

    predictions = forecast_fn(history_asof, cutoff)
    if "forecast_rn" not in predictions.columns:
        raise KeyError("forecast_fn debe devolver una columna 'forecast_rn'")

    # predicciones solo para stay dates observables como objetivo (>= cutoff
    # es correcto: es lo que se intenta predecir), pero el history de entrada
    # ya está congelado.
    pred = predictions.copy()
    pred[date_col] = pd.to_datetime(pred[date_col], errors="coerce")

    # los actuales se toman del frame completo (son la verdad posterior)
    act = actuals[[property_col, date_col, value_col]].copy()
    act[date_col] = pd.to_datetime(act[date_col], errors="coerce")

    from src.monitoring.run_manifest import fingerprint

    history_max = pd.to_datetime(history_asof[date_col]).max()
    return SnapshotEvaluation(
        cutoff=cutoff,
        predictions=pred,
        actuals=act,
        data_fingerprint=fingerprint(
            {
                "cutoff": str(cutoff.date()),
                "n_history": int(len(history_asof)),
                "history_max": str(history_max.date()),
            }
        ),
        history_max_observed=history_max,
    )


def run_multistep(
    *,
    cutoffs: list[pd.Timestamp],
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    forecast_fn: Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame],
    date_col: str = "stay_date",
    value_col: str = "room_nights",
    property_col: str = "property",
) -> MultiStepResult:
    """Evalúa cada snapshot de forma independiente y congelada."""
    results = []
    for cutoff in cutoffs:
        results.append(
            evaluate_at_cutoff(
                cutoff=cutoff,
                history=history,
                actuals=actuals,
                forecast_fn=forecast_fn,
                date_col=date_col,
                value_col=value_col,
                property_col=property_col,
            )
        )
    return MultiStepResult(snapshots=results)


def assert_no_future_reads(result: MultiStepResult) -> None:
    """Invariante: cada snapshot solo leyó datos ≤ su corte congelado."""
    for snap in result.snapshots:
        if snap.predictions.empty:
            raise AssertionError(f"snapshot {snap.cutoff.date()} sin predicciones")
        if snap.history_max_observed is None:
            raise AssertionError(f"snapshot {snap.cutoff.date()} sin huella de historial")
        if pd.Timestamp(snap.history_max_observed) > pd.Timestamp(snap.cutoff):
            raise AssertionError(
                f"snapshot {snap.cutoff.date()} leyó historia hasta "
                f"{pd.Timestamp(snap.history_max_observed).date()} (lookahead)"
            )
