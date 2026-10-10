"""Monitoring: drift, outliers y retraining (spec ``monitoring``, 10.1–10.4).

- 10.1 drift de MAPE/WMAPE/Bias → alerta con métrica y magnitud.
- 10.2 drift de features, curva de pickup y distribución de demanda.
- 10.3 detección de outliers con clasificación, SIN eliminación automática.
- 10.4 retraining configurable con omisión documentada si no hay datos
  nuevos suficientes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Severity = Literal["info", "warning", "alert"]


@dataclass
class DriftAlert:
    metric: str
    severity: Severity
    message: str
    reference: float
    current: float

    @property
    def magnitude(self) -> float:
        return abs(self.current - self.reference)

    @property
    def relative_change(self) -> float:
        if self.reference == 0:
            return float("inf") if self.current != 0 else 0.0
        return (self.current - self.reference) / abs(self.reference)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "severity": self.severity,
            "message": self.message,
            "reference": self.reference,
            "current": self.current,
            "magnitude": self.magnitude,
            "relative_change": self.relative_change,
        }


# Umbrales conservadores (design: arrancar conservador y re-calibrar)
DRIFT_THRESHOLDS = {
    "mape": {"warning": 0.10, "alert": 0.25},     # relativo
    "wmape": {"warning": 0.10, "alert": 0.25},
    "bias": {"warning": 0.05, "alert": 0.10},     # absoluto (puntos)
}


def check_metric_drift(
    reference: dict[str, float],
    current: dict[str, float],
    thresholds: dict[str, dict[str, float]] | None = None,
) -> list[DriftAlert]:
    """10.1 — Deterioro de MAPE/WMAPE/Bias vs referencia histórica."""
    thresholds = thresholds or DRIFT_THRESHOLDS
    alerts: list[DriftAlert] = []
    for metric, thr in thresholds.items():
        if metric not in reference or metric not in current:
            continue
        ref, cur = float(reference[metric]), float(current[metric])
        if not (np.isfinite(ref) and np.isfinite(cur)):
            continue

        if metric == "bias":
            # bias: magnitud absoluta en puntos porcentuales
            delta = abs(cur) - abs(ref)
            if delta >= thr["alert"]:
                sev: Severity = "alert"
            elif delta >= thr["warning"]:
                sev = "warning"
            else:
                continue
            rel = delta
        else:
            rel = (cur - ref) / abs(ref) if ref else (float("inf") if cur else 0.0)
            if rel >= thr["alert"]:
                sev = "alert"
            elif rel >= thr["warning"]:
                sev = "warning"
            else:
                continue

        direction = "deterioró" if metric != "bias" else "se acentuó"
        alerts.append(
            DriftAlert(
                metric=metric,
                severity=sev,
                message=(
                    f"Drift de {metric.upper()}: {ref:.4f} → {cur:.4f} "
                    f"({rel:+.1%}) — el modelo {direction}"
                ),
                reference=ref,
                current=cur,
            )
        )
    return alerts


def check_distribution_drift(
    reference: pd.Series,
    current: pd.Series,
    metric_name: str,
    warning: float = 0.10,
    alert: float = 0.25,
) -> DriftAlert | None:
    """Drift de distribución vía distancia de medias normalizada (PSI-like)."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty or ref.std() == 0:
        return None
    # diferencia de medias en desviaciones de la referencia
    shift = abs(cur.mean() - ref.mean()) / ref.std()
    if shift >= alert:
        sev: Severity = "alert"
    elif shift >= warning:
        sev = "warning"
    else:
        return None
    return DriftAlert(
        metric=metric_name,
        severity=sev,
        message=(
            f"Drift de {metric_name}: media {ref.mean():.2f} → {cur.mean():.2f} "
            f"(desplazamiento {shift:.2f}σ)"
        ),
        reference=float(ref.mean()),
        current=float(cur.mean()),
    )


def check_pickup_curve_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    lead_col: str = "lead_days",
    value_col: str = "pickup_rn",
    warning: float = 0.15,
    alert: float = 0.35,
) -> DriftAlert | None:
    """Drift de la curva de pickup por lead: desplazamiento relativo medio."""
    ref = reference.groupby(lead_col)[value_col].mean()
    cur = current.groupby(lead_col)[value_col].mean()
    common = ref.index.intersection(cur.index)
    if len(common) == 0:
        return None
    rel = ((cur.loc[common] - ref.loc[common]).abs() / ref.loc[common].replace(0, np.nan)).dropna()
    if rel.empty:
        return None
    mean_shift = float(rel.mean())
    if mean_shift >= alert:
        sev: Severity = "alert"
    elif mean_shift >= warning:
        sev = "warning"
    else:
        return None
    return DriftAlert(
        metric="pickup_curve",
        severity=sev,
        message=(
            f"Drift de la curva de pickup: desplazamiento medio {mean_shift:.1%} "
            f"en {len(common)} leads comunes"
        ),
        reference=float(ref.loc[common].mean()),
        current=float(cur.loc[common].mean()),
    )


def run_drift_suite(
    *,
    reference_metrics: dict[str, float],
    current_metrics: dict[str, float],
    reference_demand: pd.Series | None = None,
    current_demand: pd.Series | None = None,
    reference_pickup: pd.DataFrame | None = None,
    current_pickup: pd.DataFrame | None = None,
    feature_reference: pd.DataFrame | None = None,
    feature_current: pd.DataFrame | None = None,
) -> list[DriftAlert]:
    """10.1/10.2 — Suite completa de drift."""
    alerts = check_metric_drift(reference_metrics, current_metrics)

    if reference_demand is not None and current_demand is not None:
        a = check_distribution_drift(reference_demand, current_demand, "demand_distribution")
        if a:
            alerts.append(a)

    if reference_pickup is not None and current_pickup is not None:
        a = check_pickup_curve_drift(reference_pickup, current_pickup)
        if a:
            alerts.append(a)

    if feature_reference is not None and feature_current is not None:
        for col in feature_reference.columns:
            if col in feature_current.columns and pd.api.types.is_numeric_dtype(feature_reference[col]):
                a = check_distribution_drift(
                    feature_reference[col], feature_current[col], f"feature:{col}"
                )
                if a:
                    alerts.append(a)
    return alerts


# --------------------------------------------------------------------------
# 10.3 — Outliers con clasificación (sin eliminación automática)
# --------------------------------------------------------------------------

OUTLIER_CATEGORIES = (
    "data_error",      # error de datos
    "event",           # evento real
    "group",           # grupo
    "seasonality",     # temporada
    "structural",      # cambio estructural
    "unclassified",    # no clasificado aún
)


@dataclass
class ClassifiedOutlier:
    property: str
    date: str
    value: float
    zscore: float
    category: str
    rationale: str
    removed: bool = False  # SIEMPRE False: nunca se eliminan automáticamente

    def to_dict(self) -> dict:
        return {
            "property": self.property,
            "date": self.date,
            "value": self.value,
            "zscore": self.zscore,
            "category": self.category,
            "rationale": self.rationale,
            "removed": self.removed,
        }


def detect_outliers(
    df: pd.DataFrame,
    *,
    value_col: str = "room_nights",
    date_col: str = "stay_date",
    property_col: str = "property",
    capacity_col: str | None = "capacity",
    threshold: float = 3.0,
) -> list[ClassifiedOutlier]:
    """10.3 — Detecta picos/caídas y los CLASIFICA (nunca los elimina)."""
    out: list[ClassifiedOutlier] = []
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

    for prop, grp in work.groupby(property_col, dropna=False):
        vals = pd.to_numeric(grp[value_col], errors="coerce")
        mean, std = vals.mean(), vals.std()
        if not std or not np.isfinite(std):
            continue
        z = (vals - mean) / std
        flagged = grp.loc[z.abs() >= threshold]
        for idx, row in flagged.iterrows():
            zi = float(z.loc[idx])
            date = pd.Timestamp(row[date_col])
            value = float(row[value_col])

            category, rationale = _classify(
                row=row,
                zscore=zi,
                date=date,
                grp=grp,
                capacity_col=capacity_col,
                value_col=value_col,
            )
            out.append(
                ClassifiedOutlier(
                    property=str(prop),
                    date=str(date.date()),
                    value=value,
                    zscore=zi,
                    category=category,
                    rationale=rationale,
                    removed=False,
                )
            )
    return out


def _classify(*, row, zscore, date, grp, capacity_col, value_col) -> tuple[str, str]:
    """Reglas de clasificación (spec: clasificar ANTES de decidir).

    Un pico puede ser: error de datos, evento real, grupo, temporada o cambio
    estructural. Los eventos reales se CONSERVAN.
    """
    # error de datos: RN > capacidad (imposible físicamente)
    if capacity_col and capacity_col in row.index:
        cap = row.get(capacity_col)
        if pd.notna(cap) and float(row[value_col]) > float(cap) * 1.05:
            return (
                "data_error",
                f"RN {row[value_col]:.0f} > capacidad {float(cap):.0f} (imposible físicamente)",
            )

    # capacidad reducida ese día: no es demanda anómala, es inventario
    if capacity_col and capacity_col in grp.columns:
        cap_mean = pd.to_numeric(grp[capacity_col], errors="coerce").mean()
        if pd.notna(row.get(capacity_col)) and float(row[capacity_col]) < cap_mean * 0.7:
            return (
                "structural",
                f"capacidad reducida ({float(row[capacity_col]):.0f} vs media {cap_mean:.0f})",
            )

    # fin de semana / temporada alta: patrón estacional conocido
    dow = date.dayofweek
    if dow >= 5 and zscore > 0:
        return ("seasonality", f"fin de semana (dow={dow}) con pico positivo")

    month = date.month
    if month in (12, 1, 2) and zscore > 0:
        return ("seasonality", f"temporada alta (mes={month}) con pico positivo")

    # cambio estructural: nivel que se mantiene tras el pico
    idx = grp.index.get_loc(row.name) if hasattr(row, "name") else 0
    if isinstance(idx, slice):
        idx = idx.start or 0
    try:
        after = grp.iloc[int(idx) + 1 : int(idx) + 8][value_col].astype(float)
        if len(after) >= 3 and zscore > 0 and after.mean() > grp[value_col].astype(float).mean() * 1.15:
            return ("structural", "el nivel se mantiene elevado después del pico (cambio estructural)")
    except Exception:  # pragma: no cover - posicionamiento defensivo
        pass

    # evento: aislado (el entorno vuelve a la normalidad)
    return ("event", f"pico aislado |z|={abs(zscore):.1f} con entorno normal (evento real: se conserva)")


def classified_frame(outliers: list[ClassifiedOutlier]) -> pd.DataFrame:
    if not outliers:
        return pd.DataFrame(
            columns=["property", "date", "value", "zscore", "category", "rationale", "removed"]
        )
    return pd.DataFrame([o.to_dict() for o in outliers])


# --------------------------------------------------------------------------
# 10.4 — Retraining configurable
# --------------------------------------------------------------------------

_RETRAIN_FREQ_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90,
    "yearly": 365,
    "never": None,
}


@dataclass
class RetrainingDecision:
    should_retrain: bool
    reason: str
    frequency: str
    new_records: int
    min_history_days: int

    def to_dict(self) -> dict:
        return {
            "should_retrain": self.should_retrain,
            "reason": self.reason,
            "frequency": self.frequency,
            "new_records": self.new_records,
            "min_history_days": self.min_history_days,
        }


def should_retrain(
    *,
    frequency: str,
    new_records: int,
    days_since_last: int,
    min_history_days: int,
    current_history_days: int,
    min_new_records: int = 1,
) -> RetrainingDecision:
    """10.4 — Decide si re-entrena; si no, documenta el motivo."""
    freq = str(frequency).lower()
    if freq == "never":
        return RetrainingDecision(
            should_retrain=False,
            reason="RETRAIN_FREQUENCY=never: retraining deshabilitado por configuración",
            frequency=freq,
            new_records=new_records,
            min_history_days=min_history_days,
        )

    interval = _RETRAIN_FREQ_DAYS.get(freq)
    if interval is None:
        return RetrainingDecision(
            should_retrain=False,
            reason=f"frecuencia desconocida '{frequency}'; no se re-entrena",
            frequency=freq,
            new_records=new_records,
            min_history_days=min_history_days,
        )

    if current_history_days < min_history_days:
        return RetrainingDecision(
            should_retrain=False,
            reason=(
                f"histórico insuficiente: {current_history_days}d < MIN_HISTORY "
                f"{min_history_days}d"
            ),
            frequency=freq,
            new_records=new_records,
            min_history_days=min_history_days,
        )

    if days_since_last < interval:
        return RetrainingDecision(
            should_retrain=False,
            reason=(
                f"frecuencia '{freq}' aún no vencida: {days_since_last}d "
                f"< {interval}d desde el último retraining"
            ),
            frequency=freq,
            new_records=new_records,
            min_history_days=min_history_days,
        )

    if new_records < min_new_records:
        return RetrainingDecision(
            should_retrain=False,
            reason=(
                f"sin información nueva suficiente: {new_records} registros nuevos "
                f"< mínimo {min_new_records}"
            ),
            frequency=freq,
            new_records=new_records,
            min_history_days=min_history_days,
        )

    return RetrainingDecision(
        should_retrain=True,
        reason=(
            f"retraining programado ({freq}): {days_since_last}d desde el último, "
            f"{new_records} registros nuevos"
        ),
        frequency=freq,
        new_records=new_records,
        min_history_days=min_history_days,
    )
