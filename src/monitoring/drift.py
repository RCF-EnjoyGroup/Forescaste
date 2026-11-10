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

from src.data.dates import parse_dates_flexible

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
    "mape": {"warning": 0.10, "alert": 0.25},  # relativo
    "wmape": {"warning": 0.10, "alert": 0.25},
    "bias": {"warning": 0.05, "alert": 0.10},  # absoluto (puntos)
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


def population_stability_index(
    current: pd.Series, reference: pd.Series, n_bins: int = 5
) -> float:
    """PSI (Population Stability Index) entre referencia y ventana actual.

    Bins de igual probabilidad sobre la referencia → PSI = Σ
    (p_cur − p_ref)·ln(p_cur/p_ref). Diseñado para muestras diarias chicas
    (una ventana de 90d):

    - **5 bins** (no 10): el PSI nulo escala con (k−1)·(1/n₁ + 1/n₂)/2;
      con deciles y n=90 la media nula ya era 0.23 — todo alertaba.
    - **suavizado de conteos** (máx(count, 0.5)/n): un bin vacío por
      azar no dispara ln(p curvo/ε) — remedio estándar de PSI.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) < 2 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        # referencia (casi) constante: cualquier desviación de nivel es drift
        return 0.0 if np.allclose(cur.mean(), ref.mean()) else float("inf")
    edges = edges.tolist()
    edges[0], edges[-1] = -np.inf, np.inf
    idx_ref = np.clip(np.digitize(ref, edges[1:-1], right=True), 0, len(edges) - 2)
    idx_cur = np.clip(np.digitize(cur, edges[1:-1], right=True), 0, len(edges) - 2)
    n_b = len(edges) - 1
    # suavizado 0.5: cuenta vacía → 0.5/n (estándar en monitoreo de PSI)
    c_ref = np.maximum(np.bincount(idx_ref, minlength=n_b), 0.5) / len(ref)
    c_cur = np.maximum(np.bincount(idx_cur, minlength=n_b), 0.5) / len(cur)
    return float(np.sum((c_cur - c_ref) * np.log(c_cur / c_ref)))


def demand_drift_alerts(
    history: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    current_window_days: int = 90,
    psi_warning: float = 0.30,
    psi_alert: float = 0.50,
    property_col: str = "property",
    date_col: str = "stay_date",
    value_col: str = "room_nights",
) -> tuple[list[DriftAlert], dict]:
    """Drift de demanda POR PROPIEDAD contra la misma ventana del año previo.

    La comparación "primera mitad del histórico vs últimos 90d" dispara
    SIEMPRE: el portafolio creció de 1-2 hoteles a 6 (medias 51 → 560 RN),
    así que la alerta era ruido puro (fatiga). La referencia honesta es
    estacionalmente comparable: la misma ventana calendario un año antes
    (YoY), por propiedad. PSI (escala libre) en lugar de shift de medias.

    Propiedades con < 1 año de histórico no tienen ventana YoY: se
    documentan en ``skipped`` (sin alerta, sin inventar referencia).
    """
    cutoff = pd.Timestamp(cutoff)
    work = history.copy()
    work[date_col] = parse_dates_flexible(work[date_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    cur_start = cutoff - pd.Timedelta(days=current_window_days)
    ref_start = cur_start - pd.Timedelta(days=365)
    ref_end = cutoff - pd.Timedelta(days=365)

    alerts: list[DriftAlert] = []
    skipped: dict[str, str] = {}
    for prop, grp in work.groupby(property_col, dropna=False):
        s = (
            grp.dropna(subset=[date_col, value_col])
            .groupby(date_col)[value_col]
            .sum()
            .sort_index()
        )
        if s.empty:
            continue
        cur = s.loc[(s.index > cur_start) & (s.index <= cutoff)]
        ref = s.loc[(s.index > ref_start) & (s.index <= ref_end)]
        if ref.empty:
            skipped[str(prop)] = (
                "sin ventana YoY (< 365d de histórico): drift no evaluable"
            )
            continue
        if len(cur) < 14:
            skipped[str(prop)] = f"ventana actual con < 14 días con datos"
            continue
        psi = population_stability_index(cur, ref)
        if not np.isfinite(psi):
            sev: Severity = "alert"
        elif psi >= psi_alert:
            sev = "alert"
        elif psi >= psi_warning:
            sev = "warning"
        else:
            continue
        alerts.append(
            DriftAlert(
                metric=f"demand_psi:{prop}",
                severity=sev,
                message=(
                    f"Drift de demanda en {prop}: PSI YoY {psi:.3f} "
                    f"(ventana {current_window_days}d vs misma ventana hace 1 año)"
                ),
                reference=float(ref.mean()),
                current=float(cur.mean()),
            )
        )
    return alerts, skipped


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
    rel = (
        (cur.loc[common] - ref.loc[common]).abs() / ref.loc[common].replace(0, np.nan)
    ).dropna()
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
        a = check_distribution_drift(
            reference_demand, current_demand, "demand_distribution"
        )
        if a:
            alerts.append(a)

    if reference_pickup is not None and current_pickup is not None:
        a = check_pickup_curve_drift(reference_pickup, current_pickup)
        if a:
            alerts.append(a)

    if feature_reference is not None and feature_current is not None:
        for col in feature_reference.columns:
            if col in feature_current.columns and pd.api.types.is_numeric_dtype(
                feature_reference[col]
            ):
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
    "data_error",  # error de datos
    "event",  # evento real
    "group",  # grupo
    "seasonality",  # temporada
    "structural",  # cambio estructural
    "unclassified",  # no clasificado aún
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


def _rolling_zscore_by_date(
    vals: pd.Series,
    dates: pd.DatetimeIndex,
    *,
    window_days: int = 57,
    min_periods: int = 14,
) -> pd.Series:
    """z-score contra baseline LOCAL (ventana rodante centrada por fecha).

    La media/σ de TODO el histórico distorsiona el z cuando la propiedad
    crece o cambia de régimen (MARINA 2023: su propio arranque de baja
    demanda quedaba flaggeado como 'evento' contra la media 2023-26). El
    baseline local compara cada día con su entorno de ±window_days/2:
    captura picos aislados sin castigar niveles de arranque o estación
    propios de la propiedad. Fallback al σ global solo cuando la ventana
    local es degenerada (plana con pico incluido).
    """
    s = pd.Series(
        pd.to_numeric(vals, errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates),
    ).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    if s.index.duplicated().any():
        s = s.groupby(level=0).mean()
    min_p = max(3, min(min_periods, len(s)))
    med = s.rolling(f"{window_days}D", center=True, min_periods=min_p).median()
    std = s.rolling(f"{window_days}D", center=True, min_periods=min_p).std()
    scale = std.where(std > 0)
    global_std = float(s.std()) if len(s) > 1 else 0.0
    if np.isfinite(global_std) and global_std > 0:
        scale = scale.fillna(global_std)
    return ((s - med) / scale).dropna()


def detect_outliers(
    df: pd.DataFrame,
    *,
    value_col: str = "room_nights",
    date_col: str = "stay_date",
    property_col: str = "property",
    capacity_col: str | None = "capacity",
    threshold: float = 3.0,
    window_days: int = 57,
) -> list[ClassifiedOutlier]:
    """10.3 — Detecta picos/caídas y los CLASIFICA (nunca los elimina).

    El z-score usa un baseline LOCAL por fecha (``window_days`` centrado)
    con fallback al σ de la propiedad — ver ``_rolling_zscore_by_date``.
    """
    out: list[ClassifiedOutlier] = []
    work = df.copy()
    work[date_col] = parse_dates_flexible(work[date_col])
    # orden cronológico dentro de cada propiedad: la clasificación
    # estructural (nivel que se mantiene tras el pico) asume secuencia
    work = work.sort_values([property_col, date_col], kind="stable")

    for prop, grp in work.groupby(property_col, dropna=False, sort=False):
        z_by_date = _rolling_zscore_by_date(
            grp[value_col], pd.DatetimeIndex(grp[date_col]), window_days=window_days
        )
        if z_by_date.empty:
            continue
        z = grp[date_col].map(z_by_date)
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
        if (
            pd.notna(row.get(capacity_col))
            and float(row[capacity_col]) < cap_mean * 0.7
        ):
            return (
                "structural",
                f"capacidad reducida ({float(row[capacity_col]):.0f} vs media {cap_mean:.0f})",
            )

    # fin de semana / temporada alta: patrón estacional conocido
    dow = date.day_of_week
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
        if (
            len(after) >= 3
            and zscore > 0
            and after.mean() > grp[value_col].astype(float).mean() * 1.15
        ):
            return (
                "structural",
                "el nivel se mantiene elevado después del pico (cambio estructural)",
            )
    except Exception:  # pragma: no cover - posicionamiento defensivo
        pass

    # evento: aislado (el entorno vuelve a la normalidad)
    return (
        "event",
        f"pico aislado |z|={abs(zscore):.1f} con entorno normal (evento real: se conserva)",
    )


def classified_frame(outliers: list[ClassifiedOutlier]) -> pd.DataFrame:
    if not outliers:
        return pd.DataFrame(
            columns=[
                "property",
                "date",
                "value",
                "zscore",
                "category",
                "rationale",
                "removed",
            ]
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
