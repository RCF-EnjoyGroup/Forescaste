"""Explicabilidad del forecast (spec ``explainability``, tasks 8.1–8.3).

- 8.1 explicaciones por predicción y globales para modelos ML
  (SHAP si está disponible; si no, importancia de features nativa).
- 8.2 la restricción de capacidad aparece como contribución negativa.
- 8.3 versión de negocio: reservas / pickup / estacionalidad / capacidad,
  legible para Revenue Managers; las contribuciones suman el forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Agrupación de features → driver de negocio (spec explainability)
DRIVER_GROUPS = {
    "current_bookings": ("current_booked_rn", "current_bookings", "booked"),
    "pickup": ("pickup", "lead", "expected_pickup"),
    "seasonality": ("day_of_week", "dow", "month", "season", "weekend", "quarter",
                    "week_of_year", "is_weekend", "lag_", "same_dow", "same_month"),
    "capacity": ("capacity", "occupancy", "inventory"),
    "trend": ("trend", "time", "index", "year"),
}


def classify_driver(feature: str) -> str:
    f = str(feature).lower()
    for driver, tokens in DRIVER_GROUPS.items():
        if any(tok in f for tok in tokens):
            return driver
    return "other"


@dataclass
class PredictionExplanation:
    """Explicación de UN forecast (property × stay date)."""

    property: str
    stay_date: str
    forecast_rn: float
    drivers: list[dict] = field(default_factory=list)
    base_value: float = 0.0
    method: str = "unknown"
    has_capacity_constraint: bool = False

    def top_drivers(self, n: int = 5) -> list[dict]:
        return sorted(self.drivers, key=lambda d: abs(d["contribution"]), reverse=True)[:n]

    def contributions_sum(self) -> float:
        return float(sum(d["contribution"] for d in self.drivers))

    def business_summary(self) -> list[str]:
        """8.3 — Versión legible para Revenue Management."""
        lines = [
            f"Forecast RN: {self.forecast_rn:.0f} ({self.property}, {self.stay_date})"
        ]
        for d in self.top_drivers(5):
            sign = "+" if d["contribution"] >= 0 else ""
            lines.append(f"- {d['business_label']}: {sign}{d['contribution']:.1f}")
        if self.has_capacity_constraint:
            cap = next((d for d in self.drivers if d["driver"] == "capacity_constraint"), None)
            if cap is not None:
                lines.append(f"- Restricción de capacidad: {cap['contribution']:.1f}")
        return lines

    def validate(self, tol: float = 1e-6) -> None:
        """Las contribuciones + base deben reconciliar con el forecast."""
        total = self.base_value + self.contributions_sum()
        if abs(total - self.forecast_rn) > max(tol, 1e-6 * max(1.0, abs(self.forecast_rn))):
            raise ValueError(
                f"contribuciones no suman el forecast: base={self.base_value:.4f} "
                f"+ Σ={self.contributions_sum():.4f} ≠ forecast={self.forecast_rn:.4f}"
            )


def explain_from_importance(
    *,
    property: str,
    stay_date: str,
    forecast_rn: float,
    feature_values: dict[str, float],
    feature_importance: dict[str, float],
    base_value: float = 0.0,
    capacity: float | None = None,
    raw_forecast: float | None = None,
    method: str = "feature_importance",
) -> PredictionExplanation:
    """8.1/8.2/8.3 — Construye una explicación a partir de importancias.

    La contribución de cada feature es ``importancia_normalizada * valor``
    escalado para reconciliar con ``forecast - base`` (spec: las
    contribuciones deben sumar el forecast reportado).
    """
    if not feature_importance:
        raise ValueError("sin importancia de features: no se puede explicar")

    drivers = []
    for feat, imp in feature_importance.items():
        val = float(feature_values.get(feat, 0.0))
        drivers.append(
            {
                "feature": feat,
                "importance": float(imp),
                "value": val,
                "raw_contribution": float(imp) * val,
                "driver": classify_driver(feat),
            }
        )

    raw_sum = float(sum(d["raw_contribution"] for d in drivers))
    target_sum = float(forecast_rn) - float(base_value)
    scale = (target_sum / raw_sum) if raw_sum != 0 else 0.0
    for d in drivers:
        d["contribution"] = d["raw_contribution"] * scale

    # 8.2 — restricción de capacidad como contribución explícita
    has_constraint = False
    if capacity is not None and raw_forecast is not None:
        raw = float(raw_forecast)
        if raw > float(capacity):
            has_constraint = True
            # el forecast reportado YA está capado: la contribución es la
            # diferencia negativa que explica raw -> final
            drivers.append(
                {
                    "feature": "__capacity_constraint__",
                    "importance": 1.0,
                    "value": float(capacity),
                    "raw_contribution": -(raw - float(capacity)),
                    "contribution": -(raw - float(capacity)),
                    "driver": "capacity_constraint",
                }
            )
            # re-escalar las features para que Σ + constraint = forecast
            others = [d for d in drivers if d["driver"] != "capacity_constraint"]
            remaining = target_sum + (raw - float(capacity))
            o_sum = float(sum(d["raw_contribution"] for d in others))
            o_scale = (remaining / o_sum) if o_sum != 0 else 0.0
            for d in others:
                d["contribution"] = d["raw_contribution"] * o_scale

    BUSINESS_LABELS = {
        "current_bookings": "Reservas actuales",
        "pickup": "Pickup histórico esperado",
        "seasonality": "Patrón estacional / día de semana",
        "capacity": "Capacidad disponible",
        "trend": "Tendencia",
        "capacity_constraint": "Restricción de capacidad",
        "other": "Otras features",
    }
    for d in drivers:
        d["business_label"] = BUSINESS_LABELS.get(d["driver"], "Otras features")

    expl = PredictionExplanation(
        property=property,
        stay_date=str(stay_date),
        forecast_rn=float(forecast_rn),
        drivers=drivers,
        base_value=float(base_value),
        method=method,
        has_capacity_constraint=has_constraint,
    )
    expl.validate()
    return expl


def explain_with_shap(
    model,
    X: pd.DataFrame,
    *,
    property: str,
    stay_date: str,
    forecast_rn: float,
    feature_names: list[str] | None = None,
    background: pd.DataFrame | None = None,
) -> PredictionExplanation:
    """8.1 — SHAP por predicción (TreeExplainer si el modelo lo soporta)."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "SHAP no está instalado; usar explain_from_importance con la "
            "importancia nativa del modelo"
        ) from exc

    names = feature_names or list(X.columns)
    if hasattr(model, "_model"):
        booster = model._model
    else:  # pragma: no cover
        booster = model

    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)
    except Exception:
        # modelos sin soporte TreeExplainer: fallback a importancia nativa
        if hasattr(model._model, "feature_importances_"):
            imp = dict(zip(names, map(float, model._model.feature_importances_)))
            return explain_from_importance(
                property=property,
                stay_date=stay_date,
                forecast_rn=forecast_rn,
                feature_values={n: float(X.iloc[0][n]) for n in names if n in X.columns},
                feature_importance=imp,
                method="feature_importance_fallback",
            )
        raise

    sv = np.asarray(shap_values)
    if sv.ndim == 3:  # multi-output
        sv = sv[0]
    row = sv[0] if len(sv.shape) > 1 else sv

    raw = {n: float(v) for n, v in zip(names, row)}
    base = float(getattr(explainer, "expected_value", 0.0))
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[0])

    expl = explain_from_importance(
        property=property,
        stay_date=stay_date,
        forecast_rn=forecast_rn,
        feature_values={n: float(X.iloc[0][n]) for n in names if n in X.columns},
        feature_importance={n: abs(v) for n, v in raw.items()},
        base_value=base,
        method="shap",
    )
    # SHAP: usar los valores SHAP directos como contribuciones. Por
    # construcción de SHAP, Σ shap + expected_value = predicción; al no tener
    # la predicción exacta del booster aquí, escalamos para reconciliar con
    # forecast (spec: las contribuciones deben sumar el forecast reportado).
    total_raw = float(sum(raw.values()))
    for d in expl.drivers:
        v = raw.get(d["feature"], 0.0)
        d["contribution"] = v * ((forecast_rn - base) / total_raw) if total_raw else 0.0
    expl.validate()
    return expl


def global_importance(explanations: list[PredictionExplanation]) -> pd.DataFrame:
    """8.1 — Importancia global agregada sobre muchas explicaciones."""
    if not explanations:
        return pd.DataFrame(columns=["feature", "driver", "mean_abs_contribution", "n"])
    rows: dict[str, dict] = {}
    for e in explanations:
        for d in e.drivers:
            r = rows.setdefault(
                d["feature"],
                {"feature": d["feature"], "driver": d["driver"], "_sum": 0.0, "n": 0},
            )
            r["_sum"] += abs(d["contribution"])
            r["n"] += 1
    out = pd.DataFrame(rows.values())
    out["mean_abs_contribution"] = out["_sum"] / out["n"]
    return out.drop(columns=["_sum"]).sort_values(
        "mean_abs_contribution", ascending=False
    ).reset_index(drop=True)
