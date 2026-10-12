"""Salidas de forecast (spec ``reporting``, tasks 9.1–9.4).

- 9.1 tabla final con las columnas obligatorias del spec.
- 9.2 columnas extendidas cuando la información lo permite.
- 9.3 dataset listo para Power BI (forecast / accuracy / pickup / modelo).
- 9.4 los 7 KPIs ejecutivos.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics

# spec reporting §26 — columnas obligatorias
REQUIRED_COLUMNS = [
    "property",
    "stay_date",
    "forecast_date",
    "lead_days",
    "current_booked_rn",
    "capacity",
    "raw_forecast_rn",
    "final_forecast_rn",
    "p10_forecast",
    "p50_forecast",
    "p90_forecast",
    "model_name",
    "model_version",
    "forecast_run_timestamp",
]

# columnas extendidas (solo cuando la información lo permite)
EXTENDED_COLUMNS = [
    "historical_same_day_rn",
    "expected_pickup_rn",
    "pickup_rate",
    "forecast_occupancy",
    "forecast_bias",
    "confidence_level",
]


def build_forecast_table(df: pd.DataFrame) -> pd.DataFrame:
    """9.1 — Construye la tabla final validando columnas obligatorias.

    Las columnas obligatorias ausentes se crean como NaN y se reporta en
    ``_missing_required``; el caller debe verificar. Las extendidas solo se
    agregan si existen en la entrada (spec: "cuando sea posible").
    """
    out = df.copy()
    if "stay_date" in out.columns:
        out["stay_date"] = pd.to_datetime(out["stay_date"], errors="coerce")

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    for col in missing:
        out[col] = np.nan

    present_extended = [c for c in EXTENDED_COLUMNS if c in df.columns]
    out["_missing_required"] = ",".join(missing) if missing else ""
    out["_present_extended"] = ",".join(present_extended)
    return out[REQUIRED_COLUMNS + present_extended + ["_missing_required", "_present_extended"]]


def validate_forecast_table(table: pd.DataFrame) -> list[str]:
    """Devuelve la lista de problemas (vacía = tabla conforme al spec)."""
    problems: list[str] = []
    for col in REQUIRED_COLUMNS:
        if col not in table.columns:
            problems.append(f"columna obligatoria ausente: {col}")
    if table.empty:
        problems.append("tabla de forecast vacía")
        return problems

    # nulos en columnas obligatorias no-nulas (excepto intervalos si el modelo no los tiene)
    always_required = [
        "property", "stay_date", "forecast_date", "lead_days",
        "capacity", "raw_forecast_rn", "final_forecast_rn",
        "model_name", "forecast_run_timestamp",
    ]
    for col in always_required:
        if col in table.columns and table[col].isna().any():
            n = int(table[col].isna().sum())
            problems.append(f"columna obligatoria con {n} nulos: {col}")

    # invariantes numéricas: piso de booked + techo de capacidad
    if {"raw_forecast_rn", "capacity", "final_forecast_rn"} <= set(table.columns):
        raw = pd.to_numeric(table["raw_forecast_rn"], errors="coerce")
        cap = pd.to_numeric(table["capacity"], errors="coerce")
        fin = pd.to_numeric(table["final_forecast_rn"], errors="coerce")
        if "current_booked_rn" in table.columns:
            booked = pd.to_numeric(table["current_booked_rn"], errors="coerce")
            # NaN en booked → sin piso → esperado = min(raw, cap)
            has_booked = np.isfinite(booked)
            floored = np.where(has_booked, np.maximum(raw, booked), raw)
            esperado = np.minimum(floored, cap)
        else:
            esperado = np.minimum(raw, cap)
        if not (fin <= cap + 1e-9).all():
            problems.append("final_forecast_rn excede la capacidad")
        if not np.allclose(fin, esperado, equal_nan=True):
            problems.append(
                "final_forecast_rn no coincide con min(max(raw, booked), capacity)"
            )
    return problems


# --------------------------------------------------------------------------
# 9.2 — presencia/ausencia de extendidas según disponibilidad
# --------------------------------------------------------------------------


def extended_presence(table: pd.DataFrame) -> dict[str, bool]:
    return {c: (c in table.columns) for c in EXTENDED_COLUMNS}


# --------------------------------------------------------------------------
# 9.3 — dataset Power BI
# --------------------------------------------------------------------------


POWERBI_AREAS = {
    "forecast": [
        "property", "stay_date", "final_forecast_rn", "actual_rn",
        "variance", "variance_pct", "forecast_occupancy", "capacity",
    ],
    "accuracy": [
        "property", "stay_date", "lead_band", "mape", "wmape", "mae",
        "rmse", "bias", "breakdown",
    ],
    "pickup": [
        "property", "stay_date", "current_booked_rn", "expected_pickup_rn",
        "pickup_pct", "lead_days",
    ],
    "model": [
        "model_name", "model_version", "is_champion", "is_challenger",
    ],
}


def build_powerbi_dataset(
    forecast: pd.DataFrame,
    accuracy: pd.DataFrame | None = None,
    pickup: pd.DataFrame | None = None,
    models: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """9.3 — Cuatro tablas planas, una por área del dashboard."""
    fc = forecast.copy()
    if "stay_date" in fc.columns:
        fc["stay_date"] = pd.to_datetime(fc["stay_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Variance / Variance % cuando existan reales
    if "actual_rn" in fc.columns:
        fc["variance"] = pd.to_numeric(fc["final_forecast_rn"], errors="coerce") - pd.to_numeric(
            fc["actual_rn"], errors="coerce"
        )
        fc["variance_pct"] = np.where(
            pd.to_numeric(fc["actual_rn"], errors="coerce") != 0,
            fc["variance"] / pd.to_numeric(fc["actual_rn"], errors="coerce").replace(0, np.nan),
            np.nan,
        )
    else:
        fc["variance"] = np.nan
        fc["variance_pct"] = np.nan

    forecast_area = fc[
        [c for c in POWERBI_AREAS["forecast"] if c in fc.columns]
    ].copy()
    for c in POWERBI_AREAS["forecast"]:
        if c not in forecast_area.columns:
            forecast_area[c] = np.nan

    out: dict[str, pd.DataFrame] = {"forecast": forecast_area}

    out["accuracy"] = _area_frame(accuracy, POWERBI_AREAS["accuracy"])

    pick = pickup.copy() if pickup is not None else pd.DataFrame()
    if not pick.empty and {"current_booked_rn", "expected_pickup_rn"} <= set(pick.columns):
        booked = pd.to_numeric(pick["current_booked_rn"], errors="coerce")
        final = booked + pd.to_numeric(pick["expected_pickup_rn"], errors="coerce")
        pick["pickup_pct"] = np.where(
            final != 0, pd.to_numeric(pick["expected_pickup_rn"], errors="coerce") / final.replace(0, np.nan),
            np.nan,
        )
    out["pickup"] = _area_frame(pick, POWERBI_AREAS["pickup"])

    out["model"] = _area_frame(models, POWERBI_AREAS["model"])
    return out


def _area_frame(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    if "stay_date" in out.columns:
        out["stay_date"] = pd.to_datetime(out["stay_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    return out[[c for c in columns]]


# --------------------------------------------------------------------------
# 9.4 — KPIs ejecutivos
# --------------------------------------------------------------------------


def executive_kpis(
    forecast: pd.DataFrame,
    *,
    actual: pd.DataFrame | None = None,
    previous_forecast: pd.DataFrame | None = None,
    expected_pickup_total: float | None = None,
) -> dict[str, float]:
    """Los 7 KPIs del spec (Forecast Room Nights … Forecast vs Previous)."""
    fc = pd.to_numeric(forecast["final_forecast_rn"], errors="coerce")

    kpis: dict[str, float] = {}
    kpis["forecast_room_nights"] = float(fc.sum())

    # Forecast Occupancy = RN / Capacity
    if "capacity" in forecast.columns:
        cap = pd.to_numeric(forecast["capacity"], errors="coerce")
        kpis["forecast_occupancy"] = (
            float(fc.sum() / cap.sum()) if cap.sum() else float("nan")
        )
        # Capacity Utilization: misma base con la capacidad disponible
        kpis["capacity_utilization"] = kpis["forecast_occupancy"]
    else:
        kpis["forecast_occupancy"] = float("nan")
        kpis["capacity_utilization"] = float("nan")

    # Accuracy WMAPE / MAE / Bias cuando hay reales
    if actual is not None and not actual.empty:
        merged = _align_actual(forecast, actual)
        m = compute_metrics(merged["_actual"], merged["_forecast"])
        kpis["forecast_accuracy_wmape"] = m.wmape
        kpis["forecast_accuracy_mape"] = m.mape
        kpis["forecast_bias"] = m.bias
        kpis["mape_excluded_zero_actual"] = float(m.n_mape_excluded)
    else:
        kpis["forecast_accuracy_wmape"] = float("nan")
        kpis["forecast_accuracy_mape"] = float("nan")
        kpis["forecast_bias"] = float("nan")
        kpis["mape_excluded_zero_actual"] = 0.0

    # Expected Pickup
    if expected_pickup_total is not None:
        kpis["expected_pickup"] = float(expected_pickup_total)
    elif "expected_pickup_rn" in forecast.columns:
        kpis["expected_pickup"] = float(
            pd.to_numeric(forecast["expected_pickup_rn"], errors="coerce").sum()
        )
    else:
        kpis["expected_pickup"] = float("nan")

    # Forecast vs Previous Forecast
    if previous_forecast is not None and not previous_forecast.empty:
        prev = pd.to_numeric(previous_forecast["final_forecast_rn"], errors="coerce").sum()
        kpis["forecast_vs_previous"] = float(fc.sum()) - float(prev)
        kpis["forecast_vs_previous_pct"] = (
            float((fc.sum() - prev) / prev) if prev else float("nan")
        )
    else:
        kpis["forecast_vs_previous"] = float("nan")
        kpis["forecast_vs_previous_pct"] = float("nan")

    return kpis


def _align_actual(forecast: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("property", "stay_date") if c in forecast.columns and c in actual.columns]
    f = forecast.copy()
    a = actual.copy()
    f["_forecast"] = pd.to_numeric(f["final_forecast_rn"], errors="coerce")
    if "room_nights" in a.columns:
        a["_actual"] = pd.to_numeric(a["room_nights"], errors="coerce")
    elif "actual_rn" in a.columns:
        a["_actual"] = pd.to_numeric(a["actual_rn"], errors="coerce")
    else:
        raise KeyError("actual debe tener 'room_nights' o 'actual_rn'")
    f["stay_date"] = pd.to_datetime(f["stay_date"], errors="coerce")
    a["stay_date"] = pd.to_datetime(a["stay_date"], errors="coerce")
    return f.merge(a[keys + ["_actual"]], on=keys, how="inner").dropna(subset=["_actual", "_forecast"])
