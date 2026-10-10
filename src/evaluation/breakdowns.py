"""Desgloses de precisión (spec ``evaluation``, tasks 6.4/6.5).

- Por banda de lead: 90+, 60–89, 30–59, 14–29, 7–13, 3–6, 1–2 (cada
  observación cae en EXACTAMENTE una banda).
- Por propiedad: MAPE/WMAPE/MAE/RMSE/Bias por property.
- Por estacionalidad: mes, día de semana, weekend vs weekday, temporadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import MetricResult, compute_metrics

LEAD_BINS = [-np.inf, 2, 6, 13, 29, 59, 89, np.inf]
LEAD_LABELS = ["1-2", "3-6", "7-13", "14-29", "30-59", "60-89", "90+"]


def lead_band(lead_days) -> str:
    """Asigna una banda de lead a una observación (exactamente una)."""
    lead = float(lead_days)
    if lead <= 2:
        return "1-2"
    if lead <= 6:
        return "3-6"
    if lead <= 13:
        return "7-13"
    if lead <= 29:
        return "14-29"
    if lead <= 59:
        return "30-59"
    if lead <= 89:
        return "60-89"
    return "90+"


def _metrics_by(
    df: pd.DataFrame,
    group_cols: list[str],
    actual_col: str = "actual",
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    # observed=True evita grupos vacíos en columnas categóricas (p. ej. season)
    for keys, grp in df.groupby(grouper, dropna=False, observed=True):
        if grp.empty:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        if any(pd.isna(k) for k in keys):
            continue
        a = pd.to_numeric(grp[actual_col], errors="coerce")
        f = pd.to_numeric(grp[forecast_col], errors="coerce")
        if not (np.isfinite(a) & np.isfinite(f)).any():
            continue  # grupo sin pares válidos (no es un error: se omite)
        m = compute_metrics(grp[actual_col], grp[forecast_col])
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "mape": m.mape,
                "wmape": m.wmape,
                "mae": m.mae,
                "rmse": m.rmse,
                "bias": m.bias,
                "n_obs": m.n_obs,
                "n_mape_excluded": m.n_mape_excluded,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_lead_band(
    df: pd.DataFrame,
    lead_col: str = "lead_days",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    """6.4 — MAPE/WMAPE/Bias por banda de lead; partición exacta de obs."""
    work = df.copy()
    work["_lead_band"] = work[lead_col].map(lead_band)
    # partición exacta: toda observación cae en una banda conocida
    unknown = work["_lead_band"].isna()
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} observaciones con lead no clasificable")
    out = _metrics_by(work, ["_lead_band"], actual_col, forecast_col)
    out = out.rename(columns={"_lead_band": "lead_band"})
    out["lead_band"] = pd.Categorical(out["lead_band"], categories=LEAD_LABELS, ordered=True)
    # cobertura: la suma de n_obs coincide con el total
    assert int(out["n_obs"].sum()) == len(work), "las bandas no particionan las observaciones"
    return out.sort_values("lead_band").reset_index(drop=True)


def metrics_by_property(
    df: pd.DataFrame,
    property_col: str = "property",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    """6.5 — Tabla por propiedad + ranking de precisión."""
    out = _metrics_by(df, [property_col], actual_col, forecast_col)
    out = out.sort_values("wmape").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def property_ranking(table: pd.DataFrame) -> dict:
    """Mejores/peores propiedades y volatilidad del error."""
    if table.empty:
        return {}
    sorted_t = table.sort_values("wmape")
    return {
        "best": sorted_t.iloc[0]["property"],
        "worst": sorted_t.iloc[-1]["property"],
        "error_volatility": float(table["wmape"].std()) if len(table) > 1 else 0.0,
        "worst_bias": float(
            table.loc[table["wmape"].idxmax(), "bias"]
        ),
    }


def metrics_by_seasonality(
    df: pd.DataFrame,
    date_col: str = "stay_date",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
) -> dict[str, pd.DataFrame]:
    """6.5 — Desgloses: mes, día de semana, weekend/weekday, temporada."""
    work = df.copy()
    dates = pd.to_datetime(work[date_col], errors="coerce")
    work["_month"] = dates.dt.month
    work["_dow"] = dates.dt.dayofweek
    work["_weekend"] = np.where(work["_dow"] >= 5, "weekend", "weekday")
    work["_season"] = pd.Categorical(
        dates.dt.month.map(
            {12: "high", 1: "high", 2: "high", 3: "shoulder", 4: "shoulder",
             5: "shoulder", 6: "low", 7: "low", 8: "low", 9: "low",
             10: "shoulder", 11: "shoulder"}
        ),
        categories=["high", "shoulder", "low"],
        ordered=True,
    )
    return {
        "month": _metrics_by(work, ["_month"], actual_col, forecast_col).rename(
            columns={"_month": "month"}
        ),
        "day_of_week": _metrics_by(work, ["_dow"], actual_col, forecast_col).rename(
            columns={"_dow": "day_of_week"}
        ),
        "weekend_weekday": _metrics_by(work, ["_weekend"], actual_col, forecast_col).rename(
            columns={"_weekend": "day_type"}
        ),
        "season": _metrics_by(work, ["_season"], actual_col, forecast_col).rename(
            columns={"_season": "season"}
        ),
    }


def systematic_pattern(
    breakdown: pd.DataFrame,
    group_col: str,
    bias_threshold: float = 0.05,
    min_obs: int = 10,
) -> list[str]:
    """6.5 — Insights accionables: patrones de fallo sistemático detectados.

    Ejemplo: 'Weekend demand is consistently under-forecast by 12%.'
    """
    insights: list[str] = []
    if "n_obs" in breakdown.columns:
        breakdown = breakdown.loc[breakdown["n_obs"] >= min_obs]
    for _, row in breakdown.iterrows():
        bias = row.get("bias", np.nan)
        if not np.isfinite(bias) or abs(bias) < bias_threshold:
            continue
        direction = "sub-pronostica" if bias < 0 else "sobre-pronostica"
        group = row[group_col]
        insights.append(
            f"Grupo '{group}': {direction} sistemáticamente {abs(bias):.1%} "
            f"(bias={bias:+.3f}, n={int(row['n_obs'])})"
        )
    return insights
