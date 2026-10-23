"""Tracking forecast-vs-real por vintage (spec ``monitoring``).

Cierra el lazo de producción: cuando los reales de una ventana futura se
materializan, el forecast **publicado** por la corrida anterior se puntúa
tal cual fue publicado (por vintage de ``forecast_date`` y banda de lead),
incluida la cobertura empírica de la banda P10–P90 que se publicó con él.

Esto es distinto del backtest interno: aquí se mide el artefacto que el
negocio vio, no una simulación. Alimenta además los KPIs
``forecast_vs_previous`` (revisión entre corridas consecutivas).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.breakdowns import interval_coverage, lead_band
from src.evaluation.metrics import compute_metrics

PUBLISHED_FORECAST_FILENAME = "future_forecast_dataset.csv"

# columnas del detalle forecast-vs-real por estancia (entregable CSV)
DETAIL_COLUMNS = [
    "property", "stay_date", "forecast_date", "lead_days", "model_name",
    "current_booked_rn", "p10_forecast", "published_forecast_rn",
    "p90_forecast", "actual_rn", "error_rn", "error_pct", "in_band",
    "published_run_id",
]


def _materialized(
    published: pd.DataFrame,
    actuals: pd.DataFrame,
    through: pd.Timestamp,
) -> pd.DataFrame | None:
    """Merge del forecast publicado con los reales YA materializados.

    Solo estancias ≤ ``through`` (con real) y POSTERIORES al corte del
    forecast (``forecast_date < stay_date``): lo anterior a su corte ya era
    historia cuando se publicó. Devuelve el frame con ``forecast``/``actual``
    numéricos y ``lead_days`` resuelto; ``None`` sin traslape evaluable.
    """
    if published is None or published.empty or actuals is None or actuals.empty:
        return None
    point_col = next(
        (c for c in ("final_forecast_rn", "p50_forecast") if c in published.columns),
        None,
    )
    if point_col is None:
        return None

    work = published.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work = work.loc[
        (work["stay_date"] <= pd.Timestamp(through)) & work["stay_date"].notna()
    ]
    if work.empty:
        return None

    act = actuals[["property", "stay_date", "room_nights"]].copy()
    act["stay_date"] = pd.to_datetime(act["stay_date"], errors="coerce")
    act = act.dropna(subset=["stay_date"]).rename(columns={"room_nights": "actual"})
    merged = work.merge(act, on=["property", "stay_date"], how="inner")
    # solo estancias POSTERIORES al corte de ese forecast: lo anterior a su
    # corte ya era historia cuando se publicó
    if "forecast_date" in merged.columns:
        merged = merged.loc[
            pd.to_datetime(merged["forecast_date"], errors="coerce")
            < merged["stay_date"]
        ]
    if merged.empty:
        return None

    merged["forecast"] = pd.to_numeric(merged[point_col], errors="coerce")
    merged["actual"] = pd.to_numeric(merged["actual"], errors="coerce")
    merged = merged.dropna(subset=["forecast", "actual"])
    if merged.empty:
        return None
    if "lead_days" not in merged.columns:
        merged["lead_days"] = 30
    merged["lead_days"] = (
        pd.to_numeric(merged["lead_days"], errors="coerce").fillna(30).astype(int)
    )
    return merged


def vintage_detail(
    published: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    through: pd.Timestamp,
    published_run_id: str | None = None,
) -> pd.DataFrame:
    """Detalle forecast-vs-real por estancia del forecast PUBLICADO.

    Es la base del entregable ``vintage_tracking_detail.csv``: para cada
    (propiedad, fecha de estancia) ya materializada, lo que se publicó
    (P50 y banda P10–P90, con lo reservado al momento de publicar) contra
    lo que realmente llegó — error en RN, error % y si el real cayó dentro
    de la banda publicada.

    Returns
    -------
    Frame con ``DETAIL_COLUMNS``; vacío cuando no hay estancias
    materializadas (se publica sin filas, nunca se inventa).
    """
    merged = _materialized(published, actuals, through)
    empty = pd.DataFrame(columns=DETAIL_COLUMNS)
    if merged is None:
        return empty

    out = pd.DataFrame(index=merged.index)
    for col in ("property", "stay_date", "forecast_date", "lead_days",
                "model_name", "current_booked_rn", "p10_forecast",
                "p90_forecast"):
        if col in merged.columns:
            out[col] = merged[col]
    out["published_forecast_rn"] = merged["forecast"]
    out["actual_rn"] = merged["actual"]
    out["error_rn"] = out["published_forecast_rn"] - out["actual_rn"]
    denom = out["actual_rn"].replace(0, np.nan).abs()
    out["error_pct"] = out["error_rn"] / denom
    if "p10_forecast" in out.columns and "p90_forecast" in out.columns:
        lo = pd.to_numeric(out["p10_forecast"], errors="coerce")
        hi = pd.to_numeric(out["p90_forecast"], errors="coerce")
        out["in_band"] = (out["actual_rn"] >= lo) & (out["actual_rn"] <= hi)
    else:
        out["in_band"] = pd.NA
    out["published_run_id"] = str(published_run_id or "")
    # columnas opcionales ausentes del dataset publicado → NA (el contrato
    # del detalle es estable aunque la fuente traiga menos columnas)
    for col in DETAIL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.dropna(subset=["property", "stay_date"])
    return out[DETAIL_COLUMNS].sort_values(
        ["property", "stay_date"]
    ).reset_index(drop=True)


def load_published_forecast(run_dir: str | Path) -> pd.DataFrame | None:
    """Lee el dataset de forecast publicado de una corrida anterior."""
    path = Path(run_dir) / PUBLISHED_FORECAST_FILENAME
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    if df.empty or "stay_date" not in df.columns or "property" not in df.columns:
        return None
    df["stay_date"] = pd.to_datetime(df["stay_date"], errors="coerce")
    return df


def score_vintage(
    published: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    through: pd.Timestamp,
) -> dict | None:
    """Puntúa el forecast publicado contra los reales materializados.

    Parameters
    ----------
    published:
        Dataset publicado (``future_forecast_dataset.csv``): columnas
        ``property``/``stay_date`` y el punto ``final_forecast_rn`` (o
        ``p50_forecast``) con ``p10_forecast``/``p90_forecast`` y
        ``lead_days``.
    actuals:
        Demanda normalizada (``property``/``stay_date``/``room_nights``).
    through:
        Fecha límite con reales (se evalúan solo las estancias ya
        materializadas; el resto del horizonte sigue siendo futuro).

    Returns
    -------
    dict con métricas globales, por banda de lead y cobertura de banda;
    ``None`` cuando el forecast anterior no tiene estancias materializadas.
    """
    merged = _materialized(published, actuals, through)
    if merged is None:
        return None

    report: dict = {
        "n_obs": int(len(merged)),
        "window": {
            "start": str(merged["stay_date"].min().date()),
            "end": str(merged["stay_date"].max().date()),
        },
        "global": compute_metrics(merged["actual"], merged["forecast"]).to_dict(),
        "by_lead_band": [],
    }
    # tracking signal (control de sesgo sistemático): Σ error con signo
    # contra la desviación media — errores simétricos → ~0; sesgo sostenido
    # en una dirección → crece (umbral clásico |TS| ≤ 4). El MAPE sin signo
    # no puede ver este patrón; la TS sí (es la estadística que habría
    # atrapado el ciclo presupuestario optimista 2025 de la industria).
    # Se calcula POR PROPIEDAD sobre la añada: la TS global escala con n
    # por construcción (indistinguible), la de cada hotel es el control
    # que la industria usa por departamento.
    err = (merged["forecast"] - merged["actual"]).to_numpy(dtype=float)
    mad = float(np.mean(np.abs(err))) if len(err) else 0.0
    report["global"]["tracking_signal"] = (
        float(np.sum(err) / mad) if mad > 0 else 0.0
    )
    by_property: list[dict] = []
    for prop, grp in merged.groupby("property", observed=True):
        e = (grp["forecast"] - grp["actual"]).to_numpy(dtype=float)
        mad_p = float(np.mean(np.abs(e))) if len(e) else 0.0
        m = compute_metrics(grp["actual"], grp["forecast"])
        by_property.append(
            {
                "property": str(prop),
                "n_obs": int(len(grp)),
                "wmape": m.wmape,
                "bias": m.bias,
                "tracking_signal": float(np.sum(e) / mad_p) if mad_p > 0 else 0.0,
            }
        )
    report["by_property"] = sorted(
        by_property, key=lambda r: abs(r["tracking_signal"]), reverse=True
    )
    for band, grp in merged.groupby(merged["lead_days"].map(lead_band)):
        report["by_lead_band"].append(
            {"lead_band": band, "n_obs": int(len(grp)),
             **compute_metrics(grp["actual"], grp["forecast"]).to_dict()}
        )
    report["by_lead_band"] = sorted(
        report["by_lead_band"], key=lambda r: r["lead_band"]
    )
    cov = interval_coverage(merged)
    report["interval_coverage"] = cov if cov is not None else {"note": "sin banda"}
    return report


def select_gradable_publication(
    records: list[dict],
    actuals: pd.DataFrame,
    *,
    through: pd.Timestamp,
    max_lookback: int = 5,
) -> tuple[dict, dict] | None:
    """La publicación operativa MÁS RECIENTE cuya añada ya tiene reales.

    Correr una futura con el corte en el último real deja su propio
    horizonte sin estancias materializadas — la añada graduable es la
    publicación ANTERIOR. Itera los registros de modo ``futuro`` del más
    reciente al más antiguo (máx. ``max_lookback``) y devuelve la primera
    con scoring no vacío: ``(record, score)``. ``None`` sin ninguna
    graduable (primera corrida futura del registro).

    El registro no incluye la corrida actual (se escribe al final), así
    que la iteración nunca se gradúa a sí misma.
    """
    futuros = sorted(
        (r for r in records if str(r.get("mode")) == "futuro"),
        key=lambda r: str(r.get("timestamp") or ""),
        reverse=True,
    )
    for record in futuros[:max(1, int(max_lookback))]:
        published = load_published_forecast(str(record.get("run_dir") or ""))
        if published is None:
            continue
        score = score_vintage(published, actuals, through=through)
        if score is not None:
            return record, score
    return None


def forecast_revision(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    value_col: str = "final_forecast_rn",
) -> dict | None:
    """Revisión entre corridas: delta del forecast sobre estancias comunes.

    Señal de confianza para operaciones: cuánto se movió el pronóstico
    publicado entre la corrida anterior y la actual, sobre las mismas
    (propiedad, fecha de estancia) aún futuras.
    """
    if current is None or current.empty or previous is None or previous.empty:
        return None
    if value_col not in current.columns or value_col not in previous.columns:
        return None

    def _keys(df: pd.DataFrame) -> pd.DataFrame:
        out = df[["property", "stay_date", value_col]].copy()
        out["stay_date"] = pd.to_datetime(out["stay_date"], errors="coerce")
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
        return (
            out.dropna(subset=["stay_date", value_col])
            .groupby(["property", "stay_date"], as_index=False)[value_col]
            .sum()
        )

    cur, prev = _keys(current), _keys(previous)
    merged = cur.merge(prev, on=["property", "stay_date"], how="inner",
                      suffixes=("_cur", "_prev"))
    if merged.empty:
        return None
    total_cur = float(merged[f"{value_col}_cur"].sum())
    total_prev = float(merged[f"{value_col}_prev"].sum())
    if total_prev <= 0 or not np.isfinite(total_prev):
        return None
    return {
        "n_stays": int(len(merged)),
        "previous_total": total_prev,
        "current_total": total_cur,
        "delta": total_cur - total_prev,
        "delta_pct": (total_cur - total_prev) / total_prev,
    }
