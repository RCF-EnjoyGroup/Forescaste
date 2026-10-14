"""Núcleo compartido del motor: normalización, curvas y candidatos.

Primitivas usadas por el runner end-to-end Y por la corrida real
(``run_real.py``): alias de propiedades, normalización de las tres fuentes al
canon del motor, curvas de pickup restante, grid de booked requests as-of, y
la construcción/operación de cada uno de los 7 candidatos con la MISMA
interfaz ``fit_candidate`` → ``predict_candidate``.

Remediación (spec forecasting): el campeón del benchmark opera el forecast
vía :func:`build_forecasts` — sin modelo de producción fijado en el código.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.dates import parse_dates_flexible
from src.data.schema import SourceSchema
from src.models.baselines import PickupCurveModel, SeasonalNaive
from src.models.candidates import (
    CatBoostModel,
    ETSModel,
    LightGBMModel,
    SARIMAXModel,
    XGBoostModel,
)

PORTFOLIO = "PORTFOLIO"
CANDIDATE_MODELS = [
    "seasonal_naive", "pickup_curve", "ets", "sarimax",
    "lightgbm", "xgboost", "catboost",
]
ML_FACTORIES = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
}
ML_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std_28",
    "day_of_week", "is_weekend", "month", "quarter",
    "current_booked_rn",
    "days_to_nearest_holiday",
]
STATS_TAIL_DAYS = 1095  # cola de entrenamiento ETS/SARIMAX (guard de costo)
ML_PANEL_MAX_STAYS = 365  # panel ML: últimas estancias con target observable
ML_MIN_TRAIN_OBS = 60
BACKTEST_LEADS_CHEAP = (90, 60, 30, 14, 7, 3, 1)
BACKTEST_LEADS_EXPENSIVE = (90, 30, 7, 1)
CHEAP_MODELS = {"pickup_curve", "seasonal_naive"}


# --------------------------------------------------------------------------
# normalización de propiedades y fuentes al canon del motor
# --------------------------------------------------------------------------

def _apply_aliases(
    df: pd.DataFrame, schema: SourceSchema, aliases: dict[str, str]
) -> pd.DataFrame:
    """Aplica alias de propiedad (config ``data.property_aliases``).

    Un mismo hotel puede aparecer con nombres distintos entre fuentes
    (p. ej. ``Hotel Royal Corin`` en demanda y ``CORIN`` en capacity);
    sin esto el cruce entre fuentes pierde filas silenciosamente.
    """
    if not aliases:
        return df
    out = df.copy()
    col = schema.property_col
    out[col] = out[col].astype(str).replace(aliases)
    return out


def _normalize_demand(df: pd.DataFrame, schema: SourceSchema) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "property": df[schema.property_col].astype(str),
            "stay_date": parse_dates_flexible(df[schema.date_col]),
            "room_nights": pd.to_numeric(df[schema.col("demand")], errors="coerce"),
        }
    )
    return out.dropna(subset=["stay_date"])


def _normalize_capacity(df: pd.DataFrame, schema: SourceSchema) -> pd.DataFrame:
    """Canon (property, stay_date, capacity).

    El capacity real llega en DD/MM/YYYY (estándar del negocio): el
    parser flexible evita descartar el ~60% de fechas por día > 12 y
    cruzar día/mes en las ambiguas (spec data-ingestion).
    """
    out = pd.DataFrame(
        {
            "property": df[schema.property_col].astype(str),
            "stay_date": parse_dates_flexible(df[schema.date_col]),
            "capacity": pd.to_numeric(df[schema.col("capacity")], errors="coerce"),
        }
    )
    return out.dropna(subset=["stay_date"])


def _normalize_pickup(df: pd.DataFrame, schema: SourceSchema) -> pd.DataFrame:
    prop = df[schema.property_col].astype(str)
    stay = parse_dates_flexible(df[schema.date_col])
    lead_col = schema.roles.get("lead")
    has_lead_col = bool(lead_col) and lead_col != "__derived__" and lead_col in df.columns
    lead = pd.to_numeric(df[lead_col], errors="coerce") if has_lead_col else None
    snap_col = schema.roles.get("snapshot")
    if snap_col and snap_col != "__derived__" and snap_col in df.columns:
        snap = parse_dates_flexible(df[snap_col])
    elif lead is not None:
        # snapshot derivado: la reserva se observó (stay - lead) días antes
        snap = stay - pd.to_timedelta(lead, unit="D")
    else:
        snap = stay
    if lead is None:
        lead = (stay - snap).dt.days
    out = pd.DataFrame(
        {
            "property": prop,
            "stay_date": stay,
            "snapshot_date": snap,
            "lead_days": lead,
            "pickup_rn": pd.to_numeric(df[schema.col("pickup")], errors="coerce"),
        }
    )
    return out.dropna(subset=["stay_date"])


def _remaining_pickup_curves(
    pick: pd.DataFrame, demand: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Curvas de entrenamiento con el pickup RESTANTE por lead.

    ``rooms_booked`` (pickup real) es on-the-books ACUMULADO: el total
    reservado a la fecha del snapshot. El modelo define
    ``forecast = booked + f(lead)``, de modo que ``f(lead)`` debe ser
    ``E[actual - booked | lead]``; si se entrenara con el booked acumulado
    el forecast saldría inflado (~2×).
    """
    hist = pick.loc[pick["snapshot_date"] <= cutoff].merge(
        demand.loc[
            demand["stay_date"] <= cutoff, ["property", "stay_date", "room_nights"]
        ],
        on=["property", "stay_date"],
        how="inner",
    )
    hist["pickup_rn"] = hist["room_nights"] - hist["pickup_rn"]
    return hist.dropna(subset=["lead_days", "pickup_rn"])


def _booked_requests(
    pick: pd.DataFrame,
    props: list[str],
    horizon_dates: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Grid del horizonte con el on-the-books REAL al momento del corte.

    Para cada estancia futura toma el snapshot más reciente ≤ corte (el de
    menor lead ≥ días hasta la estancia). Sin dato → ``NaN`` (cold start:
    el caller responde con el nivel medio histórico de la propiedad).
    """
    grid = pd.DataFrame(
        [(p, d) for p in props for d in horizon_dates],
        columns=["property", "stay_date"],
    )
    grid["lead_days"] = (grid["stay_date"] - cutoff).dt.days.clip(lower=1)
    cand = pick.merge(
        grid[["property", "stay_date", "lead_days"]].rename(
            columns={"lead_days": "lead_needed"}
        ),
        on=["property", "stay_date"],
    )
    cand = cand.loc[cand["lead_days"] >= cand["lead_needed"]]
    best = (
        cand.sort_values("lead_days")
        .groupby(["property", "stay_date"], as_index=False)
        .first()
    )
    out = grid.merge(
        best[["property", "stay_date", "pickup_rn"]],
        on=["property", "stay_date"],
        how="left",
    )
    return out.rename(columns={"pickup_rn": "current_booked_rn"})


# --------------------------------------------------------------------------
# primitivas de series y features ML (semántica calendario, sin target en features)
# --------------------------------------------------------------------------

def _daily_values(dem_p: pd.DataFrame, c: pd.Timestamp) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Serie diaria (reindexada a freq D) de una propiedad hasta el corte."""
    s = (
        dem_p.loc[dem_p["stay_date"] <= c]
        .groupby("stay_date")["room_nights"]
        .mean()
        .sort_index()
    )
    if s.empty:
        return pd.DatetimeIndex([]), np.asarray([], dtype=float)
    grid = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return grid, s.reindex(grid).to_numpy(dtype=float)


def _ml_row(
    vals: np.ndarray,
    t: int,
    d: pd.Timestamp,
    booked: float,
    holiday_calendar: Any = None,
) -> dict[str, float]:
    """Features as-of del índice ``t`` del grid diario (ventana [t-w, t-1]).

    El target de la propia fila JAMÁS entra a sus features: los rollings
    excluyen ``vals[t]`` (equivalente a ``shift(1)`` antes de la ventana).
    ``days_to_nearest_holiday`` es exógena y conocida a cualquier lead
    (feriados computados por calendario, no observados).
    """
    def lag(k: int) -> float:
        return float(vals[t - k]) if t - k >= 0 else float("nan")

    def roll(w: int, stat: str) -> float:
        seg = vals[max(0, t - w):t]
        if seg.size == 0 or np.all(np.isnan(seg)):
            return float("nan")
        if stat == "mean":
            return float(np.nanmean(seg))
        if stat == "std":
            return (
                float(np.nanstd(seg, ddof=1))
                if np.count_nonzero(~np.isnan(seg)) >= 2
                else float("nan")
            )
        raise ValueError(stat)

    return {
        "lag_1": lag(1),
        "lag_7": lag(7),
        "lag_14": lag(14),
        "lag_28": lag(28),
        "rolling_mean_7": roll(7, "mean"),
        "rolling_mean_14": roll(14, "mean"),
        "rolling_mean_28": roll(28, "mean"),
        "rolling_std_28": roll(28, "std"),
        "day_of_week": float(d.dayofweek),
        "is_weekend": float(d.dayofweek >= 5),
        "month": float(d.month),
        "quarter": float((d.month - 1) // 3 + 1),
        "current_booked_rn": float(booked),
        "days_to_nearest_holiday": (
            holiday_calendar.days_to_nearest(d) if holiday_calendar is not None
            else float("nan")
        ),
    }


def _booked_training_panel(
    pick_p: pd.DataFrame, dem_p: pd.DataFrame, c: pd.Timestamp, horizon: int,
) -> pd.DataFrame:
    """Panel (stay, lead, current_booked_rn) as-of con target observable.

    Una fila por (estancia, lead): la señal ``current_booked_rn`` es el
    on-the-books del snapshot ``stay - lead`` (≤ corte por construcción),
    exactamente la misma feature que opera en el grid del horizonte.
    """
    pk = pick_p.loc[
        (pick_p["snapshot_date"] <= c) & (pick_p["lead_days"] <= horizon)
    ]
    if pk.empty:
        return pd.DataFrame(columns=["stay_date", "lead_days", "current_booked_rn"])
    actuals = dem_p.loc[dem_p["stay_date"] <= c, ["property", "stay_date", "room_nights"]]
    panel = pk.merge(actuals, on=["property", "stay_date"], how="inner")
    panel = panel.dropna(subset=["lead_days", "pickup_rn", "room_nights"])
    panel = panel.drop_duplicates(["property", "stay_date", "lead_days"])
    last_stays = panel["stay_date"].sort_values().unique()
    if len(last_stays) > ML_PANEL_MAX_STAYS:
        keep = pd.DatetimeIndex(last_stays[-ML_PANEL_MAX_STAYS:])
        panel = panel.loc[panel["stay_date"].isin(keep)]
    return panel.rename(columns={"pickup_rn": "current_booked_rn"})[
        ["stay_date", "lead_days", "current_booked_rn"]
    ].reset_index(drop=True)


def _stats_predict(factory, series_obs: pd.Series, horizon: int) -> np.ndarray:
    """ETS/SARIMAX sobre la cola observada de la serie (posicional)."""
    tail = series_obs.tail(STATS_TAIL_DAYS)
    return np.asarray(factory().fit(tail).predict(horizon), dtype=float)


# --------------------------------------------------------------------------
# candidatos con interfaz común: fit en el corte, predict sobre fechas
# --------------------------------------------------------------------------

@dataclass
class FittedCandidate:
    """Un candidato entrenado as-of en un corte, listo para predecir."""

    model_name: str
    property: str
    fit_cutoff: pd.Timestamp
    horizon: int
    kind: str  # pickup | seasonal_naive | stats | ml
    artifact: Any = None
    level: float = 0.0  # cold start: nivel medio histórico
    grid: pd.DatetimeIndex | None = None
    vals: np.ndarray | None = None
    holiday_calendar: Any = None  # features.holidays.HolidayCalendar (ML)

    def cold_start(self) -> bool:
        return self.artifact is None


def _fit_pickup(
    dem_p: pd.DataFrame,
    pick_p: pd.DataFrame,
    c: pd.Timestamp,
    horizon: int,
    prop: str,
    curve_params: dict | None = None,
    holiday_calendar: Any = None,
) -> FittedCandidate:
    hist = dem_p.loc[dem_p["stay_date"] <= c]
    pkf = pick_p.loc[pick_p["snapshot_date"] <= c]
    curves = _remaining_pickup_curves(pkf, hist, c)
    curves = curves.loc[curves["lead_days"] <= horizon]
    fitted = FittedCandidate(
        model_name="pickup_curve", property=prop, fit_cutoff=c,
        horizon=horizon, kind="pickup",
        # nivel medio histórico: cold start por estancia (sin on-the-books)
        level=float(hist["room_nights"].mean()) if len(hist) else 0.0,
    )
    if curves.empty:
        return fitted
    fitted.artifact = PickupCurveModel(
        **(curve_params or {}), holiday_calendar=holiday_calendar
    ).fit(curves)
    return fitted


def _fit_stats(
    model_name: str, factory, dem_p: pd.DataFrame, c: pd.Timestamp, horizon: int, prop: str,
) -> FittedCandidate:
    grid, vals = _daily_values(dem_p, c)
    fitted = FittedCandidate(
        model_name=model_name, property=prop, fit_cutoff=c,
        horizon=horizon, kind="stats",
    )
    obs = pd.Series(vals, index=grid).dropna() if len(grid) else pd.Series(dtype=float)
    if len(obs) < ML_MIN_TRAIN_OBS:
        fitted.level = float(obs.mean()) if len(obs) else 0.0
        return fitted
    tail = obs.tail(STATS_TAIL_DAYS)
    fitted.artifact = factory().fit(tail)
    return fitted


def _fit_ml(
    factory, dem_p: pd.DataFrame, pick_p: pd.DataFrame,
    c: pd.Timestamp, horizon: int, prop: str, model_name: str,
    holiday_calendar: Any = None,
) -> FittedCandidate:
    grid, vals = _daily_values(dem_p, c)
    fitted = FittedCandidate(
        model_name=model_name, property=prop, fit_cutoff=c,
        horizon=horizon, kind="ml", grid=grid, vals=vals,
        holiday_calendar=holiday_calendar,
    )
    panel = _booked_training_panel(pick_p, dem_p, c, horizon)
    pos = {d: i for i, d in enumerate(grid)}
    train_rows: list[dict[str, float]] = []
    if vals.size:
        for stay, lead, booked in panel.itertuples(index=False):
            t = pos.get(pd.Timestamp(stay))
            if t is None or np.isnan(vals[t]):
                continue
            row = _ml_row(vals, t, pd.Timestamp(stay), float(booked), holiday_calendar)
            row["room_nights"] = float(vals[t])
            train_rows.append(row)
    if len(train_rows) < ML_MIN_TRAIN_OBS:
        fitted.level = (
            float(np.nanmean(vals)) if vals.size and not np.all(np.isnan(vals)) else 0.0
        )
        return fitted
    fitted.artifact = factory()
    fitted.artifact.fit(pd.DataFrame(train_rows), "room_nights", ML_FEATURES)
    return fitted


def fit_candidate(
    model_name: str,
    dem_p: pd.DataFrame,
    pick_p: pd.DataFrame,
    c: pd.Timestamp,
    horizon: int,
    curve_params: dict | None = None,
    holiday_calendar: Any = None,
) -> FittedCandidate:
    """Entrena UN candidato para UNA propiedad con datos ≤ corte.

    ``curve_params`` configura la curva de pickup (temporada del stay,
    ventana de recencia, umbral de fallback) desde el ``config.yaml``.
    ``holiday_calendar`` (primitiva compartida de ``features.holidays``)
    activa el condicionamiento por ventana de feriado en la curva y la
    feature ``days_to_nearest_holiday`` en los candidatos ML.
    """
    c = pd.Timestamp(c)
    prop = str(dem_p["property"].iloc[0]) if len(dem_p) else PORTFOLIO
    if model_name == "pickup_curve":
        return _fit_pickup(dem_p, pick_p, c, horizon, prop, curve_params, holiday_calendar)
    if model_name == "seasonal_naive":
        hist = dem_p.loc[dem_p["stay_date"] <= c]
        fitted = FittedCandidate(
            model_name="seasonal_naive", property=prop, fit_cutoff=c,
            horizon=horizon, kind="seasonal_naive",
        )
        if len(hist) < ML_MIN_TRAIN_OBS:
            fitted.level = float(hist["room_nights"].mean()) if len(hist) else 0.0
            return fitted
        fitted.artifact = SeasonalNaive().fit(hist)
        return fitted
    if model_name == "ets":
        return _fit_stats("ets", ETSModel, dem_p, c, horizon, prop)
    if model_name == "sarimax":
        return _fit_stats("sarimax", SARIMAXModel, dem_p, c, horizon, prop)
    if model_name in ML_FACTORIES:
        return _fit_ml(
            ML_FACTORIES[model_name], dem_p, pick_p, c, horizon, prop, model_name,
            holiday_calendar,
        )
    raise ValueError(f"candidato desconocido: '{model_name}'")


def _booked_lookup(
    pick_p: pd.DataFrame, prop: str, c: pd.Timestamp, dates: pd.DatetimeIndex,
) -> pd.Series:
    requests = _booked_requests(pick_p, [prop], dates, c)
    requests = requests.set_index("stay_date")
    return requests["current_booked_rn"].reindex(dates)


def predict_candidate(
    fitted: FittedCandidate,
    pick_p: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Predice ``dates`` con el candidato entrenado (as-of su corte de fit).

    Devuelve un frame indexado por ``stay_date`` con ``raw_forecast_rn``,
    ``expected_pickup_rn`` y ``current_booked_rn``.
    """
    dates = pd.DatetimeIndex(dates)
    prop = fitted.property
    booked = _booked_lookup(pick_p, prop, fitted.fit_cutoff, dates)

    if fitted.cold_start():
        raw = pd.Series(np.full(len(dates), max(fitted.level, 0.0)), index=dates)
    elif fitted.kind == "pickup":
        requests = _booked_requests(pick_p, [prop], dates, fitted.fit_cutoff)
        no_booked = requests["current_booked_rn"].isna().to_numpy()
        requests["current_booked_rn"] = requests["current_booked_rn"].fillna(0.0)
        fc = fitted.artifact.predict(requests)
        if no_booked.any():
            # cold start por estancia: nivel medio histórico de la propiedad
            fc.loc[no_booked, "forecast_rn"] = (
                fc.loc[no_booked, "forecast_rn"].fillna(fitted.level)
            )
        raw = pd.Series(
            np.maximum(fc["forecast_rn"].to_numpy(dtype=float), 0.0), index=dates
        )
    elif fitted.kind == "seasonal_naive":
        fc = fitted.artifact.predict(dates)
        raw = pd.Series(
            np.maximum(fc["forecast_rn"].to_numpy(dtype=float), 0.0), index=dates
        )
    elif fitted.kind == "stats":
        # posicional: válido con fechas contiguas desde el corte del fit
        steps = np.asarray([(d - fitted.fit_cutoff).days for d in dates], dtype=int)
        horizon = int(steps.max()) if len(steps) else 0
        preds = np.asarray(fitted.artifact.predict(max(horizon, 1)), dtype=float)
        raw_vals = [preds[min(s, len(preds)) - 1] for s in steps]
        raw = pd.Series(np.maximum(np.asarray(raw_vals, dtype=float), 0.0), index=dates)
    elif fitted.kind == "ml":
        ext_vals = np.concatenate(
            [np.asarray(fitted.vals, dtype=float), np.full(len(dates), np.nan)]
        )
        base = len(fitted.grid)
        preds: list[float] = []
        for j, d in enumerate(dates):
            t = base + j
            booked_j = float(booked.iloc[j]) if pd.notna(booked.iloc[j]) else float("nan")
            row = _ml_row(ext_vals, t, d, booked_j, fitted.holiday_calendar)
            yhat = max(float(fitted.artifact.predict(pd.DataFrame([row]))[0]), 0.0)
            preds.append(yhat)
            ext_vals[t] = yhat
        raw = pd.Series(preds, index=dates)
    else:  # pragma: no cover - contrato interno
        raise ValueError(f"kind desconocido: {fitted.kind}")

    out = pd.DataFrame(
        {
            "raw_forecast_rn": raw,
            "current_booked_rn": booked.to_numpy(dtype=float),
        },
        index=dates,
    )
    out.index.name = "stay_date"
    out["expected_pickup_rn"] = out["raw_forecast_rn"] - out["current_booked_rn"].fillna(0.0)
    return out.reset_index()


# blend del campeón ML a largo lead: el rollout recursivo degrada en
# leads largos (predicciones se realimentan como lags). Split en el mismo
# umbral que el composite ponderado por lead.
BLEND_LEAD_SPLIT = 30
# umbral de colapso del rollout: si el cuartil 25 de las predicciones a
# largo lead cae por debajo de RATIO × el de corto lead, la cascada de
# realimentación está destruyendo los valles de día-de-semana
BLEND_COLLAPSE_RATIO = 0.85
# margen de mejora material que exige el reemplazo largo: el candidato
# alternativo debe superar al propio campeón por al menos 5% en el
# wmape_long de los folds. Sin margen, un empalme marginal no justifica
# la heterogeneidad de modelos dentro del horizonte (spec forecasting:
# el reemplazo se elige por evidencia, no por defecto)
BLEND_ACCURACY_MARGIN = 0.95
# perfil DOW para reemplazos planos: el nivel del mejor modelo largo
# (stats, plano a 60-90 pasos) se multiplica por el ritmo semanal de la
# propiedad — media por día-de-semana de los últimos 12 semanas as-of,
# normalizada a media 1. El nivel viene de la evidencia; el patrón, de
# la historia (conocible a cualquier lead, sin lookahead)
DOW_PROFILE_DAYS = 84
DOW_MIN_OBS_PER_DOW = 3


def _dow_profile(
    dem_p: pd.DataFrame,
    cutoff: pd.Timestamp,
    days: int = DOW_PROFILE_DAYS,
    min_obs_per_dow: int = DOW_MIN_OBS_PER_DOW,
) -> pd.Series | None:
    """Perfil DOW normalizado (media 1) de la historia reciente as-of.

    El ritmo semanal de una propiedad es un hecho operativo conocible a
    cualquier lead desde historia ≤ corte. Devuelve ``None`` si la
    historia no alcanza para un perfil estable: el caller deja el
    segmento plano y lo declara en el manifiesto.
    """
    if dem_p is None or dem_p.empty:
        return None
    cutoff = pd.Timestamp(cutoff)
    hist = dem_p.loc[
        (dem_p["stay_date"] > cutoff - pd.Timedelta(days=int(days)))
        & (dem_p["stay_date"] <= cutoff),
        ["stay_date", "room_nights"],
    ].dropna(subset=["room_nights"])
    if len(hist) < min_obs_per_dow * 7:
        return None
    g = hist.groupby(hist["stay_date"].dt.dayofweek)["room_nights"]
    counts = g.size()
    # los 7 dias presentes y con suficientes obs: un dow ausente no
    # aparece en counts y el map posterior daria NaN
    if set(counts.index) != set(range(7)) or bool((counts < min_obs_per_dow).any()):
        return None
    profile = g.mean()
    media = float(profile.mean())
    if not np.isfinite(media) or media <= 0:
        return None
    return profile / media


def _select_long_replacement(
    prop: str,
    champion: str,
    fold_metrics: pd.DataFrame | None,
) -> tuple[str | None, str]:
    """Elige el reemplazo largo por EVIDENCIA de folds, no por defecto.

    El colapso del rollout es un problema de distribución de predicciones;
    la calidad a largo lead de cada candidato ya está medida por
    ``real_fold_metrics`` (mismo pipeline fit/predict, rollout incluido)
    en ``wmape_long``. Gana el candidato con mejor ``wmape_long`` medio de
    la propiedad — solo si supera materialmente al propio campeón
    (``wmape_long_alt < wmape_long_campeón × BLEND_ACCURACY_MARGIN``).

    Returns
    -------
    (modelo, razón) — modelo None cuando no hay evidencia o nada califica;
    la razón siempre explica la decisión con los números.
    """
    if fold_metrics is None or len(fold_metrics) == 0:
        return None, "sin fold_metrics (sin evidencia no hay reemplazo)"
    needed = {"property", "model", "wmape_long"}
    if not needed <= set(fold_metrics.columns):
        return None, "fold_metrics sin wmape_long (sin evidencia no hay reemplazo)"

    prop_rows = fold_metrics.loc[fold_metrics["property"] == prop]
    champ_rows = prop_rows.loc[
        (prop_rows["model"] == champion) & prop_rows["wmape_long"].notna()
    ]
    alt_rows = prop_rows.loc[
        (prop_rows["model"] != champion) & prop_rows["wmape_long"].notna()
    ]
    if champ_rows.empty:
        return None, f"campeón '{champion}' sin wmape_long en los folds"
    if alt_rows.empty:
        return None, "sin candidatos alternativos con wmape_long"

    champ_long = float(champ_rows["wmape_long"].mean())
    ranking = alt_rows.groupby("model")["wmape_long"].mean().sort_values()
    best_model = str(ranking.index[0])
    best_long = float(ranking.iloc[0])
    threshold = champ_long * BLEND_ACCURACY_MARGIN
    if best_long < threshold:
        return best_model, (
            f"mejor wmape_long: {best_model}={best_long:.3f} < "
            f"{champion}={champ_long:.3f}×{BLEND_ACCURACY_MARGIN}={threshold:.3f}"
        )
    return None, (
        f"ningún candidato supera el margen: mejor {best_model}={best_long:.3f} "
        f">= {champion}={champ_long:.3f}×{BLEND_ACCURACY_MARGIN}={threshold:.3f}"
    )


def build_forecasts(
    model_by_property: dict[str, str],
    dem: pd.DataFrame,
    pick: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int,
    curve_params: dict | None = None,
    holiday_calendar: Any = None,
    fold_metrics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, FittedCandidate], dict]:
    """Despacha el forecast del horizonte con el modelo ganador por propiedad.

    Auto-diagnóstico del rollout: cuando el campeón es un GBM y su
    predicción a largo lead colapsa (q25 largo < ``BLEND_COLLAPSE_RATIO`` ×
    q25 corto), el rollout recursivo se está realimentando. El reemplazo de
    los leads > ``BLEND_LEAD_SPLIT`` se elige por evidencia: el candidato
    con mejor ``wmape_long`` en los folds de la propiedad, solo si es
    materialmente mejor que el propio campeón (spec forecasting). Sin
    evidencia de folds, el colapso se registra pero no se reemplaza.

    Parameters
    ----------
    fold_metrics:
        Tabla (model, fold, property, ..., wmape_long) del benchmark real;
        ``None`` en corridas de muestra → sin reemplazo (sin cirugía a
        ciegas).

    Returns
    -------
    (grid_frame, fitted_by_property, blend_by_property):
        ``blend_by_property``: {prop: {trigger, short_model, long_model,
        blend_split, applied, long_selection}} para propiedades con
        colapso detectado — ``applied`` distingue empalme real de campeón
        retenido.
    """
    cutoff = pd.Timestamp(cutoff)
    horizon_dates = pd.date_range(
        cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D"
    )
    frames: list[pd.DataFrame] = []
    fitted_by_property: dict[str, FittedCandidate] = {}
    blend_by_property: dict = {}

    for prop in sorted(model_by_property):
        model_name = model_by_property[prop]
        dem_p = dem.loc[dem["property"] == prop]
        pick_p = pick.loc[pick["property"] == prop]
        fitted = fit_candidate(
            model_name, dem_p, pick_p, cutoff, horizon_days, curve_params, holiday_calendar
        )
        fitted_by_property[prop] = fitted
        pred = predict_candidate(fitted, pick_p, horizon_dates)
        pred["property"] = prop
        pred["model_name"] = model_name
        pred["lead_days"] = (pred["stay_date"] - cutoff).dt.days.clip(lower=1)

        # blend auto-diagnóstico: si el rollout del GBM colapsa a largo
        # lead (cuartil inferior de las predicciones largas << el de las
        # cortas), la cascada de predicciones realimentadas está destruyendo
        # los valles de dia-de-semana. Se usa el percentil 25 (no la media)
        # porque el colapso se concentra en los valles (mar-jue) mientras
        # que los picos (vie-dom) quedan estables y diluirian la media.
        # El reemplazo largo se elige por EVIDENCIA (mejor wmape_long de
        # los folds), no por defecto — pickup_curve a ciegas resultó ser el
        # PEOR modelo largo de FIESTA (26.3% vs 11.3% del campeón que
        # "rescataba"): sobre-pronóstico de findes recortado al techo y
        # desplome de lunes-jueves.
        if model_name in ML_FACTORIES and not fitted.cold_start():
            short_vals = pred.loc[
                pred["lead_days"] <= BLEND_LEAD_SPLIT, "raw_forecast_rn"
            ]
            long_vals = pred.loc[
                pred["lead_days"] > BLEND_LEAD_SPLIT, "raw_forecast_rn"
            ]
            rollout_collapsed = False
            if len(short_vals) > 0 and len(long_vals) > 0:
                short_q25 = float(short_vals.quantile(0.25))
                long_q25 = float(long_vals.quantile(0.25))
                rollout_collapsed = long_q25 < short_q25 * BLEND_COLLAPSE_RATIO
            if rollout_collapsed:
                replacement, sel_reason = _select_long_replacement(
                    prop, model_name, fold_metrics
                )
                if replacement is not None:
                    fitted_alt = fit_candidate(
                        replacement, dem_p, pick_p, cutoff, horizon_days,
                        curve_params, holiday_calendar,
                    )
                    if fitted_alt.cold_start():
                        replacement = None
                        sel_reason = (
                            f"candidato con cold-start ({sel_reason}); campeón retenido"
                        )
                if replacement is not None:
                    pred_alt = predict_candidate(fitted_alt, pick_p, horizon_dates)
                    long_mask = pred["lead_days"] > BLEND_LEAD_SPLIT
                    if long_mask.any():
                        pred.loc[long_mask, "raw_forecast_rn"] = (
                            pred_alt.loc[long_mask, "raw_forecast_rn"].to_numpy()
                        )
                        pred.loc[long_mask, "model_name"] = (
                            f"{model_name}+{replacement}(blend>{BLEND_LEAD_SPLIT})"
                        )
                # patrón DOW para reemplazos planos: los modelos stats
                # convergen a nivel medio a largo horizonte (la estacional
                # AR decae); el nivel ganador × el ritmo semanal de la
                # propiedad restaura los picos sin tocar la selección
                long_pattern = "flat"
                if (
                    replacement is not None
                    and getattr(fitted_alt, "kind", "") == "stats"
                ):
                    profile = _dow_profile(dem_p, cutoff)
                    if profile is not None:
                        long_idx = pred.index[pred["lead_days"] > BLEND_LEAD_SPLIT]
                        dows = pd.Series(
                            pd.DatetimeIndex(pred.loc[long_idx, "stay_date"]).dayofweek,
                            index=long_idx,
                        )
                        pred.loc[long_idx, "raw_forecast_rn"] = (
                            pred.loc[long_idx, "raw_forecast_rn"]
                            * dows.map(profile)
                        )
                        long_pattern = f"dow_profile({DOW_PROFILE_DAYS}d)"
                blend_by_property[prop] = {
                    "short_model": model_name,
                    "long_model": replacement if replacement is not None else model_name,
                    "blend_split": BLEND_LEAD_SPLIT,
                    "applied": replacement is not None,
                    "long_pattern": long_pattern,
                    "trigger": (
                        f"rollout_collapse(q25_long={long_q25:.0f} < "
                        f"q25_short*{BLEND_COLLAPSE_RATIO}={short_q25 * BLEND_COLLAPSE_RATIO:.0f})"
                    ),
                    "long_selection": sel_reason,
                }

        frames.append(pred)

    if not frames:
        return (
            pd.DataFrame(columns=[
                "property", "stay_date", "lead_days", "current_booked_rn",
                "raw_forecast_rn", "expected_pickup_rn", "model_name",
            ]),
            fitted_by_property,
            blend_by_property,
        )
    grid = pd.concat(frames, ignore_index=True)
    return (
        grid[["property", "stay_date", "lead_days", "current_booked_rn",
              "raw_forecast_rn", "expected_pickup_rn", "model_name"]],
        fitted_by_property,
        blend_by_property,
    )
