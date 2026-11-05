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
    # lead de la estancia respecto al corte: exógeno, conocido a cualquier
    # lead sin fuga (aritmética de calendario). Sin él el GBM solo distingue
    # leads a través de current_booked_rn — una versión ruidosa de la misma
    # información — y no puede aprender la forma de la curva de pickup
    # (el pickup restante decae con el lead aunque el booked no cambie).
    "lead_days",
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
    lead: float | None = None,
) -> dict[str, float]:
    """Features as-of del índice ``t`` del grid diario (ventana [t-w, t-1]).

    El target de la propia fila JAMÁS entra a sus features: los rollings
    excluyen ``vals[t]`` (equivalente a ``shift(1)`` antes de la ventana).
    ``days_to_nearest_holiday`` es exógena y conocida a cualquier lead
    (feriados computados por calendario, no observados). ``lead`` es la
    anticipación de la estancia respecto al corte de fit — también exógena:
    en el panel de entrenamiento es el lead del snapshot y en el horizonte
    es ``stay - cutoff``; sin fuga en ninguno de los dos casos.
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
        "lead_days": float(lead) if lead is not None else float("nan"),
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
            row = _ml_row(
                vals, t, pd.Timestamp(stay), float(booked), holiday_calendar,
                lead=float(lead),
            )
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
            # lead as-of: anticipación de la estancia respecto al corte de
            # fit — la misma semántica del panel de entrenamiento (lead del
            # snapshot), conocida sin fuga
            lead_j = float((d - fitted.fit_cutoff).days)
            row = _ml_row(
                ext_vals, t, d, booked_j, fitted.holiday_calendar, lead=lead_j
            )
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


def _operative_forecast_frame(
    fold_predictions: pd.DataFrame | None,
    model_by_property: dict[str, str],
    blend_by_property: dict[str, dict],
    dem: pd.DataFrame,
) -> pd.DataFrame:
    """Ensambla el forecast operativo por fold (campeón + blend + DOW).

    Es el forecast que opera en producción reutilizando las predicciones
    que el benchmark ya computó (``real_fold_metrics`` las retiene), sin
    re-entrenar nada. La decisión de blend del corte T0 se aplica
    uniforme a los folds; el perfil DOW se calcula as-of del corte de
    CADA fold (el fold cutoff se recupera como ``stay_date - lead_days``).

    Returns
    -------
    Frame ``property/fold/stay_date/lead_days/forecast/actual`` — la base
    compartida de los residuos de la banda Y de la corrección de bias.
    """
    cols = ["property", "fold", "stay_date", "lead_days", "forecast", "actual"]
    if fold_predictions is None or len(fold_predictions) == 0:
        return pd.DataFrame(columns=cols)
    needed = {"property", "fold", "model", "stay_date", "lead_days", "forecast", "actual"}
    if not needed <= set(fold_predictions.columns):
        return pd.DataFrame(columns=cols)

    work = fold_predictions.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"])
    frames: list[pd.DataFrame] = []
    for prop, prop_rows in work.groupby("property"):
        champion = model_by_property.get(str(prop))
        if champion is None or champion not in set(prop_rows["model"]):
            continue
        base = prop_rows.loc[prop_rows["model"] == champion].set_index("stay_date")
        entry = blend_by_property.get(str(prop)) or {}
        long_model = entry.get("long_model") if entry.get("applied") else None
        split = int(entry.get("blend_split", BLEND_LEAD_SPLIT))
        use_profile = str(entry.get("long_pattern", "")).startswith("dow_profile")
        operative = base[["fold", "lead_days", "actual", "forecast"]].copy()
        if long_model and long_model in set(prop_rows["model"]):
            alt = prop_rows.loc[prop_rows["model"] == long_model].set_index("stay_date")
            long_dates = operative.index[operative["lead_days"] > split]
            operative.loc[long_dates, "forecast"] = (
                alt.loc[long_dates, "forecast"].to_numpy()
            )
            if use_profile:
                dem_p = dem.loc[dem["property"] == prop]
                # corte del fold de cada fecha: stay - lead
                fold_cutoffs = (
                    pd.Series(long_dates, index=long_dates)
                    - pd.to_timedelta(operative.loc[long_dates, "lead_days"], unit="D")
                )
                for fc in fold_cutoffs.dt.normalize().unique():
                    dates_fc = fold_cutoffs.index[fold_cutoffs == fc]
                    profile = _dow_profile(dem_p, pd.Timestamp(fc))
                    if profile is None:
                        continue
                    dows = pd.Series(
                        pd.DatetimeIndex(dates_fc).dayofweek, index=dates_fc
                    )
                    operative.loc[dates_fc, "forecast"] = (
                        operative.loc[dates_fc, "forecast"] * dows.map(profile)
                    )
        operative = operative.reset_index()
        operative["property"] = str(prop)
        frames.append(operative[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def operative_residual_bands(
    fold_predictions: pd.DataFrame | None,
    model_by_property: dict[str, str],
    blend_by_property: dict[str, dict],
    dem: pd.DataFrame,
    bias_factors: dict | None = None,
    recency_decay: float = 1.0,
    seasonal_cells: bool = False,
    fallback_factors: dict | None = None,
) -> dict:
    """Residuos del forecast OPERATIVO por (propiedad, banda de lead).

    Ensambla por fold el forecast que opera en producción — campeón con
    su blend (reemplazo largo) y su patrón DOW — a partir de las
    predicciones que el benchmark ya computó (``real_fold_metrics`` las
    retiene), sin re-entrenar nada, y emite sus residuos agrupados por
    (propiedad, banda). Los intervalos describen el error de lo que se
    publica, no el de los candidatos puros (spec forecasting).

    Con ``bias_factors`` (ver ``lead_bias_factors``) los residuos son los
    del forecast CORREGIDO: ``forecast·factor − actual`` — la banda calibra
    el pipeline completo que se publica (punto corregido + banda), no una
    versión intermedia.

    Con ``seasonal_cells=True`` la agrupación pasa a
    (propiedad, banda, temporada del stay) y el factor por fila se
    resuelve con la misma composición del punto (estacional aplicable →
    banda → 1.0): la banda calibra exactamente el pipeline que se
    publica, fila a fila.

    Con ``recency_decay < 1.0`` los residuos salen como DataFrame
    ``(residual, weight)`` con peso ``decay^(edad del fold)``: en un
    portafolio en transición de régimen, el trimestre reciente describe
    mejor el error vigente que hace un año — los cuantiles de la banda se
    ponderan (``build_intervals``). ``1.0`` emite pesos uniformes
    (equivalente al comportamiento sin ponderación).
    """
    from src.evaluation.breakdowns import lead_band, season_of_stay

    frame = _operative_forecast_frame(
        fold_predictions, model_by_property, blend_by_property, dem
    )
    if frame.empty:
        return {}
    work = frame.copy()
    work["lead_band"] = work["lead_days"].map(lead_band)
    if seasonal_cells:
        work["season"] = season_of_stay(work["stay_date"]).fillna("__sin_temporada__")
    # edad del fold: el más reciente (máximo índice) pesa 1.0, el anterior
    # decay, y así hacia atrás — los folds tesian el año previo al corte
    max_fold = int(work["fold"].max()) if "fold" in work.columns else 0
    decay = float(min(max(recency_decay, 0.0), 1.0))
    work["weight"] = decay ** (max_fold - pd.to_numeric(work["fold"], errors="coerce").fillna(0))
    work["weight"] = work["weight"].fillna(1.0)

    if seasonal_cells and bias_factors:
        # factor por fila con la MISMA composición que el punto: clave
        # estacional → fallback de banda → 1.0
        fkeys = list(bias_factors.keys())
        seasonal = all(isinstance(k, tuple) and len(k) == 3 for k in fkeys)
        band_facts = dict(fallback_factors or {}) if seasonal else {}

        def _factor_row(prop: str, band: str, season: str) -> float:
            if seasonal:
                f = bias_factors.get((prop, band, season))
                if f is not None:
                    return float(f)
                return float(band_facts.get((prop, band), 1.0))
            return float((bias_factors or {}).get((prop, band), 1.0))

        if seasonal:
            work["factor"] = [
                _factor_row(str(p), str(b), str(s))
                for p, b, s in zip(
                    work["property"].astype(str),
                    work["lead_band"].astype(str),
                    work["season"].astype(str), strict=True)
            ]
        else:
            work["factor"] = [
                _factor_row(str(p), str(b), "")
                for p, b in zip(work["property"].astype(str),
                                work["lead_band"].astype(str), strict=True)
            ]
    else:
        # sin celdas estacionales: un solo factor por (prop, banda)
        flat = {
            (str(p), str(b)): float(f) for (p, b), f in (bias_factors or {}).items()
        }
        work["factor"] = [
            float(flat.get((str(p), str(b)), 1.0))
            for p, b in zip(work["property"].astype(str),
                            work["lead_band"].astype(str), strict=True)
        ]

    group_cols = (
        ["property", "lead_band", "season"] if seasonal_cells
        else ["property", "lead_band"]
    )
    out: dict = {}
    for keys, grp in work.groupby(group_cols, observed=True):
        cell = pd.DataFrame({
            "residual": grp["forecast"] * grp["factor"] - grp["actual"],
            "weight": grp["weight"],
        }).dropna(subset=["residual"])
        if cell.empty:
            continue
        out[keys if len(keys) > 1 else keys[0]] = cell.reset_index(drop=True)
    return out


def lead_bias_factors(
    fold_predictions: pd.DataFrame | None,
    model_by_property: dict[str, str],
    blend_by_property: dict[str, dict],
    dem: pd.DataFrame,
    *,
    min_obs: int = 20,
    shrinkage: int = 30,
    max_rel: float = 0.30,
    min_correction: float = 0.005,
    min_consistency: float = 0.75,
    seasonal_cells: bool = False,
) -> tuple[dict, dict]:
    """Corrección de bias por (propiedad, banda de lead) — evidencia de folds.

    El backtest operativo documenta sobre-pronóstico sistemático a leads
    largos (+23% bias en 60-89, +30% en 90+): la convergencia estacional
    de la curva sobre-estima en temporada baja. La corrección se aprende
    de los residuos del forecast OPERATIVO (mismo ensamble que la banda):

    - ``rel_bias = Σ(forecast − actual) / Σ|actual|`` por celda →
      ``factor = 1 − rel_bias``;
    - **gate de consistencia**: el bias debe tener el mismo signo en ≥
      ``min_consistency`` de los folds — con 4 folds, el bias por celda
      oscila salvajemente entre ventanas (p. ej. +25/−17/−7/+55% en una
      celda real) y la media la domina un solo fold: sesgo inestable =
      ruido de régimen (p. ej. ``seasonal_naive`` retrasa la tendencia y
      su sesgo gira con la estación), NO evidencia para corregir;
    - **shrinkage** hacia 1.0 según el tamaño de muestra
      (``n/(n+shrinkage)``): con 4 folds, una celda chica no corrige a
      ciegas;
    - clip ``±max_rel`` (correcciones > 30% no son bias, son otro régimen);
    - celdas con < ``min_obs`` observaciones y correcciones < 0.5% no
      tocan el punto (ruido, no evidencia).

    Con ``seasonal_cells=True`` además se estima por
    **(propiedad, banda, temporada del stay)** (high/shoulder/low por
    calendario — dimensión conocible a-priori): la celda estacional lleva
    la MISMA disciplina de gates, y el factor final compone con fallback
    honesto (celda estacional aplicable → celda (prop, banda) → 1.0). El
    caso real que motiva esto: FIESTA pronosticaba −13.5% abajo en el
    régimen del horizonte mientras el gate por banda bloqueaba la
    corrección porque el sesgo gira ENTRE estaciones (consistencia 50%) —
    el sesgo dentro de UNA temporada sí es consistente. Los llaves del
    dict devuelto son 3-tuplas ``(prop, banda, temporada)`` y el reporte
    incluye ``band_factors`` (2-tuplas) como fallback documentado para
    combinaciones estacionales sin filas en los folds.

    Returns
    -------
    (factors, report): factores por (propiedad, banda) — o 3-tuplas con
    ``seasonal_cells`` — y detalle por celda (n, bias relativo,
    consistencia, factor) para el manifiesto.
    """
    from src.evaluation.breakdowns import lead_band, season_of_stay

    factors: dict = {}
    frame = _operative_forecast_frame(
        fold_predictions, model_by_property, blend_by_property, dem
    )
    if frame.empty:
        return factors, {"note": "sin predicciones de folds (sin corrección)"}
    work = frame.dropna(subset=["forecast", "actual"]).copy()
    work["lead_band"] = work["lead_days"].map(lead_band)

    cells: list[dict] = []
    band_factors: dict[tuple[str, str], float] = {}
    n_corrected = 0
    for (prop, band), grp in work.groupby(["property", "lead_band"], observed=True):
        result = _bias_cell_estimate(
            grp, min_obs=min_obs, shrinkage=shrinkage, max_rel=max_rel,
            min_correction=min_correction, min_consistency=min_consistency,
        )
        if result["applied"]:
            n_corrected += 1
            band_factors[(str(prop), str(band))] = result["factor"]
        # contratos rincon: solo celdas con gates pasados entran al dict
        # (los callers tratan "sin llave" como factor 1.0)
        if result["reason"] is None:
            factors[(str(prop), str(band))] = result["factor"]
        cells.append(
            {"property": str(prop), "lead_band": str(band), **_cell_public(result)}
        )

    if not seasonal_cells:
        return factors, {
            "n_cells": len(cells),
            "n_corrected": n_corrected,
            "min_obs": min_obs,
            "shrinkage": shrinkage,
            "max_rel": max_rel,
            "min_consistency": min_consistency,
            "cells": cells,
        }

    # ----- celdas estacionales: el régimen del que el gate por banda
    # desconfía, aislado. El sesgo FIESTA −13.5% del horizonte ago-oct
    # era inconsistente ENTRE estaciones y consistente DENTRO de una.
    work["season"] = season_of_stay(work["stay_date"])
    season_cells: list[dict] = []
    composed: dict[tuple[str, str, str], float] = {}
    n_season_cells = 0
    for (prop, band, season), grp in work.groupby(
        ["property", "lead_band", "season"], observed=True
    ):
        est = _bias_cell_estimate(
            grp, min_obs=min_obs, shrinkage=shrinkage, max_rel=max_rel,
            min_correction=min_correction, min_consistency=min_consistency,
        )
        band_factor = float(band_factors.get((str(prop), str(band)), 1.0))
        if est["applied"]:
            final, source = float(est["factor"]), "season"
            n_season_cells += 1
        elif band_factor != 1.0:
            final, source = band_factor, "band"
        else:
            final, source = 1.0, "none"
        composed[(str(prop), str(band), str(season))] = final
        season_cells.append(
            {"property": str(prop), "lead_band": str(band), "season": str(season),
             "band_factor": band_factor,
             **_cell_public(est), "factor": final, "source": source}
        )
    return composed, {
        "n_cells": len(cells),
        "n_corrected": n_corrected,
        "n_season_cells": n_season_cells,
        "min_obs": min_obs,
        "shrinkage": shrinkage,
        "max_rel": max_rel,
        "min_consistency": min_consistency,
        "cells": cells,
        "season_cells": season_cells,
        "band_factors": band_factors,
    }


def _bias_cell_estimate(
    grp: pd.DataFrame,
    *,
    min_obs: int,
    shrinkage: int,
    max_rel: float,
    min_correction: float,
    min_consistency: float,
) -> dict:
    """Estimador de UNA celda de bias (banda o estacional) con sus gates.

    Cuerpo compartido del estimador de ``lead_bias_factors``: celdas
    (propiedad, banda) y (propiedad, banda, temporada) aplican la MISMA
    disciplina de evidencia — ``rel_bias`` por grupo, consistencia del
    signo entre folds, shrinkage por tamaño de muestra, clip y umbral
    mínimo de corrección. Devuelve SIEMPRE el detalle (aplicado o la
    razón del bloqueo) — nada silencioso.
    """
    n = int(len(grp))
    denom = float(grp["actual"].abs().sum())
    if n < min_obs or denom <= 0:
        return {"n_obs": n, "rel_bias": None, "consistency": None,
                "factor": 1.0, "applied": False,
                "reason": f"n<{min_obs} o sin reales", "en_gates": False}
    rel = float((grp["forecast"] - grp["actual"]).sum() / denom)
    # consistencia del signo del bias entre folds (estabilidad del
    # sesgo): un sesgo que gira entre ventanas es régimen, no bias
    fold_biases = (
        grp.groupby("fold")
        .apply(lambda g: float((g["forecast"] - g["actual"]).sum()
                               / g["actual"].abs().sum()), include_groups=False)
        if "fold" in grp.columns and grp["fold"].nunique() > 1
        else None
    )
    consistency = None
    if fold_biases is not None and len(fold_biases.dropna()) > 0:
        agg_sign = 1 if rel > 0 else -1
        signs = fold_biases.dropna().map(lambda v: 1 if v > 0 else -1)
        consistency = float((signs == agg_sign).mean())
    if consistency is not None and consistency < min_consistency:
        return {"n_obs": n, "rel_bias": rel, "consistency": consistency,
                "factor": 1.0, "applied": False,
                "reason": (
                    f"sesgo inestable entre folds (consistencia "
                    f"{consistency:.0%} < {min_consistency:.0%})"
                ), "en_gates": False}
    raw = float(np.clip(1.0 - rel, 1.0 - max_rel, 1.0 + max_rel))
    weight = n / (n + shrinkage)
    factor = 1.0 + (raw - 1.0) * weight
    applied = abs(factor - 1.0) >= min_correction
    if not applied:
        factor = 1.0
    return {"n_obs": n, "rel_bias": rel, "consistency": consistency,
            "factor": float(factor), "applied": bool(applied),
            "reason": None, "en_gates": True}


def _cell_public(est: dict) -> dict:
    """Detalle público de una celda de bias para el manifiesto."""
    return {"n_obs": est["n_obs"], "rel_bias": est["rel_bias"],
            "consistency": est["consistency"], "factor": est["factor"],
            "applied": est["applied"], "reason": est["reason"]}


def apply_bias_factors(
    frame: pd.DataFrame,
    factors: dict | None,
    *,
    property_col: str = "property",
    lead_col: str = "lead_days",
    target_col: str = "raw_forecast_rn",
    fallback_factors: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Aplica la corrección por (propiedad, banda) al forecast publicado.

    Multiplicativa sobre el punto ANTES del piso de reservas y el techo
    de capacidad (las restricciones se re-aplican después): una corrección
    a la baja jamás publica menos que lo ya reservado, y una al alza
    jamás supera el inventario.

    Con llaves 3-tuplas ``(prop, banda, temporada)`` (modo estacional) la
    temporada de cada fila sale del calendario del stay (sin lookahead) y
    una combinación estacional sin clave cae al ``fallback_factors`` de
    banda si existe — nunca se inventa un factor fuera del reporte.
    """
    if not factors or frame is None or frame.empty:
        return frame
    if target_col not in frame.columns or lead_col not in frame.columns:
        return frame
    from src.evaluation.breakdowns import lead_band, season_of_stay

    seasonal = all(isinstance(k, tuple) and len(k) == 3 for k in factors)
    out = frame.copy()
    bands = out[lead_col].map(lead_band)
    props = out[property_col].astype(str)
    if seasonal:
        date_key = "stay_date" if "stay_date" in out.columns else None
        if date_key is None:
            raise ValueError(
                "factors estacional requiere la columna de fecha de estancia"
            )
        seasons = season_of_stay(out[date_key]).fillna("__sin_temporada__")
    factor = pd.Series(1.0, index=out.index)
    llave_presente = pd.Series(False, index=out.index)
    for key, f in factors.items():
        if seasonal:
            prop, band, season = key
            mask = (props == str(prop)) & (bands == str(band)) & (seasons == str(season))
        else:
            prop, band = key
            mask = (props == str(prop)) & (bands == str(band))
        if mask.any():
            if float(f) != 1.0:
                factor.loc[mask] = float(f)
            llave_presente[mask] = True
    if seasonal and fallback_factors:
        # SOLO combinaciones estacionales sin filas en los folds (llave
        # ausente del dict) heredan el factor de banda; las filas con
        # llave estacional presente ya fueron decididas por el estimador
        # con su fallback interno — re-aplicar la banda sería doble conteo
        con_clave = llave_presente.reindex(out.index, fill_value=False)
        applied_fallback = 0
        for (prop, band), f in fallback_factors.items():
            if float(f) == 1.0:
                continue
            missing = (
                (props == str(prop)) & (bands == str(band)) & ~con_clave
            )
            if missing.any():
                factor.loc[missing] = float(f)
                applied_fallback += int(missing.sum())
        if applied_fallback:
            out.attrs["bias_fallback_records"] = applied_fallback
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce") * factor
    out["bias_correction_factor"] = factor
    return out


def level_correction_factors(
    detail: pd.DataFrame | None,
    *,
    trigger_ts: float = 4.0,
    window_days: int = 60,
    min_obs: int = 10,
    shrinkage: int = 10,
    max_rel: float = 0.20,
    min_correction: float = 0.005,
) -> tuple[dict[str, float], dict]:
    """Corrección de nivel por propiedad — drift del forecast PUBLICADO.

    Los folds del backtest mezclan regímenes del año y sus gates (con
    razón) bloquean sesgos que giran entre ventanas; un drift de nivel
    RECIENTE (el boom YoY de FIESTA, el cambio de régimen de VILLAS) no
    es un bias del año: es el régimen de las últimas semanas. La señal
    honesta es el error del forecast que el negocio ya vio: el detalle
    forecast-vs-real de la publicación anterior (``vintage_detail``),
    walk-forward puro — cada corrección usa solo información ≤ su corte.

    Sobre la ventana reciente de estancias (``window_days``) por propiedad:

    - ``rel = Σ(fc − act) / Σ|act|`` — magnitud y signo del desvío;
    - ``ts = Σerr / media|err|`` — la tracking signal del monitoreo,
      acotada por tamaño de muestra por construcción;
    - **peso suave** ``w = min(|ts|/trigger, 1)``: sin estado, sin
      parpadeo alrededor del umbral — a TS=4 corrige completo, a TS=2 a
      medias; el control fino lo da el shrinkage, no un interruptor;
    - **disciplina**: shrinkage ``n/(n+shrinkage)``, clip ``±max_rel``,
      ``min_obs`` para estimar y correcciones < ``min_correction`` no
      tocan el punto.

    Returns
    -------
    (factors, report): factor por propiedad (solo las aplicadas; callers
    tratan "sin llave" como 1.0) y el detalle completo por propiedad
    (n, rel, ts, peso, factor, razón) para el manifiesto — nada
    silencioso.
    """
    factors: dict[str, float] = {}
    report: dict = {
        "trigger_ts": trigger_ts,
        "window_days": window_days,
        "min_obs": min_obs,
        "shrinkage": shrinkage,
        "max_rel": max_rel,
        "properties": [],
    }
    if detail is None or detail.empty:
        report["note"] = "sin errores publicados evaluables"
        return factors, report

    work = detail.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work = work.dropna(subset=["stay_date"])
    if work.empty:
        report["note"] = "sin fechas de estancia"
        return factors, report
    fin = pd.Timestamp(work["stay_date"].max())
    inicio = fin - pd.Timedelta(days=int(window_days))
    win = work.loc[
        (work["stay_date"] > inicio) & (work["stay_date"] <= fin)
    ].copy()
    report["window"] = {"start": str(inicio.date()), "end": str(fin.date())}
    if win.empty:
        report["note"] = f"sin estancias en la ventana de {window_days} dias"
        return factors, report

    for prop, grp in win.groupby("property", observed=True):
        n = int(len(grp))
        fc = pd.to_numeric(grp["published_forecast_rn"], errors="coerce")
        act = pd.to_numeric(grp["actual_rn"], errors="coerce")
        ok = fc.notna() & act.notna()
        n = int(ok.sum())
        cell: dict = {"property": str(prop), "n_obs": n,
                      "window_days": int(window_days)}
        if n < min_obs:
            cell.update({"rel_error": None, "ts": None, "weight": 0.0,
                         "factor": 1.0, "applied": False,
                         "reason": f"n<{min_obs} en la ventana"})
            report["properties"].append(cell)
            continue
        f_, a_ = fc[ok].to_numpy(dtype=float), act[ok].to_numpy(dtype=float)
        err = f_ - a_
        denom = float(np.abs(a_).sum())
        rel = float(err.sum() / denom) if denom > 0 else 0.0
        mad = float(np.mean(np.abs(err)))
        ts = float(err.sum() / mad) if mad > 0 else 0.0
        weight = float(min(abs(ts) / max(trigger_ts, 1e-9), 1.0))
        shrink = n / (n + shrinkage)
        raw = 1.0 - rel * weight * shrink
        factor = float(np.clip(raw, 1.0 - max_rel, 1.0 + max_rel))
        applied = abs(factor - 1.0) >= min_correction
        cell.update({
            "rel_error": rel, "ts": ts, "weight": weight,
            "shrinkage_w": shrink,
            "factor": float(factor) if applied else 1.0,
            "applied": bool(applied),
            "reason": None if applied else "correccion inmaterial (<0.5%)",
        })
        if applied:
            factors[str(prop)] = float(factor)
        report["properties"].append(cell)
    report["n_applied"] = len(factors)
    return factors, report


def apply_level_factors(
    frame: pd.DataFrame,
    factors: dict[str, float] | None,
    *,
    property_col: str = "property",
    target_col: str = "raw_forecast_rn",
) -> pd.DataFrame:
    """Aplica la corrección de nivel por propiedad al forecast.

    Multiplicativa sobre el punto ANTES de la corrección por banda y de
    las restricciones (piso de reservas / techo de capacidad se
    re-aplican después). Escribe ``level_correction_factor`` — sin
    factores, el frame vuelve intacto (knob OFF bit a bit).
    """
    if not factors or frame is None or frame.empty:
        return frame
    if target_col not in frame.columns:
        return frame
    out = frame.copy()
    props = out[property_col].astype(str)
    factor = pd.Series(1.0, index=out.index)
    for prop, f in factors.items():
        if float(f) == 1.0:
            continue
        mask = props == str(prop)
        if mask.any():
            factor.loc[mask] = float(f)
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce") * factor
    out["level_correction_factor"] = factor
    return out


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
