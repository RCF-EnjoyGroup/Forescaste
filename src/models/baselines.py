"""Baselines obligatorios (spec ``forecasting``, tasks 5.1/5.2).

- **Seasonal Naive**: mismo día de la semana histórico (baseline 1).
- **Pickup Curve**: ``Current Bookings + f(lead)`` no lineal aprendido de
  curvas históricas (baseline 2). La trayectoria T-90 → T-1 es decreciente en
  pickup conforme baja el lead (monotonicidad por construcción del binnig).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.calendar import season_of_month


# --------------------------------------------------------------------------
# Baseline 1 — Seasonal Naive (mismo día de la semana)
# --------------------------------------------------------------------------


@dataclass
class SeasonalNaive:
    """Forecast = valor del mismo día de la semana de la última semana disponible."""

    date_col: str = "stay_date"
    value_col: str = "room_nights"
    season_days: int = 7
    # media de las últimas K ocurrencias del mismo día de semana: una sola
    # observación es un contrincante débil (todo lo que "gana" contra él está
    # menos probado de lo que parece); K=4 ≈ 4 semanas suaviza ruido sin
    # aplanar la estacionalidad semanal
    dow_window: int = 4

    def fit(self, history: pd.DataFrame) -> "SeasonalNaive":
        self._history = history.copy()
        self._history[self.date_col] = pd.to_datetime(self._history[self.date_col])
        self._history = self._history.sort_values(self.date_col)
        return self

    def predict(self, future_dates: pd.DatetimeIndex | pd.Series) -> pd.DataFrame:
        hist = self._history
        dates = pd.DatetimeIndex(pd.to_datetime(future_dates))
        rows = []
        last = hist[self.value_col].iloc[-1]
        for d in dates:
            # últimas K ocurrencias del mismo día de la semana previas a d
            same_dow = hist.loc[hist[self.date_col].dt.dayofweek == d.dayofweek]
            same_dow = same_dow.loc[same_dow[self.date_col] < d]
            window = same_dow[self.value_col].tail(max(int(self.dow_window), 1))
            value = float(window.mean()) if len(window) else float(last)
            rows.append(
                {
                    self.date_col: d,
                    "forecast_rn": value,
                    "model_name": "seasonal_naive",
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Baseline 2 — Pickup Curve (Current Bookings + pickup esperado no lineal)
# --------------------------------------------------------------------------


@dataclass
class PickupCurveModel:
    """``Final = Current Bookings + f(lead)`` con ``f`` no lineal.

    ``f`` se aprende de curvas históricas: para cada lead bin, la media del
    pickup observado hasta la fecha final. La no linealidad viene de los bins
    (booking curve real es cóncava).

    Largo plazo (cambio improve-long-horizon-forecast): con
    ``season_conditional`` la curva se condiciona por la TEMPORADA de la
    fecha de estancia (alta/media/baja) — a leads largos el on-the-books
    cubre poco y el pickup esperado debe reflejar el ritmo de reserva de la
    temporada del stay, no el promedio anual. ``max_history_days`` recorta
    el entrenamiento a una ventana de recencia (regímenes viejos no
    promedian). Un bucket (propiedad, temporada) con menos de
    ``min_stays_per_bucket`` estancias degrada a la curva de la propiedad y,
    si tampoco existe, a la global.
    """

    n_bins: int = 10
    # Bordes de bins de lead EXPLÍCITOS (toman precedencia sobre ``n_bins``):
    # la curva de reservas es cóncava — con bins uniformes de ~9 días el lead
    # 1 y el lead 9 comparten pickup esperado, comprimiendo el gradiente
    # exactamente donde el negocio decide (leads operativos). El último borde
    # es catch-all: leads ≥ último borde usan la curva del bin final.
    # ``None`` ⇒ ``n_bins`` uniformes en [min_lead, max_lead] (comportamiento
    # previo).
    bin_edges: list[int] | None = None
    min_lead: int = 1
    date_col: str = "stay_date"
    lead_col: str = "lead_days"
    booked_col: str = "current_booked_rn"
    pickup_col: str = "pickup_rn"
    property_col: str = "property"
    season_conditional: bool = False
    max_history_days: int | None = None
    min_stays_per_bucket: int = 0
    season_min_lead: int = 0  # la curva estacional aplica solo a leads ≥ este umbral
    # nivel de condicionamiento por ventana de feriado (semana_santa, fin_de_ano,
    # thanksgiving, …): (propiedad, feriado) → (propiedad, temporada) → propiedad → global
    holiday_conditional: bool = False
    holiday_calendar: Any = None  # features.holidays.HolidayCalendar (o None)
    # umbral de estancias para los buckets de FERIADO: las ventanas son ~9 días
    # por construcción (vs meses de una temporada), así que la misma barra que
    # las temporadas sería estructuralmente inalcanzable; None ⇒ reutiliza
    # min_stays_per_bucket
    min_stays_per_holiday_bucket: int | None = None

    def fit(self, curves: pd.DataFrame) -> "PickupCurveModel":
        """``curves``: snapshot final por stay date con lead y pickup restante."""
        df = curves.copy()
        df = df.loc[df[self.lead_col] >= self.min_lead]
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        if self.max_history_days is not None and not df.empty:
            keep_from = df[self.date_col].max() - pd.Timedelta(days=int(self.max_history_days))
            df = df.loc[df[self.date_col] >= keep_from]
        if self.bin_edges is not None:
            edges_arr = np.asarray(self.bin_edges, dtype=float)
            if len(edges_arr) < 2 or np.any(np.diff(edges_arr) <= 0):
                raise ValueError(
                    f"bin_edges debe tener >= 2 bordes estrictamente crecientes: {self.bin_edges}"
                )
            if edges_arr[0] < self.min_lead:
                raise ValueError(
                    f"el primer borde ({edges_arr[0]}) no puede ser < min_lead ({self.min_lead})"
                )
            # catch-all: leads >= último borde caen en el bin final
            edges = np.concatenate([edges_arr, [np.inf]])
        else:
            max_lead = int(df[self.lead_col].max()) if len(df) else 365
            self.max_lead_ = max_lead
            edges = np.linspace(self.min_lead, max(max_lead, self.min_lead) + 1, self.n_bins + 1)
        df["_bin"] = pd.cut(df[self.lead_col], bins=edges, include_lowest=True, right=False)
        by_prop: dict[str, pd.Series] = {}
        for prop, grp in df.groupby(self.property_col):
            by_prop[str(prop)] = grp.groupby("_bin", observed=True)[self.pickup_col].mean()
        self.curves_ = by_prop
        self.global_curve_ = df.groupby("_bin", observed=True)[self.pickup_col].mean()
        self.season_curves_: dict[tuple[str, str], pd.Series] = {}
        if self.season_conditional:
            df["_season"] = df[self.date_col].dt.month.map(
                lambda m: season_of_month(m) if pd.notna(m) else None
            )
            for (prop, season), grp in df.groupby([self.property_col, "_season"], dropna=True):
                n_stays = int(grp[self.date_col].nunique())
                if n_stays >= int(self.min_stays_per_bucket):
                    self.season_curves_[(str(prop), str(season))] = (
                        grp.groupby("_bin", observed=True)[self.pickup_col].mean()
                    )
        self.holiday_curves_: dict[tuple[str, str], pd.Series] = {}
        if self.holiday_conditional and self.holiday_calendar is not None:
            holiday_min = (
                int(self.min_stays_per_holiday_bucket)
                if self.min_stays_per_holiday_bucket is not None
                else int(self.min_stays_per_bucket)
            )
            # claves de ventana por fila, con scoping: un evento con propiedad
            # asignada solo condiciona a ESA propiedad
            keys = pd.Series(index=df.index, dtype=object)
            for prop_g, grp in df.groupby(self.property_col, sort=False):
                stay_dates_g = pd.DatetimeIndex(grp[self.date_col])
                keys.loc[grp.index] = [
                    self.holiday_calendar.window_key(d, str(prop_g))
                    for d in stay_dates_g
                ]
            df["_holiday"] = keys.to_numpy()
            for (prop, hkey), grp in df.groupby([self.property_col, "_holiday"], dropna=True):
                n_stays = int(grp[self.date_col].nunique())
                if n_stays >= holiday_min:
                    self.holiday_curves_[(str(prop), str(hkey))] = (
                        grp.groupby("_bin", observed=True)[self.pickup_col].mean()
                    )
        return self

    def _curve_for(self, prop: str, stay_date, lead: int | None = None) -> pd.Series:
        """Curva aplicable: (propiedad, feriado) → (propiedad, temporada) →
        propiedad → global.

        Ambos condicionamientos solo se consultan a partir de
        ``season_min_lead``: a leads cortos el on-the-books domina y el
        pooling de la curva de propiedad (más datos por bin) es menos
        ruidoso que cualquier bucket fino; a leads largos la ventana de
        feriado es la señal más específica (Semana Santa ≠ un marzo
        cualquiera), con la temporada como primer nivel de fallback.
        """
        if stay_date is not None:
            if lead is None or int(lead) >= int(self.season_min_lead):
                ts = pd.Timestamp(stay_date)
                if not pd.isna(ts):
                    if self.holiday_conditional and self.holiday_calendar is not None:
                        hkey = self.holiday_calendar.window_key(ts, str(prop))
                        if hkey is not None:
                            curve = self.holiday_curves_.get((str(prop), hkey))
                            if curve is not None:
                                return curve
                    if self.season_conditional:
                        key = (str(prop), season_of_month(ts.month))
                        curve = self.season_curves_.get(key)
                        if curve is not None:
                            return curve
        return self.curves_.get(str(prop), self.global_curve_)

    def _lookup(self, prop: str, lead: int, stay_date=None) -> float:
        curve = self._curve_for(prop, stay_date, lead)
        idx = pd.IntervalIndex(curve.index)
        hit = idx.contains(lead)
        if hit.any():
            return float(curve.iloc[np.argmax(hit)])
        # fuera de rango: usar el extremo más cercano
        centers = idx.mid.to_numpy(dtype=float)
        return float(curve.iloc[int(np.argmin(np.abs(centers - lead)))])

    def predict(self, requests: pd.DataFrame) -> pd.DataFrame:
        """``requests``: property, stay_date, lead_days, current_booked_rn."""
        rows = []
        for _, r in requests.iterrows():
            expected_pickup = max(
                self._lookup(str(r[self.property_col]), int(r[self.lead_col]), r.get(self.date_col)),
                0.0,
            )
            rows.append(
                {
                    self.property_col: r[self.property_col],
                    self.date_col: r[self.date_col],
                    self.lead_col: int(r[self.lead_col]),
                    "current_booked_rn": float(r[self.booked_col]),
                    "expected_pickup_rn": expected_pickup,
                    "forecast_rn": float(r[self.booked_col]) + expected_pickup,
                    "model_name": "pickup_curve",
                }
            )
        return pd.DataFrame(rows)
