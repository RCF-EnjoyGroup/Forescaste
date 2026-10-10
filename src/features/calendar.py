"""Features de calendario y estacionalidad (spec ``feature-engineering``, 7.1).

Se derivan **solo** de la fecha de estancia: funcionan también en el horizonte
futuro sin requerir datos observados. El indicador de feriado solo se llena
si existe una fuente confiable (si no, se omite y se documenta).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# Temporadas del hemisferio norte / Costa Rica (mes -> temporada)
_SEASON_BY_MONTH = {
    12: "high", 1: "high", 2: "high",          # alta
    3: "shoulder", 4: "shoulder", 5: "shoulder",  # media
    6: "low", 7: "low", 8: "low", 9: "low",     # baja
    10: "shoulder", 11: "shoulder",
}


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "stay_date",
    holidays: Iterable[pd.Timestamp] | None = None,
    holiday_col: str = "is_holiday",
) -> pd.DataFrame:
    """Agrega features de calendario derivadas exclusivamente de ``date_col``.

    Si ``holidays`` es ``None`` (sin fuente confiable) la columna de feriado
    **no** se crea — se documenta la ausencia en ``has_holiday_feature``.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")

    out["day_of_week"] = dates.dt.dayofweek           # 0=lun .. 6=dom
    out["is_weekend"] = out["day_of_week"] >= 5
    out["week_of_year"] = dates.dt.isocalendar().week.astype("int64")
    out["month"] = dates.dt.month
    out["quarter"] = dates.dt.quarter
    out["year"] = dates.dt.year
    out["day_of_month"] = dates.dt.day
    # días hasta fin de mes
    month_end = dates.dt.to_period("M")
    end_of_month = month_end.dt.end_time.dt.normalize()
    out["days_to_month_end"] = (end_of_month - dates).dt.days
    out["season"] = out["month"].map(_SEASON_BY_MONTH).astype("category")

    if holidays is not None:
        holiday_index = pd.DatetimeIndex([pd.Timestamp(h) for h in holidays])
        out[holiday_col] = dates.isin(holiday_index)

    return out


def has_holiday_feature(df: pd.DataFrame, holiday_col: str = "is_holiday") -> bool:
    """True si existe la columna de feriado (fuente confiable disponible)."""
    return holiday_col in df.columns
