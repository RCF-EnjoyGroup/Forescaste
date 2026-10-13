"""Fuente opcional de eventos privados (conciertos, congresos, torneos).

Contrato (README_SISTEMAS.md): CSV en ``data.events_path`` (default
``data/events.csv``) con columnas ``event_name`` (o ``evento``),
``event_date`` (o ``fecha``) y ``property`` opcional (blank/NaN = evento de
portafolio). El archivo es OPCIONAL: sin él, el motor queda idéntico a una
corrida sin eventos y el manifiesto declara la fuente ausente.

Un evento es una ventana más del calendario compartido: su fecha entra a
``HolidayCalendar.dates`` (la feature ML de distancia al "feriado/evento más
cercano" la cubre) y su ventana ±``window_days`` al mapa de ventanas con
clave = slug del nombre. Los eventos con propiedad asignada generan ventanas
SOLO para esa propiedad (scoping); ``is_holiday`` queda reservado a feriados
del calendario público.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dates import parse_dates_flexible

_NAME_CANDIDATES = ("event_name", "evento", "event", "nombre", "name")
_DATE_CANDIDATES = ("event_date", "fecha", "date", "dia")
_PROPERTY_CANDIDATES = ("property", "propiedad", "hotel", "hotel_id")


class EventsError(ValueError):
    """El archivo de eventos no cumple el contrato."""


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...], what: str) -> str:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    raise EventsError(f"events.csv sin columna de {what}: use alguna de {list(candidates)}")


def load_events(path: str | Path) -> pd.DataFrame | None:
    """Carga y valida el archivo de eventos; ``None`` si no existe.

    Validación dura (antes de entrenar): fechas legibles y sin duplicados por
    (nombre, fecha, propiedad). Un evento con fecha corrupta degradaría
    ventanas silenciosamente — mejor fallar con mensaje claro.
    """
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    name_col = _find_col(df, _NAME_CANDIDATES, "nombre de evento")
    date_col = _find_col(df, _DATE_CANDIDATES, "fecha de evento")
    prop_col = None
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in _PROPERTY_CANDIDATES:
        if cand in lowered:
            prop_col = lowered[cand]
            break

    out = pd.DataFrame(
        {
            "event_name": df[name_col].astype(str).str.strip(),
            # ISO o DD/MM (estándar del negocio): mismo parser flexible que
            # las fuentes principales — "05/10/2026" es 5-oct, no 10-may.
            # prefer_dayfirst: un events.csv corto puede ser 100% ambiguo
            # (día ≤ 12) y el conteo solo no puede desambiguar.
            "event_date": parse_dates_flexible(df[date_col], prefer_dayfirst=True),
        }
    )
    if prop_col is not None:
        out["property"] = df[prop_col].astype(str).str.strip()
        out.loc[out["property"].isin(("", "nan", "None", "NULL")), "property"] = None
    else:
        out["property"] = None

    bad_dates = out["event_date"].isna().sum()
    if bad_dates:
        raise EventsError(
            f"events.csv: {bad_dates} filas con fecha ilegible "
            f"(formatos aceptados: YYYY-MM-DD o DD/MM/YYYY)"
        )
    dup_mask = out.duplicated(subset=["event_name", "event_date", "property"])
    if dup_mask.any():
        sample = out.loc[dup_mask, ["event_name", "event_date"]].head(3)
        raise EventsError(
            f"events.csv: {int(dup_mask.sum())} eventos duplicados por nombre+fecha+propiedad "
            f"(ejemplos: {sample.to_dict('records')})"
        )
    return out
