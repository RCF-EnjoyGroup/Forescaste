"""Calendario de feriados CR + mercados emisores (spec ``feature-engineering``).

Fuente confiable: librería ``holidays`` — resuelve fechas movibles
(computus de Semana Santa; N-ésimo día de semana de Thanksgiving/Memorial/
Labor Day) para cualquier año. Los feriados son exógenos y conocidos con
anticipación: usables a cualquier lead sin leakage.

Dos vistas:

- ``dates``: fechas exactas de feriado (indicador ``is_holiday`` y
  ``days_to_nearest_holiday`` para los candidatos ML).
- ``windows``: Series fecha → clave semántica del feriado para cada día
  dentro de ± ``window_days`` (la "ventana de viaje"); empate → el feriado
  más cercano. Es la base del condicionamiento por feriado de la curva de
  pickup (p. ej. Semana Santa ≠ un marzo cualquiera).

Agrupación semántica: los días de Semana Santa se funden en ``semana_santa``,
Navidad + Año Nuevo en ``fin_de_ano``; el resto de feriados conserva su slug
o recibe una clave estándar (thanksgiving, memorial_day, …).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

import holidays as _holidays

# (subcadena insensible a mayúsculas/acentos, clave semántica)
_GROUP_RULES: list[tuple[tuple[str, ...], str]] = [
    (("santo", "easter"), "semana_santa"),
    (("navidad", "christmas", "ano nuevo", "new year"), "fin_de_ano"),
    (("thanksgiving",), "thanksgiving"),
    (("memorial",), "memorial_day"),
    (("labor day",), "labor_day_us"),
    (("independence day", "julio"), "independencia_us"),
    (("independencia",), "independencia_cr"),
    (("anexion",), "anexion"),
    (("madre", "mother"), "dia_de_la_madre"),
]

_NORMALIZE = re.compile(r"[^a-z0-9]+")


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def _slugify(name: str) -> str:
    return _NORMALIZE.sub("_", _strip_accents(name.strip().lower())).strip("_")


def holiday_key(name: str) -> str:
    """Clave semántica estable para un feriado de la librería."""
    lowered = _strip_accents(name.lower())
    for tokens, key in _GROUP_RULES:
        if any(tok in lowered for tok in tokens):
            return key
    return _slugify(name)


@dataclass(frozen=True)
class HolidayCalendar:
    """Calendario compartido de la corrida (benchmark/forecast/backtest)."""

    dates: pd.DatetimeIndex  # fechas exactas de feriado, ordenadas
    windows: pd.Series       # fecha → clave semántica (día dentro de ventana)
    countries: tuple[str, ...]
    window_days: int

    def is_holiday(self, date) -> bool:
        if len(self.dates) == 0:
            return False
        return pd.Timestamp(date).normalize() in set(self.dates)

    def days_to_nearest(self, date) -> float:
        """Días (absolutos) al feriado más cercano; NaN sin calendario."""
        if len(self.dates) == 0:
            return float("nan")
        d = pd.Timestamp(date).normalize()
        return float(abs((self.dates - d).days).min())

    def window_key(self, date):
        """Clave del feriado cuya ventana contiene la fecha (None fuera)."""
        if len(self.windows) == 0:
            return None
        key = self.windows.get(pd.Timestamp(date).normalize())
        return None if pd.isna(key) else str(key)

    def describe(self) -> dict:
        return {
            "countries": list(self.countries),
            "window_days": int(self.window_days),
            "n_holiday_dates": int(len(self.dates)),
            "n_window_days": int(len(self.windows)),
            "keys": sorted(set(self.windows.astype(str))),
        }


_CALENDAR_CACHE: dict[tuple, HolidayCalendar] = {}


def build_holiday_calendar(
    years: range | list[int] | tuple[int, ...],
    countries: tuple[str, ...] | list[str] = ("CR", "US"),
    window_days: int = 4,
) -> HolidayCalendar:
    """Construye (con cache) el calendario de feriados para un rango de años.

    ``windows`` asigna a cada fecha dentro de ±``window_days`` de un feriado
    la clave del feriado MÁS CERCANO (empates: primero en el orden de fechas).
    """
    years = tuple(sorted(int(y) for y in years))
    countries = tuple(countries)
    key = (years, countries, int(window_days))
    cached = _CALENDAR_CACHE.get(key)
    if cached is not None:
        return cached
    if not years or not countries:
        cal = HolidayCalendar(
            dates=pd.DatetimeIndex([]),
            windows=pd.Series(dtype=object),
            countries=countries,
            window_days=int(window_days),
        )
        _CALENDAR_CACHE[key] = cal
        return cal

    exact: dict[pd.Timestamp, str] = {}
    window_rows: dict[pd.Timestamp, tuple[int, str]] = {}
    for country in countries:
        locale = getattr(_holidays, str(country).upper(), None)
        if locale is None:
            raise ValueError(f"país de feriados no soportado por la librería: '{country}'")
        for date, name in locale(years=years).items():
            d = pd.Timestamp(date).normalize()
            key_sem = holiday_key(name)
            exact[d] = key_sem
            for offset in range(-int(window_days), int(window_days) + 1):
                wd = d + pd.Timedelta(days=offset)
                dist = abs(offset)
                current = window_rows.get(wd)
                if current is None or dist < current[0]:
                    window_rows[wd] = (dist, key_sem)

    dates = pd.DatetimeIndex(sorted(exact))
    windows = pd.Series(
        {d: k for d, (_, k) in sorted(window_rows.items())}, dtype=object
    )
    windows.index.name = "date"
    cal = HolidayCalendar(
        dates=dates,
        windows=windows,
        countries=countries,
        window_days=int(window_days),
    )
    _CALENDAR_CACHE[key] = cal
    return cal


def holiday_calendar_from_config(
    config, min_year: int, max_year: int
) -> HolidayCalendar | None:
    """Construye el calendario desde ``config.holidays`` (None si deshabilitado)."""
    cfg = dict(getattr(config, "holidays", {}) or {})
    countries = tuple(cfg.get("countries") or ())
    if not countries:
        return None
    window_days = int(cfg.get("window_days", 4))
    return build_holiday_calendar(
        range(int(min_year) - 1, int(max_year) + 2), countries, window_days
    )
