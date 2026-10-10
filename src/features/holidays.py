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
    """Calendario compartido de la corrida (benchmark/forecast/backtest).

    ``dates``: fechas de feriados Y eventos (alimentan la feature de
    distancia ``days_to_nearest_holiday`` — "feriado o evento más cercano").
    ``holiday_dates``: solo feriados del calendario público (el indicador
    ``is_holiday`` queda reservado a ellos, nunca marca eventos).
    ``windows``: fecha → clave (día dentro de ventana de feriado o evento de
    portafolio). ``scoped``: ventanas de eventos con propiedad asignada —
    solo aplican a esa propiedad.
    """

    dates: pd.DatetimeIndex  # feriados + eventos, ordenadas
    windows: pd.Series       # fecha → clave (feriados y eventos de portafolio)
    countries: tuple[str, ...]
    window_days: int
    holiday_dates: pd.DatetimeIndex | None = None  # solo feriados públicos
    scoped: dict | None = None  # property → Series(fecha → clave de evento)
    # distancia (días) al feriado/evento dueño de cada fecha de ventana —
    # permite desempates verdaderos al combinar calendarios
    window_dists: pd.Series | None = None

    def is_holiday(self, date) -> bool:
        """Solo feriados del calendario público (nunca eventos privados)."""
        holiday_dates = self.holiday_dates if self.holiday_dates is not None else self.dates
        if len(holiday_dates) == 0:
            return False
        return pd.Timestamp(date).normalize() in set(holiday_dates)

    def days_to_nearest(self, date) -> float:
        """Días (absolutos) al feriado/evento más cercano; NaN sin calendario."""
        if len(self.dates) == 0:
            return float("nan")
        d = pd.Timestamp(date).normalize()
        return float(abs((self.dates - d).days).min())

    def window_key(self, date, property: str | None = None):
        """Clave del feriado/evento cuya ventana contiene la fecha.

        Consulta primero el mapa global (feriados + eventos de portafolio) y
        luego el mapa scoped de ``property`` (eventos con propiedad asignada).
        Empate entre ambos: gana el global (el feriado es más amplio).
        """
        if len(self.windows) > 0:
            key = self.windows.get(pd.Timestamp(date).normalize())
            if key is not None and not pd.isna(key):
                return str(key)
        if property is not None and self.scoped:
            scoped = self.scoped.get(str(property))
            if scoped is not None and len(scoped) > 0:
                key = scoped.get(pd.Timestamp(date).normalize())
                if key is not None and not pd.isna(key):
                    return str(key)
        return None

    def describe(self) -> dict:
        keys = set(self.windows.astype(str))
        for scoped in (self.scoped or {}).values():
            keys.update(scoped.astype(str))
        holiday_dates = self.holiday_dates
        if holiday_dates is None or len(holiday_dates) == 0:
            holiday_dates = self.dates
        return {
            "countries": list(self.countries),
            "window_days": int(self.window_days),
            "n_holiday_dates": int(len(holiday_dates)),
            "n_dates": int(len(self.dates)),  # feriados + eventos
            "n_window_days": int(len(self.windows)),
            "n_scoped_properties": int(len(self.scoped or {})),
            "keys": sorted(keys),
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
    window_dists = pd.Series(
        {d: dist for d, (dist, _) in sorted(window_rows.items())}, dtype="int64"
    )
    window_dists.index.name = "date"
    cal = HolidayCalendar(
        dates=dates,
        windows=windows,
        countries=countries,
        window_days=int(window_days),
        holiday_dates=dates,  # sin eventos: todo es calendario público
        window_dists=window_dists,
    )
    _CALENDAR_CACHE[key] = cal
    return cal


def _expand_windows(
    date: pd.Timestamp, key_sem: str, window_days: int,
    window_rows: dict[pd.Timestamp, tuple[int, str]],
) -> None:
    """Añade la ventana ±window_days de una fecha, empate → la más cercana."""
    for offset in range(-int(window_days), int(window_days) + 1):
        wd = date + pd.Timedelta(days=offset)
        dist = abs(offset)
        current = window_rows.get(wd)
        if current is None or dist < current[0]:
            window_rows[wd] = (dist, key_sem)


def calendar_with_events(
    base: HolidayCalendar, events: pd.DataFrame, window_days: int | None = None
) -> HolidayCalendar:
    """Combina el calendario de feriados con los eventos privados.

    Los eventos sin propiedad (o con ``property`` nula) generan ventanas
    globales con clave = slug de su nombre; los eventos con propiedad
    generan ventanas SOLO para esa propiedad (``scoped``). Las fechas de
    todos los eventos entran a ``dates`` (feature de distancia ML);
    ``holiday_dates`` conserva solo los feriados públicos.
    """
    if events is None or events.empty:
        return base
    wdays = int(window_days if window_days is not None else base.window_days)

    dates = list(base.dates)
    # preserva el mapa base con sus distancias REALES (empates por cercanía
    # verdaderos: un evento más cercano puede reclamar una fecha de ventana
    # de un feriado, y no al revés)
    global_rows: dict[pd.Timestamp, tuple[int, str]] = {}
    base_dists = base.window_dists
    for d, k in base.windows.items():
        dist = 0
        if base_dists is not None:
            got = base_dists.get(pd.Timestamp(d))
            if got is not None and not pd.isna(got):
                dist = int(got)
        global_rows[pd.Timestamp(d)] = (dist, str(k))
    scoped_rows: dict[str, dict[pd.Timestamp, tuple[int, str]]] = {}
    for prop, series in (base.scoped or {}).items():
        scoped_rows[str(prop)] = {
            pd.Timestamp(d): (0, str(k)) for d, k in series.items()
        }

    for row in events.itertuples(index=False):
        d = pd.Timestamp(row.event_date).normalize()
        key_sem = _slugify(str(row.event_name))
        dates.append(d)
        prop = getattr(row, "property", None)
        if prop is None or (isinstance(prop, float) and pd.isna(prop)) or str(prop) in ("", "nan", "None"):
            _expand_windows(d, key_sem, wdays, global_rows)
        else:
            prop = str(prop)
            scoped_rows.setdefault(prop, {})
            _expand_windows(d, key_sem, wdays, scoped_rows[prop])

    windows = pd.Series(
        {d: k for d, (_, k) in sorted(global_rows.items())}, dtype=object
    )
    windows.index.name = "date"
    window_dists = pd.Series(
        {d: dist for d, (dist, _) in sorted(global_rows.items())}, dtype="int64"
    )
    window_dists.index.name = "date"
    scoped = {
        prop: pd.Series({d: k for d, (_, k) in sorted(rows.items())}, dtype=object)
        for prop, rows in scoped_rows.items()
        if rows
    }
    holiday_dates = base.holiday_dates if base.holiday_dates is not None else base.dates
    return HolidayCalendar(
        dates=pd.DatetimeIndex(sorted(set(dates))),
        windows=windows,
        countries=base.countries,
        window_days=wdays,
        holiday_dates=holiday_dates,
        scoped=scoped or None,
        window_dists=window_dists,
    )


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
