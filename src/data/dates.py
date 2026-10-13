"""Parseo flexible de fechas para fuentes heterogéneas (spec data-ingestion).

Las exportaciones reales mezclan formatos: ``YYYY-MM-DD`` (ISO) en
demanda/pickup y ``DD/MM/YYYY`` (estándar del negocio) en capacity. El
parser por defecto de pandas asume ``MM/DD`` ante fechas con barras, por
lo que en una fuente DD/MM:

* descarta silenciosamente ~60% de las fechas (día > 12 no puede ser
  mes), y
* cruza día y mes en las ambiguas ("05/01/2025" = 5-ene se lee como
  5-may).

Peor aún: ``dayfirst=True`` a ciegas CRUZA las ISO ("2026-10-05" se lee
como 2026-05-10), así que no es un superset seguro. La detección aquí
es por patrón de la columna, con evidencia explícita:

* ISO puro → formato estricto ``%Y-%m-%d`` (nunca se cruza).
* Barras → la evidencia decide: algún día > 12 en la primera posición
  prueba DD/MM; en la segunda posición prueba MM/DD. Columna 100%
  ambigua → ``prefer_dayfirst`` desempata (estándar del negocio).
* Otro/mixto → comparación por conteo como mejor esfuerzo.

Sin configuración: el motor se adapta a la fuente.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

_ISO_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


def _attempt(series: pd.Series, **kwargs: object) -> pd.Series:
    """Intento de parseo con el UserWarning de pandas silenciado.

    Pandas avisa ("Parsing dates in %d/%m/%Y format when dayfirst=False")
    exactamente en el caso que aquí sondeamos a propósito.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce", **kwargs)


def _by_count(
    default: pd.Series, dayfirst: pd.Series, prefer_dayfirst: bool
) -> pd.Series:
    """Gana el intento que resuelva más fechas; empate → preferencia."""
    n_default = int(default.notna().sum())
    n_dayfirst = int(dayfirst.notna().sum())
    if n_dayfirst > n_default:
        return dayfirst
    if n_dayfirst == n_default and prefer_dayfirst and n_default > 0:
        return dayfirst
    return default


def parse_dates_flexible(
    series: pd.Series, *, prefer_dayfirst: bool = False
) -> pd.Series:
    """Parsea una columna de fechas con detección de formato por patrón.

    1. Series datetime (o vacía) se devuelve tal cual — el resto del
       motor ya trabaja con datetime.
    2. Muestra no nula de strings → clasificación:

       * ISO puro (≥90%): ``format='%Y-%m-%d'`` estricto — la lectura
         ISO jamás se cruza, venga ``dayfirst`` o no.
       * Barras (≥90%): evidencia día>12 en primera posición → DD/MM;
         en segunda → MM/DD; columna 100% ambigua (todo día ≤ 12) →
         ``prefer_dayfirst`` desempata. Un events.csv cargado a mano
         en el estándar del negocio puede ser todo ambiguo: ahí el
         conteo no prueba nada y la preferencia del negocio manda.
       * Otro/mixto: comparación por conteo entre default y
         ``dayfirst=True`` (mejor esfuerzo).

    Las fechas que ningún formato resuelve quedan NaT: la validación de
    calidad las reporta (ilegibles), no las oculta.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if series.empty:
        return pd.to_datetime(series, errors="coerce")

    sample = (
        series.dropna().astype(str).str.strip()
    )
    sample = sample.loc[sample != ""]
    if sample.empty:
        return pd.to_datetime(series, errors="coerce")

    iso_frac = float(sample.str.match(_ISO_RE).mean())
    if iso_frac >= 0.9:
        return _attempt(series, format="%Y-%m-%d")

    slash_mask = sample.str.match(_SLASH_RE)
    if float(slash_mask.mean()) >= 0.9:
        parts = sample.loc[slash_mask].str.extract(_SLASH_RE)
        first = pd.to_numeric(parts[0], errors="coerce")
        second = pd.to_numeric(parts[1], errors="coerce")
        if bool((first > 12).any()):
            dayfirst = True
        elif bool((second > 12).any()):
            dayfirst = False
        else:
            dayfirst = prefer_dayfirst
        return _attempt(series, dayfirst=dayfirst)

    # mixto u otro: comparación por conteo (comportamiento previo)
    default = _attempt(series)
    if not bool(default.isna().any()) and not prefer_dayfirst:
        return default
    dayfirst_attempt = _attempt(series, dayfirst=True)
    return _by_count(default, dayfirst_attempt, prefer_dayfirst)
