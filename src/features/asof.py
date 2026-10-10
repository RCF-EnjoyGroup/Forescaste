"""Primitiva única anti-leakage: as-of snapshot builder (spec ``leakage-prevention``, D3).

Todas las features del motor se construyen a través de :class:`AsOfBuilder`.
Recibe la fecha de corte y **solo** emite columnas computadas con datos
observados ≤ corte. Cualquier intento de leer datos posteriores al corte
lanza :class:`LeakageError`, de modo que las pruebas de leakage (3.4) atacan
este contrato en un único punto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

CUTOFF_ATTR = "_asof_cutoff"


class LeakageError(RuntimeError):
    """Se intentó usar información posterior a la fecha de corte."""


def enforce_cutoff(df: pd.DataFrame, date_col: str, cutoff: pd.Timestamp, label: str = "") -> None:
    """Falla si ``df`` contiene observaciones posteriores a ``cutoff``."""
    if df.empty:
        return
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().all():
        raise LeakageError(f"{label}: columna de fecha ilegible; no se puede garantizar el corte")
    latest = dates.max()
    if latest > cutoff:
        future = sorted(str(pd.Timestamp(d).date()) for d in dates[dates > cutoff].unique())[:5]
        raise LeakageError(
            f"{label or 'dataset'} viola el corte as-of {pd.Timestamp(cutoff).date()}: "
            f"contiene {int((dates > cutoff).sum())} observaciones futuras "
            f"(fechas ejemplo: {future})"
        )


@dataclass(frozen=True)
class AsOfBuilder:
    """Construye el frame de features para una fecha de corte dada.

    Parameters
    ----------
    date_col:
        Columna de fecha de observación (snapshot) del frame fuente.
    cutoff:
        Fecha de corte del forecast (inclusive: datos ≤ cutoff son legales).
    """

    date_col: str
    cutoff: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff", pd.Timestamp(self.cutoff))

    # -- contrato central --------------------------------------------------
    def frame(self, df: pd.DataFrame, label: str = "features") -> pd.DataFrame:
        """Devuelve solo las filas observadas ≤ cutoff (y valida el resultado)."""
        dates = pd.to_datetime(df[self.date_col], errors="coerce")
        out = df.loc[dates <= self.cutoff].copy()
        enforce_cutoff(out, self.date_col, self.cutoff, label)
        return out

    def apply(
        self,
        df: pd.DataFrame,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        label: str = "features",
    ) -> pd.DataFrame:
        """Aplica ``func`` solo al frame recortado al corte y valida la salida.

        Cualquier feature calculada dentro de ``func`` que resulte con fechas
        > cutoff provoca :class:`LeakageError`.
        """
        base = self.frame(df, label)
        out = func(base)
        if self.date_col in out.columns:
            enforce_cutoff(out, self.date_col, self.cutoff, label)
        elif isinstance(out.index, pd.DatetimeIndex):
            if out.index.max() > self.cutoff:
                raise LeakageError(f"{label}: el índice resultante excede el corte {self.cutoff.date()}")
        return out

    def filter_future_targets(
        self, df: pd.DataFrame, stay_date_col: str, label: str = "targets"
    ) -> pd.DataFrame:
        """Targets (fechas de estancia) aún no observables → fuera.

        Para un corte T, una fecha de estancia > T solo puede usarse como
        *predicción*, nunca como feature ni como target de entrenamiento.
        """
        dates = pd.to_datetime(df[stay_date_col], errors="coerce")
        return df.loc[dates <= self.cutoff].copy()


def lag(series: pd.Series, periods: int) -> pd.Series:
    """Lag retrospectivo (positivo = hacia atrás). Nunca mira al futuro."""
    if periods < 0:
        raise ValueError(f"lag negativo ({periods}) miraría al futuro; use shift(-n) con cuidado")
    return series.shift(periods)


def rolling_backward(
    series: pd.Series,
    window: int,
    stat: str = "mean",
    min_periods: int = 1,
) -> pd.Series:
    """Ventana rolling solo hacia atrás, EXCLUYENDO la propia fila.

    La ventana cubre ``[i-window, i-1]``: la fila actual (típicamente el
    target) nunca participa de su propia feature. Sin el ``shift(1)``, una
    ``rolling_mean`` incluiría el target de la fila en el entrenamiento ML
    (leakage de target en la matriz de features).
    """
    if window <= 0:
        raise ValueError("window debe ser positiva")
    roll = series.shift(1).rolling(window=window, min_periods=min_periods, center=False)
    if stat == "mean":
        return roll.mean()
    if stat == "std":
        return roll.std()
    if stat == "median":
        return roll.median()
    if stat == "min":
        return roll.min()
    if stat == "max":
        return roll.max()
    if stat == "sum":
        return roll.sum()
    raise ValueError(f"stat no soportado: {stat}")


def expanding_backward(series: pd.Series, min_periods: int = 1) -> pd.Series:
    """Estadística expansiva (solo datos pasados y presentes)."""
    return series.expanding(min_periods=min_periods).mean()
