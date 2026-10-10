"""Métricas de evaluación (spec ``evaluation``, tasks 6.2/6.3).

MAPE (con manejo explícito de ``Actual = 0``: exclusión + conteo documentado),
WMAPE (métrica corporativa principal), MAE (Room Nights), RMSE y Bias
(``Σ(Forecast - Actual) / ΣActual`` → dirección del error).

Nunca se sustituye el cero por un valor arbitrario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    """Métricas + documentación del manejo de Actual = 0."""

    mape: float
    wmape: float
    mae: float
    rmse: float
    bias: float
    n_obs: int = 0
    n_mape_excluded: int = 0
    excluded_zero_actual: bool = False

    def to_dict(self) -> dict:
        return {
            "mape": self.mape,
            "wmape": self.wmape,
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "n_obs": self.n_obs,
            "n_mape_excluded": self.n_mape_excluded,
        }

    @property
    def mape_note(self) -> str:
        if not self.excluded_zero_actual:
            return "MAPE calculado sobre todas las observaciones (sin Actual=0)."
        return (
            f"MAPE excluye {self.n_mape_excluded} observaciones con Actual=0 "
            f"(nunca se sustituye el cero); MAE/WMAPE las incluyen."
        )


def compute_metrics(
    actual: pd.Series | np.ndarray | list,
    forecast: pd.Series | np.ndarray | list,
) -> MetricResult:
    a = pd.to_numeric(pd.Series(np.asarray(actual, dtype=float).ravel()), errors="coerce").to_numpy()
    f = pd.to_numeric(pd.Series(np.asarray(forecast, dtype=float).ravel()), errors="coerce").to_numpy()
    if len(a) != len(f):
        raise ValueError(f"longitudes distintas: actual={len(a)} forecast={len(f)}")

    mask = np.isfinite(a) & np.isfinite(f)
    a, f = a[mask], f[mask]
    n = len(a)
    if n == 0:
        raise ValueError("sin observaciones válidas para calcular métricas")

    abs_err = np.abs(f - a)
    sq_err = (f - a) ** 2

    # MAPE: excluir Actual = 0 (documentar), nunca reemplazar el cero
    nonzero = a != 0
    n_excluded = int((~nonzero).sum())
    mape = float(np.mean(abs_err[nonzero] / a[nonzero]) * 100) if nonzero.any() else float("nan")

    # WMAPE: Σ|A-F| / ΣA (sobre el conjunto completo)
    denom = float(np.sum(np.abs(a)))
    wmape = float(np.sum(abs_err) / denom) if denom != 0 else float("nan")

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    bias = float(np.sum(f - a) / denom) if denom != 0 else float("nan")

    return MetricResult(
        mape=mape,
        wmape=wmape,
        mae=mae,
        rmse=rmse,
        bias=bias,
        n_obs=n,
        n_mape_excluded=n_excluded,
        excluded_zero_actual=n_excluded > 0,
    )


def accuracy_band(mape: float) -> str:
    """Referencias Lighthouse (spec evaluation): referencia, no objetivo rígido."""
    if not np.isfinite(mape):
        return "sin datos"
    if mape < 10:
        return "excelente (<10%)"
    if mape < 20:
        return "generalmente aceptable (10-20%)"
    if mape < 25:
        return "razonable en propiedades estacionales (15-25%)"
    return "fuera de las referencias (>=25%)"


def bias_direction(bias: float) -> str:
    """Dirección del error: separada de la precisión (spec evaluation)."""
    if not np.isfinite(bias):
        return "sin datos"
    if bias > 0.001:
        return "over-forecast (sobre-pronostica)"
    if bias < -0.001:
        return "under-forecast (sub-pronostica)"
    return "sin sesgo"


# ---------------------------------------------------------------------------
# Prueba de significancia Diebold-Mariano (empate estadístico entre candidatos)
# ---------------------------------------------------------------------------


@dataclass
class DMResult:
    """Resultado de la prueba Diebold-Mariano entre dos candidatos."""

    dm: float
    pvalue: float
    significant: bool
    better: str | None  # nombre del candidato con menor pérdida (si hay significancia)
    n_obs: int

    def to_dict(self) -> dict:
        return {
            "dm": self.dm,
            "pvalue": self.pvalue,
            "significant": self.significant,
            "better": self.better,
            "n_obs": self.n_obs,
        }


def diebold_mariano(
    actual: np.ndarray | pd.Series,
    forecast_a: np.ndarray | pd.Series,
    forecast_b: np.ndarray | pd.Series,
    horizon: int = 1,
    alpha: float = 0.05,
) -> DMResult:
    """Diebold-Mariano con pérdida absoluta y corrección HAC + Harvey.

    Compara la pérdida |e_a| vs |e_b| sobre las MISMAS observaciones (pares
    alineados; NaN excluidos). ``horizon`` = pasos hacia adelante de la
    predicción → corrección de autocorrelación hasta lag ``horizon-1`` y
    ajuste de tamaño muestral de Harvey (1997).
    """
    a = np.asarray(actual, dtype=float)
    fa = np.asarray(forecast_a, dtype=float)
    fb = np.asarray(forecast_b, dtype=float)
    if not (len(a) == len(fa) == len(fb)):
        raise ValueError(f"longitudes distintas: actual={len(a)} a={len(fa)} b={len(fb)}")
    mask = np.isfinite(a) & np.isfinite(fa) & np.isfinite(fb)
    a, fa, fb = a[mask], fa[mask], fb[mask]
    n = len(a)
    if n < 5:
        return DMResult(
            dm=float("nan"), pvalue=float("nan"), significant=False,
            better=None, n_obs=n,
        )

    d = np.abs(fa - a) - np.abs(fb - a)  # >0 ⇒ a pierde más
    dbar = float(np.mean(d))
    ddof = d - dbar
    gamma0 = float(np.dot(ddof, ddof) / n)
    h = max(int(horizon), 1)
    hac = gamma0
    for k in range(1, min(h, n - 1)):
        gamma_k = float(np.dot(ddof[k:], ddof[:-k]) / n)
        hac += 2.0 * (1.0 - k / h) * gamma_k
    var_dbar = hac / n if hac > 0 else float("nan")
    if not np.isfinite(var_dbar) or var_dbar <= 0:
        return DMResult(
            dm=float("nan"), pvalue=float("nan"), significant=False,
            better=None, n_obs=n,
        )
    dm = dbar / np.sqrt(var_dbar)
    # corrección de pequeño muestra (Harvey, Leybourne & Newbold 1997)
    correction = (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm_hln = dm * np.sqrt(correction) if correction > 0 else dm
    pvalue = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_hln) / np.sqrt(2.0)))))
    # tie-break por pérdida media (menor |e| gana) cuando es significativo
    better = None
    if pvalue < alpha:
        better = "b" if dbar > 0 else "a"  # dbar>0 ⇒ |e_a|>|e_b| ⇒ b mejor
    return DMResult(
        dm=float(dm_hln), pvalue=pvalue, significant=bool(pvalue < alpha),
        better=better, n_obs=n,
    )
