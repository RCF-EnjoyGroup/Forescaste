"""
Per-horizon split-conformal uncertainty bands.

Point forecasts carry no uncertainty; businesses need ranges. This module
builds empirical bands from the winning model's pooled backtest residuals,
one width per horizon step (uncertainty grows with the horizon — the bands
must reflect that, unlike constant-width intervals).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def per_horizon_conformal_bands(
    residuals: np.ndarray,
    forecast: np.ndarray,
    alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build asymmetric conformal bands around a forecast.

    Parameters
    ----------
    residuals : np.ndarray
        Signed residuals (actual - prediction) with shape
        (n_windows, horizon) from rolling-origin backtesting.
    forecast : np.ndarray
        Point forecast of length `horizon`.
    alpha : float
        Miscoverage level: alpha=0.2 -> nominal 80% bands (P10/P90).

    Returns
    -------
    tuple of (lower, upper)
        Arrays of length `horizon` (band = forecast + residual quantiles).
    """
    residuals = np.asarray(residuals, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if residuals.ndim != 2:
        raise ValueError(f"residuals must be 2-D (n_windows, horizon); got {residuals.shape}")
    if residuals.shape[1] != len(forecast):
        raise ValueError(
            f"horizon mismatch: residuals have {residuals.shape[1]} steps, "
            f"forecast has {len(forecast)}."
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")

    finite = residuals[:, :]
    if finite.size == 0 or not np.isfinite(finite).all():
        raise ValueError("residuals contain non-finite values.")

    # Guard: with very few windows, quantiles are coarse but still valid
    n_windows = finite.shape[0]
    if n_windows < 4:
        logger.warning(
            "Only %d backtest windows: conformal quantiles are coarse "
            "(prefer >= 10 windows for stable bands).", n_windows,
        )

    q_low = np.quantile(finite, alpha / 2, axis=0)
    q_high = np.quantile(finite, 1 - alpha / 2, axis=0)

    lower = forecast + q_low
    upper = forecast + q_high
    logger.info(
        "Conformal bands (alpha=%.2f): mean width=%.2f | step-1 width=%.2f | "
        "final-step width=%.2f",
        alpha, float(np.mean(upper - lower)),
        float(upper[0] - lower[0]), float(upper[-1] - lower[-1]),
    )
    return lower, upper


def empirical_coverage(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of actuals inside [lower, upper]."""
    actuals = np.asarray(actuals, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (len(actuals) == len(lower) == len(upper)):
        raise ValueError("actuals, lower and upper must have the same length.")
    return float(np.mean((actuals >= lower) & (actuals <= upper)))


def summarize_bands(
    dates: pd.DatetimeIndex,
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    name: str = "forecast",
) -> pd.DataFrame:
    """Publication-ready band summary (per-period forecast with bounds)."""
    return pd.DataFrame(
        {
            "date": dates,
            name: np.asarray(forecast, dtype=float),
            "lower": np.asarray(lower, dtype=float),
            "upper": np.asarray(upper, dtype=float),
            "width": np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float),
        }
    )
