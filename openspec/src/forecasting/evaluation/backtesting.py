"""
Rolling-origin (backtesting) evaluation for forecast rank stability.

A single hold-out split is one draw of chance: different windows can reorder
models. This module re-evaluates every model across multiple historical
origins, where each fold's forecast uses ONLY data observed strictly before
the fold's window start — the same honesty rule as the main evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    """One backtesting fold: a forecast window and its strictly-past history."""
    fold_id: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    history_end: pd.Timestamp  # last allowed history date (window_start - 1 day)


@dataclass
class BacktestResult:
    """Aggregated backtesting outcome across folds and models."""
    folds: list[Fold]
    mae_table: pd.DataFrame          # index=model, columns=fold_id, values=MAE
    mean_rank: pd.Series            # model -> mean rank across folds (1 = best)
    wins: pd.Series                 # model -> number of folds won
    residuals: dict[str, np.ndarray]  # model -> (n_folds, horizon) actual - pred
    horizon_days: int


def make_rolling_folds(
    last_window_end: pd.Timestamp,
    n_folds: int = 4,
    horizon_days: int = 120,
    step_days: Optional[int] = None,
) -> list[Fold]:
    """Build chronological rolling-origin folds ending at `last_window_end`.

    The most recent fold ends exactly at `last_window_end` (use the day before
    the test window so the last fold reuses the maximal history); earlier folds
    step back by `step_days` (default: non-overlapping windows).
    """
    if n_folds < 2:
        raise ValueError("Backtesting needs at least 2 folds.")
    if horizon_days < 7:
        raise ValueError("horizon_days must be at least 7.")
    step = step_days if step_days is not None else horizon_days

    folds = []
    for k in range(n_folds):
        window_end = pd.Timestamp(last_window_end) - pd.Timedelta(days=step * k)
        window_start = window_end - pd.Timedelta(days=horizon_days - 1)
        folds.append(
            Fold(
                fold_id=n_folds - 1 - k,  # chronological ids after reversal
                window_start=window_start,
                window_end=window_end,
                history_end=window_start - pd.Timedelta(days=1),
            )
        )
    folds.reverse()  # oldest first
    return folds


def rolling_origin_backtest(
    history_df: pd.DataFrame,
    forecasters: dict[str, Callable[[pd.DataFrame, pd.DatetimeIndex], np.ndarray]],
    folds: list[Fold],
    date_col: str = "date",
    target_col: str = "rooms_sold",
    fold_metrics: tuple[str, ...] = ("MAE",),
) -> BacktestResult:
    """Evaluate multiple forecasters across rolling-origin folds.

    Parameters
    ----------
    history_df : pd.DataFrame
        Full daily frame with [date_col, target_col] (+ any columns the
        forecasters need, e.g., forward-known capacity).
    forecasters : dict[str, Callable]
        {model_name: forecaster}. Each forecaster receives the STRICTLY
        pre-window history and the window dates, and returns predictions for
        those dates (aligned, same length).
    folds : list[Fold]
        From `make_rolling_folds`.
    """
    work = history_df.copy()
    work[date_col] = pd.to_datetime(work[date_col])

    horizon = int((folds[0].window_end - folds[0].window_start).days) + 1

    mae_rows: dict[str, dict[int, float]] = {name: {} for name in forecasters}
    residuals: dict[str, np.ndarray] = {
        name: np.full((len(folds), horizon), np.nan) for name in forecasters
    }

    for i, fold in enumerate(folds):
        hist_mask = work[date_col] < fold.window_start
        fold_hist = work.loc[hist_mask].reset_index(drop=True)
        win_mask = (work[date_col] >= fold.window_start) & (work[date_col] <= fold.window_end)
        fold_win = work.loc[win_mask].reset_index(drop=True)
        window_dates = pd.DatetimeIndex(fold_win[date_col])
        actuals = fold_win[target_col].to_numpy(dtype=float)

        if len(fold_hist) == 0 or len(fold_win) != horizon:
            raise ValueError(
                f"Fold {fold.fold_id}: history empty or window incomplete "
                f"({len(fold_win)}/{horizon} days). Check fold bounds vs data range."
            )

        for name, forecaster in forecasters.items():
            preds = np.asarray(
                forecaster(fold_hist, window_dates), dtype=float
            )
            if len(preds) != horizon or np.isnan(preds).any():
                raise ValueError(
                    f"Forecaster '{name}' returned invalid predictions for fold "
                    f"{fold.fold_id} (len={len(preds)}, NaN={np.isnan(preds).sum()})."
                )
            err = actuals - preds
            mae_rows[name][fold.fold_id] = float(np.abs(err).mean())
            residuals[name][i, :] = err

        logger.info(
            "Backtest fold %d: window %s -> %s | history ends %s | %d models",
            fold.fold_id, fold.window_start.date(), fold.window_end.date(),
            fold_hist[date_col].max().date(), len(forecasters),
        )

    mae_table = pd.DataFrame(mae_rows).T
    mae_table.columns.name = "fold"
    mae_table = mae_table[sorted(mae_table.columns)]

    # Ranks per fold (1 = lowest MAE) and win counts
    ranks = mae_table.rank(axis=0, method="min")
    mean_rank = ranks.mean(axis=1).sort_values()
    wins = (ranks == 1).sum(axis=1)  # wins per model (across folds)

    return BacktestResult(
        folds=folds,
        mae_table=mae_table,
        mean_rank=mean_rank,
        wins=wins,
        residuals=residuals,
        horizon_days=horizon,
    )
