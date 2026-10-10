"""Validación exclusivamente temporal (spec ``evaluation``, task 6.1).

Expanding / rolling-origin: cada ventana entrena SOLO con datos anteriores a
ella. El split aleatorio está prohibido (lo vetó ``src.validation.leakage``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeWindow:
    """Una ventana de evaluación: train [.., train_end) → val [val_start, val_end]."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.val_start, self.val_end)

    def validate(self) -> None:
        # train = [train_start, train_end) semiabierto; val = [val_start, val_end]
        # => val_start == train_end es correcto (sin solape)
        if self.val_start < self.train_end:
            raise ValueError(
                f"fold {self.fold}: val_start ({self.val_start.date()}) debe ser "
                f">= train_end ({self.train_end.date()})"
            )
        if self.train_start >= self.train_end:
            raise ValueError(f"fold {self.fold}: ventana de entrenamiento vacía")


def expanding_windows(
    df: pd.DataFrame,
    date_col: str = "stay_date",
    n_splits: int = 4,
    val_days: int = 90,
    min_train_days: int = 365,
    step_days: int | None = None,
    expanding: bool = True,
) -> list[TimeWindow]:
    """Genera ventanas rolling-origin / expanding.

    Parameters
    ----------
    expanding:
        ``True`` → train crece desde ``train_start`` fijo (expanding).
        ``False`` → train se desplaza con la ventana (rolling).
    """
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("sin fechas para generar ventanas")
    start, end = dates.min(), dates.max()
    step = step_days or val_days

    windows: list[TimeWindow] = []
    # primera ventana: [start, start + min_train) de train, luego val de val_days
    train_start = start
    cursor = start + pd.Timedelta(days=min_train_days)
    fold = 0
    while fold < n_splits and cursor + pd.Timedelta(days=val_days) <= end + pd.Timedelta(days=1):
        val_start = cursor
        val_end = cursor + pd.Timedelta(days=val_days)
        w = TimeWindow(
            fold=fold,
            train_start=train_start if expanding else cursor - pd.Timedelta(days=min_train_days),
            train_end=val_start,
            val_start=val_start,
            val_end=val_end,
        )
        w.validate()
        windows.append(w)
        fold += 1
        cursor += pd.Timedelta(days=step)

    if not windows:
        raise ValueError(
            f"insuficiente histórico para {n_splits} ventanas "
            f"(min_train={min_train_days}d, val={val_days}d)"
        )
    return windows


def slice_window(
    df: pd.DataFrame, window: TimeWindow, date_col: str = "stay_date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recorta train/val garantizando orden temporal estricto."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    train = df.loc[(dates >= window.train_start) & (dates < window.train_end)]
    val = df.loc[(dates >= window.val_start) & (dates < window.val_end)]
    if train.empty or val.empty:
        raise ValueError(f"fold {window.fold}: train o val vacío")
    if pd.to_datetime(val[date_col]).min() <= pd.to_datetime(train[date_col]).max():
        raise ValueError(f"fold {window.fold}: solape temporal train/val")
    return train, val
