"""
Recursive multi-step forecasting for feature-based (GBM) models.

This module implements the honest evaluation protocol for ML forecasters:
predictions are generated one step at a time and fed back as lag/rolling
inputs. Test-period actuals are NEVER used as model inputs, so the reported
metrics reflect real multi-step-ahead forecasting performance, not optimistic
one-step-ahead numbers.

Two code paths, identical guarantees:
- **Incremental path** (default): history is transformed once; each future row
  is built with O(1)/O(window) lookups on the values array instead of
  re-transforming the whole context per step. On the first step it is verified
  feature-by-feature against the reference path, and the function silently
  falls back if anything differs — speed is never traded for correctness.
- **Reference path** (`use_incremental=False`): transforms the full context at
  every step. Slower, kept as the auditable ground truth for tests.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.forecasting.preprocessing.features import TemporalFeatureEngineer
from src.forecasting.preprocessing.holidays import HolidayFeatureEngineer

logger = logging.getLogger(__name__)

_CALENDAR_COLS = (
    "year", "quarter", "month", "week", "day", "dayofweek", "dayofyear",
    "is_weekend", "is_month_start", "is_month_end",
    "is_quarter_start", "is_quarter_end",
)
_CYCLICAL_PERIODS = {"dayofweek": 7, "dayofyear": 365, "month": 12, "week": 52, "quarter": 4, "day": 31}
_ROLLING_SUFFIX = {"mean": "rmean", "std": "rstd", "min": "rmin", "max": "rmax"}
_QUIET_LOGGERS = (
    "src.forecasting.preprocessing.features",
    "src.forecasting.preprocessing.holidays",
)


class _QuietLogs:
    """Temporarily silence per-step INFO spam from the feature builders."""

    def __enter__(self) -> "_QuietLogs":
        self._saved = {name: logging.getLogger(name).level for name in _QUIET_LOGGERS}
        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc) -> None:
        for name, level in self._saved.items():
            logging.getLogger(name).setLevel(level)


def _values_match(a: object, b: object, tol: float = 1e-6) -> bool:
    """NaN-aware comparison with relative tolerance."""
    a_nan, b_nan = pd.isna(a), pd.isna(b)
    if bool(a_nan) or bool(b_nan):
        return bool(a_nan) and bool(b_nan)
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    except (TypeError, ValueError):
        return a == b


class _IncrementalRowBuilder:
    """Builds the feature row for the NEXT date without re-transforming history.

    Semantics replicate TemporalFeatureEngineer + HolidayFeatureEngineer exactly
    (same shifts, same min_periods, same ddof=1 std, same holiday windows).
    Any column it cannot reproduce raises KeyError, which triggers the
    reference-path fallback.
    """

    def __init__(
        self,
        feature_engineer: TemporalFeatureEngineer,
        holiday_engineer: Optional[HolidayFeatureEngineer],
        target_col: str,
        values: np.ndarray,
        future_known: Optional[dict[str, float]] = None,
        future_series: Optional[dict[str, pd.Series]] = None,
    ):
        self.eng = feature_engineer
        self.he = holiday_engineer
        self.target_col = target_col
        self.values: list[float] = [float(v) for v in values]
        # Forward-known passthrough columns (e.g., available_rooms): their
        # future value is the last observed value (planned capacity).
        self.future_known: dict[str, float] = dict(future_known or {})
        # Forward-known per-DATE covariates (e.g., rotb_d90 from current books):
        # value for a future date = the covariate series at that date.
        self.future_series: dict[str, pd.Series] = dict(future_series or {})

        if holiday_engineer is not None:
            self.pre_days = holiday_engineer.pre_days
            self.post_days = holiday_engineer.post_days
            self.holiday_dates: Optional[pd.DatetimeIndex] = holiday_engineer._holiday_dates
            self.custom_events = {
                name: pd.DatetimeIndex(pd.to_datetime(dates))
                for name, dates in (holiday_engineer.custom_events or {}).items()
            }
        else:
            self.pre_days, self.post_days = 0, 0
            self.holiday_dates = None
            self.custom_events = {}

    # -- public API ----------------------------------------------------------

    def append(self, value: float) -> None:
        """Feed a prediction back as if it were an observed value."""
        self.values.append(float(value))

    def build_row(self, date: pd.Timestamp, columns: list[str]) -> dict[str, object]:
        """Feature values for `date` (position = len(values))."""
        row: dict[str, object] = {}
        cal = self._calendar(date)
        for col in columns:
            row[col] = self._compute(col, date, cal)
        return row

    def verify(self, probe_row: pd.Series, date: pd.Timestamp, columns: list[str]) -> bool:
        """Every computed feature must match the reference transform exactly."""
        try:
            fast = self.build_row(date, columns)
        except Exception as e:  # noqa: BLE001 - any incompatibility -> fallback
            logger.info("Incremental path rejected (%s); using reference path.", e)
            return False
        for col in columns:
            if not _values_match(fast.get(col), probe_row[col]):
                logger.info(
                    "Incremental path rejected: column '%s' differs from reference "
                    "(fast=%r, ref=%r); using reference path.",
                    col, fast.get(col), probe_row[col],
                )
                return False
        return True

    # -- internals -----------------------------------------------------------

    def _compute(self, col: str, date: pd.Timestamp, cal: dict[str, int]) -> object:
        # Forward-known per-DATE covariates (books, planned events)
        if col in self.future_series:
            series = self.future_series[col]
            if date not in series.index:
                raise ValueError(
                    f"Future covariate '{col}' has no value for {date.date()}. "
                    "Covariates must cover every forecast date."
                )
            return float(series.loc[date])

        # Forward-known passthrough columns (planned capacity etc.)
        if col in self.future_known:
            return self.future_known[col]

        if col in cal:
            return cal[col]

        # Cyclical encodings (only period-map features are replicable)
        if col.endswith("_sin") or col.endswith("_cos"):
            base = col[: -len("_sin")] if col.endswith("_sin") else col[: -len("_cos")]
            if base in self.eng.cyclical_features and base in _CYCLICAL_PERIODS:
                period = _CYCLICAL_PERIODS[base]
                angle = 2.0 * math.pi * cal[base] / period
                return math.sin(angle) if col.endswith("_sin") else math.cos(angle)
            raise KeyError(col)

        # Holiday features
        if col == "is_holiday":
            return int(self.holiday_dates is not None and date in self.holiday_dates)
        if col == "is_pre_holiday":
            return int(self._holiday_flag(date, self.pre_days, "pre"))
        if col == "is_post_holiday":
            return int(self._holiday_flag(date, self.post_days, "post"))
        if col == "days_to_holiday":
            if self.holiday_dates is None or len(self.holiday_dates) == 0:
                return 365
            future = self.holiday_dates[self.holiday_dates >= date]
            return int((future[0] - date).days) if len(future) > 0 else 365
        if col.startswith("is_") and col[3:] in self.custom_events:
            return int(date in self.custom_events[col[3:]])

        prefix = f"{self.target_col}_"
        if not col.startswith(prefix):
            raise KeyError(col)
        tail = col[len(prefix):]

        t = len(self.values)

        # Lags: revenue_lag_{L}
        if tail.startswith("lag_"):
            lag = int(tail[4:])
            idx = t - lag
            return self.values[idx] if idx >= 0 else np.nan

        # Rolling: revenue_r{stat}_{W} -> stats over the W values BEFORE this position
        for stat, suffix in _ROLLING_SUFFIX.items():
            if stat in self.eng.rolling_stats and tail.startswith(f"{suffix}_"):
                window = int(tail[len(suffix) + 1:])
                if window not in self.eng.rolling_windows:
                    raise KeyError(col)
                sl = self.values[max(0, t - window):t]
                if not sl:
                    return np.nan
                if stat == "mean":
                    return float(np.mean(sl))
                if stat == "std":
                    return float(np.std(sl, ddof=1)) if len(sl) >= 2 else np.nan
                if stat == "min":
                    return float(np.min(sl))
                return float(np.max(sl))

        # Expanding: revenue_emean / revenue_estd -> stats over ALL previous values
        if tail == "emean" and "mean" in self.eng.expanding_stats:
            return float(np.mean(self.values[:t])) if t > 0 else np.nan
        if tail == "estd" and "std" in self.eng.expanding_stats:
            return float(np.std(self.values[:t], ddof=1)) if t >= 2 else np.nan

        raise KeyError(col)

    def _calendar(self, d: pd.Timestamp) -> dict[str, int]:
        iso = d.isocalendar()
        return {
            "year": d.year,
            "quarter": d.quarter,
            "month": d.month,
            "week": int(iso.week),
            "day": d.day,
            "dayofweek": d.dayofweek,
            "dayofyear": d.dayofyear,
            "is_weekend": int(d.dayofweek >= 5),
            "is_month_start": int(d.is_month_start),
            "is_month_end": int(d.is_month_end),
            "is_quarter_start": int(d.is_quarter_start),
            "is_quarter_end": int(d.is_quarter_end),
        }

    def _holiday_flag(self, date: pd.Timestamp, days: int, side: str) -> bool:
        if self.holiday_dates is None:
            return False
        pre_td = pd.Timedelta(days=days)
        post_td = pd.Timedelta(days=days)
        for hdate in self.holiday_dates:
            if side == "pre" and (hdate - pre_td <= date < hdate):
                return True
            if side == "post" and (hdate < date <= hdate + post_td):
                return True
        return False


def recursive_forecast(
    model,
    history_df: pd.DataFrame,
    future_dates: Iterable,
    target_col: str = "revenue",
    date_col: str = "date",
    feature_cols: Optional[list[str]] = None,
    feature_engineer: Optional[TemporalFeatureEngineer] = None,
    holiday_engineer: Optional[HolidayFeatureEngineer] = None,
    use_incremental: bool = True,
    future_known_cols: Optional[list[str]] = None,
    future_covariates: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Generate leakage-free multi-step predictions with a fitted GBM model.

    Parameters
    ----------
    model : BaseForecaster
        Fitted model (LightGBM/XGBoost/CatBoost forecaster).
    history_df : pd.DataFrame
        Observed daily series with at least [date_col, target_col]. Must end
        before the first date in `future_dates`.
    future_dates : Iterable
        Dates to predict, in order.
    target_col, date_col : str
        Column names.
    feature_cols : list[str]
        Exact feature columns used at training time.
    feature_engineer : TemporalFeatureEngineer
        Configured exactly as in training (same lags/windows/stats).
    holiday_engineer : HolidayFeatureEngineer, optional
        Holiday feature builder used in training.
    use_incremental : bool
        Use the verified fast path (default). Pass False to force the
        reference full-transform path (auditable, slower).
    future_known_cols : list[str], optional
        Columns whose future values are known/planned (e.g., available_rooms).
        They are carried through the recursion with their last observed value.
    future_covariates : pd.DataFrame, optional
        Forward-known per-DATE covariates (e.g., on-the-books rotb_d90 from
        current reservations). Must contain `date_col` plus one column per
        covariate, with a value for every forecast date (missing dates raise).

    Returns
    -------
    np.ndarray
        One prediction per future date.
    """
    if feature_cols is None:
        raise ValueError("feature_cols is required for recursive forecasting.")
    if feature_engineer is None:
        raise ValueError("feature_engineer is required for recursive forecasting.")

    future_known_cols = list(future_known_cols or [])
    keep_cols = [c for c in [date_col, target_col] + future_known_cols if c]

    history = history_df[keep_cols].copy()
    history[date_col] = pd.to_datetime(history[date_col])
    history = history.sort_values(date_col).dropna(subset=[target_col]).reset_index(drop=True)

    future_dates = pd.to_datetime(list(future_dates))
    if len(future_dates) == 0:
        return np.array([])

    first_date = future_dates[0]
    if history[date_col].max() >= first_date:
        raise ValueError(
            f"History must end before the first forecast date. "
            f"History ends at {history[date_col].max()}, first forecast date {first_date}."
        )

    # Last observed value for each forward-known column (planned capacity)
    fk: dict[str, float] = {}
    for col in future_known_cols:
        last = history[col].dropna()
        if last.empty:
            raise ValueError(f"Forward-known column '{col}' has no observed values.")
        fk[col] = float(last.iloc[-1])

    # Forward-known per-DATE covariates: {col: Series indexed by date}
    future_series: dict[str, pd.Series] = {}
    if future_covariates is not None:
        cov = future_covariates.copy()
        cov[date_col] = pd.to_datetime(cov[date_col])
        if cov[date_col].duplicated().any():
            raise ValueError(f"future_covariates has duplicated dates in '{date_col}'.")
        cov = cov.set_index(date_col).sort_index()
        future_series = {col: cov[col].astype(float) for col in cov.columns}
        missing = [d for d in future_dates if not all(d in s.index for s in future_series.values())]
        if missing:
            raise ValueError(
                f"future_covariates missing values for {len(missing)} forecast dates "
                f"(first: {pd.Timestamp(missing[0]).date()}). Covariates must cover "
                "every forecast date."
            )

    # Outlier flag columns produced by the pipeline are unknown for future
    # rows; assume "not an outlier" (0), matching the flag semantics.
    outlier_cols = [c for c in feature_cols if c.endswith("_outlier")]

    # ------------------------------------------------------------------
    # Fast incremental path: verify against the reference on the first step
    # ------------------------------------------------------------------
    builder = None
    builder_columns: list[str] = []
    if use_incremental:
        try:
            # Build the reference probe first (this also fits the holiday
            # calendar if it was not fitted yet), then the incremental builder
            # sees the exact same state the reference path will use.
            placeholder = pd.DataFrame({
                date_col: [first_date], target_col: [np.nan],
                **{c: [fk[c]] for c in future_known_cols},
                **{c: [future_series[c].loc[first_date]] for c in future_series},
            })
            probe_ctx = pd.concat([history, placeholder], ignore_index=True)
            probe_full = feature_engineer.transform(probe_ctx)
            if holiday_engineer is not None:
                probe_full = holiday_engineer.transform(probe_full)
            probe_row = probe_full.iloc[-1]
            builder_columns = [c for c in probe_full.columns if c not in (date_col, target_col)]

            candidate = _IncrementalRowBuilder(
                feature_engineer, holiday_engineer, target_col,
                history[target_col].to_numpy(dtype=np.float64),
                future_known=fk,
                future_series=future_series,
            )
            if candidate.verify(probe_row, first_date, builder_columns):
                builder = candidate
                logger.info(
                    "Incremental path verified (%d feature columns match the reference).",
                    len(builder_columns),
                )
        except Exception as e:  # noqa: BLE001 - any setup issue -> reference path
            logger.info("Incremental path unavailable (%s); using reference path.", e)

    n_total = len(future_dates)
    predictions: list[float] = []

    with _QuietLogs():
        # ------------------------------------------------------------------
        # Incremental loop
        # ------------------------------------------------------------------
        if builder is not None:
            for i, d in enumerate(future_dates):
                row_values = builder.build_row(d, builder_columns)
                for col in outlier_cols:
                    row_values[col] = 0
                row_values[date_col] = d
                row_values[target_col] = np.nan
                features_row = pd.DataFrame([row_values])
                result = model.predict(features_row, target_col, date_col, feature_cols)
                pred = float(np.atleast_1d(result.predictions)[0])
                predictions.append(pred)
                builder.append(pred)
                if (i + 1) % 50 == 0:
                    logger.info("Recursive forecast (incremental): %d/%d steps", i + 1, n_total)
        # ------------------------------------------------------------------
        # Reference loop (full transform per step)
        # ------------------------------------------------------------------
        else:
            history_loop = history.copy()
            for i, d in enumerate(future_dates):
                row = pd.DataFrame(
                    {date_col: [d], target_col: [np.nan],
                     **{c: [fk[c]] for c in future_known_cols},
                     **{c: [future_series[c].loc[d]] for c in future_series}}
                )
                context = pd.concat([history_loop, row], ignore_index=True)

                featured = feature_engineer.transform(context)
                if holiday_engineer is not None:
                    featured = holiday_engineer.transform(featured)

                features_row = featured.iloc[[-1]].copy()

                for col in outlier_cols:
                    if col not in features_row.columns:
                        features_row[col] = 0
                    else:
                        features_row[col] = features_row[col].fillna(0)

                result = model.predict(features_row, target_col, date_col, feature_cols)
                pred = float(np.atleast_1d(result.predictions)[0])
                predictions.append(pred)

                history_loop = pd.concat(
                    [history_loop, pd.DataFrame(
                        {date_col: [d], target_col: [pred],
                         **{c: [fk[c]] for c in future_known_cols}}
                    )],
                    ignore_index=True,
                )
                if (i + 1) % 50 == 0:
                    logger.info("Recursive forecast (reference): %d/%d steps", i + 1, n_total)

    logger.info("Recursive forecast complete: %d steps (%s path)",
                n_total, "incremental" if builder is not None else "reference")
    return np.array(predictions)
