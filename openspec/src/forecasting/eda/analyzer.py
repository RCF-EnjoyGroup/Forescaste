"""
EDA analyzer: descriptive statistics, stationarity tests, seasonality detection,
outlier detection, and autocorrelation analysis for hotel revenue time series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StationarityResult:
    test_name: str
    statistic: float
    p_value: float
    critical_values: dict[str, float]
    is_stationary: bool
    interpretation: str


@dataclass
class SeasonalityResult:
    pattern: str  # "weekly", "monthly", "yearly"
    period: int
    strength: float
    p_value: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlierResult:
    method: str
    n_outliers: int
    outlier_pct: float
    outlier_indices: pd.Index
    bounds: dict[str, float]


@dataclass
class DecompositionResult:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    model_type: str  # "additive" or "multiplicative"
    period: int


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class EDAAnalyzer:
    """
    Comprehensive EDA for hotel revenue time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'revenue', 'room_revenue' columns.
    date_col : str
        Name of the date column.
    target_cols : list[str]
        Columns to analyze (default: revenue, room_revenue).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_cols: Optional[list[str]] = None,
    ):
        self.df = df.copy()
        self.date_col = date_col
        self.target_cols = target_cols or ["revenue", "room_revenue"]

        # Ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col).set_index(date_col)

    # ------------------------------------------------------------------
    # 1. Descriptive statistics
    # ------------------------------------------------------------------

    def descriptive_statistics(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for target columns.

        Returns
        -------
        pd.DataFrame
            Statistics table with mean, median, std, min, max, quartiles, skewness, kurtosis.
        """
        stats_list = []
        for col in self.target_cols:
            if col not in self.df.columns:
                logger.warning("Column '%s' not found in data.", col)
                continue

            s = self.df[col].dropna()
            row = {
                "column": col,
                "count": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "q25": s.quantile(0.25),
                "q50": s.quantile(0.50),
                "q75": s.quantile(0.75),
                "max": s.max(),
                "skewness": s.skew(),
                "kurtosis": s.kurtosis(),
                "cv": s.std() / s.mean() if s.mean() != 0 else np.nan,
            }
            stats_list.append(row)

        result = pd.DataFrame(stats_list).set_index("column")
        logger.info("Descriptive statistics computed for %d columns.", len(stats_list))
        return result

    # ------------------------------------------------------------------
    # 2. Missing value analysis
    # ------------------------------------------------------------------

    def missing_value_analysis(self) -> pd.DataFrame:
        """
        Analyze missing values per column.

        Returns
        -------
        pd.DataFrame
            Missing value counts, percentages, and patterns.
        """
        total = len(self.df)
        missing = self.df.isnull().sum()
        pct = (missing / total * 100).round(2)

        result = pd.DataFrame({
            "column": missing.index,
            "missing_count": missing.values,
            "missing_pct": pct.values,
            "dtype": [str(self.df[c].dtype) for c in missing.index],
        })
        result = result[result["missing_count"] > 0].sort_values("missing_pct", ascending=False)

        if result.empty:
            logger.info("No missing values detected.")
        else:
            logger.info("Found missing values in %d columns.", len(result))

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Time series decomposition
    # ------------------------------------------------------------------

    def time_series_decomposition(
        self,
        column: str = "revenue",
        period: int = 7,
        model: str = "additive",
    ) -> DecompositionResult:
        """
        Perform seasonal decomposition of a time series.

        Parameters
        ----------
        column : str
            Column to decompose.
        period : int
            Seasonal period (7 for weekly, 365 for yearly).
        model : str
            'additive' or 'multiplicative'.

        Returns
        -------
        DecompositionResult
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        series = self.df[column].dropna()

        # Need at least 2 full periods
        if len(series) < 2 * period:
            logger.warning(
                "Only %d observations for period=%d. Using period=%d instead.",
                len(series), period, max(2, len(series) // 2),
            )
            period = max(2, len(series) // 2)

        # Handle zeros for multiplicative model
        if model == "multiplicative" and (series <= 0).any():
            logger.warning("Non-positive values found. Switching to additive model.")
            model = "additive"

        decomposition = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")

        result = DecompositionResult(
            trend=decomposition.trend,
            seasonal=decomposition.seasonal,
            residual=decomposition.resid,
            model_type=model,
            period=period,
        )
        logger.info("Decomposition complete: model=%s, period=%d", model, period)
        return result

    # ------------------------------------------------------------------
    # 4. Stationarity tests
    # ------------------------------------------------------------------

    def stationarity_tests(
        self, column: str = "revenue"
    ) -> list[StationarityResult]:
        """
        Run ADF and KPSS stationarity tests.

        Parameters
        ----------
        column : str
            Column to test.

        Returns
        -------
        list[StationarityResult]
        """
        from statsmodels.tsa.stattools import adfuller, kpss

        series = self.df[column].dropna()
        results = []

        # ADF test
        adf = adfuller(series, autolag="AIC")
        adf_result = StationarityResult(
            test_name="Augmented Dickey-Fuller",
            statistic=adf[0],
            p_value=adf[1],
            critical_values=adf[4],
            is_stationary=adf[1] < 0.05,
            interpretation=(
                "Series IS stationary (reject unit root null)"
                if adf[1] < 0.05
                else "Series is NOT stationary (fail to reject unit root null)"
            ),
        )
        results.append(adf_result)

        # KPSS test
        try:
            kpss_stat, p_val, n_lags, crit_vals = kpss(series, regression="c", nlags="auto")
            kpss_result = StationarityResult(
                test_name="KPSS",
                statistic=kpss_stat,
                p_value=p_val,
                critical_values=crit_vals,
                is_stationary=p_val > 0.05,
                interpretation=(
                    "Series IS stationary (fail to reject stationarity null)"
                    if p_val > 0.05
                    else "Series is NOT stationary (reject stationarity null)"
                ),
            )
        except Exception as e:
            logger.warning("KPSS test failed: %s", e)
            kpss_result = StationarityResult(
                test_name="KPSS",
                statistic=np.nan,
                p_value=np.nan,
                critical_values={},
                is_stationary=False,
                interpretation=f"KPSS test failed: {e}",
            )
        results.append(kpss_result)

        for r in results:
            logger.info(
                "%s: stat=%.4f, p=%.4f — %s",
                r.test_name, r.statistic, r.p_value, r.interpretation,
            )
        return results

    # ------------------------------------------------------------------
    # 5. Autocorrelation analysis
    # ------------------------------------------------------------------

    def autocorrelation_analysis(
        self, column: str = "revenue", nlags: int = 60
    ) -> dict[str, Any]:
        """
        Compute ACF and PACF values for the time series.

        Parameters
        ----------
        column : str
            Column to analyze.
        nlags : int
            Number of lags.

        Returns
        -------
        dict
            acf_values, pacf_values, confint_acf, confint_pacf, significant_lags_acf, significant_lags_pacf
        """
        from statsmodels.tsa.stattools import acf, pacf

        series = self.df[column].dropna()
        acf_vals = acf(series, nlags=nlags, fft=True)
        pacf_vals = pacf(series, nlags=nlags)

        confint = 1.96 / np.sqrt(len(series))

        sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > confint]
        sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > confint]

        result = {
            "acf_values": acf_vals,
            "pacf_values": pacf_vals,
            "confint": confint,
            "significant_lags_acf": sig_acf,
            "significant_lags_pacf": sig_pacf,
        }

        logger.info(
            "ACF/PACF computed: %d significant ACF lags, %d significant PACF lags.",
            len(sig_acf), len(sig_pacf),
        )
        return result

    # ------------------------------------------------------------------
    # 6. Seasonality detection
    # ------------------------------------------------------------------

    def seasonality_detection(
        self, column: str = "revenue"
    ) -> list[SeasonalityResult]:
        """
        Detect seasonal patterns at weekly, monthly, and yearly granularities.

        Returns
        -------
        list[SeasonalityResult]
        """
        from scipy.stats import f_oneway

        series = self.df[column].dropna()
        results = []

        # Weekly seasonality (day-of-week effect)
        if len(series) >= 14:
            dow_groups = [group.values for _, group in series.groupby(series.index.dayofweek)]
            if all(len(g) > 1 for g in dow_groups) and len(dow_groups) > 1:
                f_stat, p_val = f_oneway(*dow_groups)
                strength = min(1.0, f_stat / (f_stat + len(series)))
                results.append(SeasonalityResult(
                    pattern="weekly", period=7, strength=strength,
                    p_value=p_val,
                    details={"f_statistic": f_stat, "day_means": series.groupby(series.index.dayofweek).mean().to_dict()},
                ))

        # Monthly seasonality
        if len(series) >= 60:
            month_groups = [group.values for _, group in series.groupby(series.index.month)]
            valid_groups = [g for g in month_groups if len(g) > 1]
            if len(valid_groups) > 2:
                f_stat, p_val = f_oneway(*valid_groups)
                strength = min(1.0, f_stat / (f_stat + len(series)))
                results.append(SeasonalityResult(
                    pattern="monthly", period=12, strength=strength,
                    p_value=p_val,
                    details={"f_statistic": f_stat},
                ))

        # Yearly seasonality (rough)
        if len(series) >= 730:
            year_groups = [group.values for _, group in series.groupby(series.index.year)]
            valid_groups = [g for g in year_groups if len(g) > 30]
            if len(valid_groups) > 1:
                f_stat, p_val = f_oneway(*valid_groups)
                strength = min(1.0, f_stat / (f_stat + len(series)))
                results.append(SeasonalityResult(
                    pattern="yearly", period=365, strength=strength,
                    p_value=p_val,
                    details={"f_statistic": f_stat},
                ))

        for r in results:
            logger.info(
                "Seasonality — %s: strength=%.3f, p=%.4f",
                r.pattern, r.strength, r.p_value or 0,
            )
        return results

    # ------------------------------------------------------------------
    # 7. Outlier detection
    # ------------------------------------------------------------------

    def outlier_detection(
        self,
        column: str = "revenue",
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> OutlierResult:
        """
        Detect outliers using IQR or z-score method.

        Parameters
        ----------
        column : str
            Column to analyze.
        method : str
            'iqr' or 'zscore'.
        threshold : float
            IQR multiplier or z-score threshold.

        Returns
        -------
        OutlierResult
        """
        series = self.df[column].dropna()

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (series < lower) | (series > upper)
        elif method == "zscore":
            z = np.abs(stats.zscore(series))
            mask = z > threshold
            lower = series.mean() - threshold * series.std()
            upper = series.mean() + threshold * series.std()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

        n_outliers = mask.sum()
        result = OutlierResult(
            method=method,
            n_outliers=int(n_outliers),
            outlier_pct=round(n_outliers / len(series) * 100, 2),
            outlier_indices=series[mask].index,
            bounds={"lower": float(lower), "upper": float(upper)},
        )

        logger.info(
            "Outlier detection (%s): %d outliers (%.2f%%)",
            method, n_outliers, result.outlier_pct,
        )
        return result

    # ------------------------------------------------------------------
    # 8. Correlation analysis
    # ------------------------------------------------------------------

    def correlation_analysis(
        self, columns: Optional[list[str]] = None, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for specified columns.

        Parameters
        ----------
        columns : list[str], optional
            Columns to correlate. Defaults to target_cols.
        method : str
            'pearson' or 'spearman'.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        cols = columns or self.target_cols
        available = [c for c in cols if c in self.df.columns]
        corr = self.df[available].corr(method=method)
        logger.info("Correlation matrix computed (%s) for %d columns.", method, len(available))
        return corr
