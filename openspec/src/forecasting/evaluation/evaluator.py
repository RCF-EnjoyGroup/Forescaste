"""
Model evaluator: metrics computation, Diebold-Mariano test, model comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for computed metrics."""
    model_name: str
    metrics: dict[str, float]
    n_obs: int
    split: str = "test"  # "val" or "test"


@dataclass
class ComparisonResult:
    """Container for model comparison table."""
    results: list[MetricsResult]
    rankings: dict[str, list[str]]  # metric -> ranked model names
    best_model: str


class Evaluator:
    """
    Comprehensive model evaluation for time series forecasting.

    Metrics: MAE, RMSE, MAPE, sMAPE, MASE, WAPE.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        epsilon : float
            Small constant to avoid division by zero in percentage metrics.
        """
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        model_name: str = "",
        split: str = "test",
        naive_forecast: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """
        Compute all standard time series metrics.

        Parameters
        ----------
        actuals : np.ndarray
            True values.
        predictions : np.ndarray
            Predicted values.
        model_name : str
            Name of the model.
        split : str
            Which split this is ('val' or 'test').
        naive_forecast : np.ndarray, optional
            Naive/seasonal naive forecast for MASE denominator.

        Returns
        -------
        MetricsResult
        """
        actuals = np.asarray(actuals, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)

        # Align lengths
        min_len = min(len(actuals), len(predictions))
        actuals = actuals[:min_len]
        predictions = predictions[:min_len]

        # Remove NaN
        mask = ~(np.isnan(actuals) | np.isnan(predictions))
        actuals = actuals[mask]
        predictions = predictions[mask]

        if len(actuals) == 0:
            logger.warning("No valid observations for metric computation.")
            return MetricsResult(
                model_name=model_name,
                metrics={m: np.nan for m in ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "WAPE"]},
                n_obs=0,
                split=split,
            )

        errors = actuals - predictions
        abs_errors = np.abs(errors)

        # MAE
        mae = abs_errors.mean()

        # RMSE
        rmse = np.sqrt((errors ** 2).mean())

        # MAPE (handle zeros)
        pct_errors = abs_errors / np.maximum(np.abs(actuals), self.epsilon)
        mape = pct_errors.mean() * 100

        # sMAPE
        denom = (np.abs(actuals) + np.abs(predictions)) / 2
        smape = (abs_errors / np.maximum(denom, self.epsilon)).mean() * 100

        # MASE (scaled by naive forecast error)
        if naive_forecast is not None and len(naive_forecast) >= len(actuals):
            naive_err = np.abs(actuals - naive_forecast[:len(actuals)])
            mase_scale = naive_err.mean()
        elif len(actuals) > 1:
            # Fallback: seasonal naive (shift by 1)
            mase_scale = np.abs(np.diff(actuals)).mean()
        else:
            mase_scale = 1.0

        mase = mae / max(mase_scale, self.epsilon)

        # WAPE
        wape = abs_errors.sum() / max(np.abs(actuals).sum(), self.epsilon) * 100

        # Forecast bias (signed): positive = over-forecasting (predicting more
        # demand than materialized), negative = under-forecasting. MAPE/WAPE
        # only carry magnitude; direction is the operational diagnostic
        # (over-forecast -> overstaffing/discounting; under-forecast -> lost revenue).
        signed_error_sum = errors.sum()  # actuals - predictions
        bias_pct = -signed_error_sum / max(np.abs(actuals).sum(), self.epsilon) * 100
        # bias_pct > 0: predictions above actuals (over-forecast), < 0: under-forecast
        n_over = int((predictions > actuals).sum())
        n_under = int((predictions < actuals).sum())

        metrics = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4),
            "sMAPE": round(smape, 4),
            "MASE": round(mase, 4),
            "WAPE": round(wape, 4),
            "Bias%": round(bias_pct, 4),
            "Over_days": n_over,
            "Under_days": n_under,
        }

        logger.info(
            "%s (%s): MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%, WAPE=%.2f%%, Bias=%+.2f%% (%d over / %d under)",
            model_name, split, mae, rmse, mape, wape, bias_pct, n_over, n_under,
        )

        return MetricsResult(
            model_name=model_name,
            metrics=metrics,
            n_obs=len(actuals),
            split=split,
        )

    def compute_all_metrics(
        self,
        models_results: dict[str, tuple[np.ndarray, np.ndarray]],
        split: str = "test",
        naive_forecast: Optional[np.ndarray] = None,
    ) -> list[MetricsResult]:
        """
        Compute metrics for multiple models.

        Parameters
        ----------
        models_results : dict
            {model_name: (actuals, predictions)}.
        split : str
            Split name.
        naive_forecast : np.ndarray, optional
            Naive forecast for MASE.

        Returns
        -------
        list[MetricsResult]
        """
        results = []
        for name, (actuals, preds) in models_results.items():
            result = self.compute_metrics(actuals, preds, name, split, naive_forecast)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        results: list[MetricsResult],
        primary_metric: str = "MAE",
    ) -> ComparisonResult:
        """
        Build comparison table and rank models.

        Parameters
        ----------
        results : list[MetricsResult]
            Metrics for each model.
        primary_metric : str
            Metric to determine overall best model.

        Returns
        -------
        ComparisonResult
        """
        # Build comparison DataFrame
        rows = []
        for r in results:
            row = {"Model": r.model_name, "N_obs": r.n_obs, "Split": r.split}
            row.update(r.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Rank by each metric
        rankings = {}
        for metric in ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "WAPE"]:
            if metric in df.columns:
                ranked = df.sort_values(metric)["Model"].tolist()
                rankings[metric] = ranked

        # Best model by primary metric
        if primary_metric in df.columns:
            best = df.loc[df[primary_metric].idxmin(), "Model"]
        else:
            best = results[0].model_name if results else "unknown"

        logger.info("Best model by %s: %s", primary_metric, best)

        return ComparisonResult(
            results=results,
            rankings=rankings,
            best_model=best,
        )

    def comparison_table(self, results: list[MetricsResult]) -> pd.DataFrame:
        """Generate a formatted comparison DataFrame."""
        rows = []
        for r in results:
            row = {"Model": r.model_name, "N": r.n_obs}
            row.update(r.metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.set_index("Model")

    # ------------------------------------------------------------------
    # Diebold-Mariano test
    # ------------------------------------------------------------------

    def diebold_mariano_test(
        self,
        actuals: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        h: int = 1,
        power: int = 2,
    ) -> dict[str, Any]:
        """
        Diebold-Mariano test for predictive accuracy.

        Tests whether two models have significantly different forecast accuracy.

        Parameters
        ----------
        actuals : np.ndarray
            True values.
        pred_a, pred_b : np.ndarray
            Forecasts from models A and B.
        h : int
            Forecast horizon.
        power : int
            Loss function power (2 for squared error).

        Returns
        -------
        dict
            dm_statistic, p_value, significant (at alpha=0.05).
        """
        actuals = np.asarray(actuals, dtype=np.float64)
        pred_a = np.asarray(pred_a, dtype=np.float64)
        pred_b = np.asarray(pred_b, dtype=np.float64)

        min_len = min(len(actuals), len(pred_a), len(pred_b))
        actuals, pred_a, pred_b = actuals[:min_len], pred_a[:min_len], pred_b[:min_len]

        # Loss differential
        loss_a = np.abs(actuals - pred_a) ** power
        loss_b = np.abs(actuals - pred_b) ** power
        d = loss_a - loss_b

        n = len(d)
        d_mean = d.mean()

        # Newey-West HAC variance: multi-step forecast errors are
        # autocorrelated, so plain variance would understate it and inflate
        # significance. Lag truncation = h - 1 (h = forecast horizon).
        n_lags = max(0, int(h) - 1)
        d_centered = d - d_mean
        gamma0 = np.mean(d_centered ** 2)
        hac_var = gamma0
        for lag in range(1, min(n_lags, n - 1) + 1):
            weight = 1.0 - lag / (n_lags + 1)
            gamma_l = np.mean(d_centered[lag:] * d_centered[:-lag])
            hac_var += 2 * weight * gamma_l

        # DM statistic
        if hac_var <= 0:
            dm_stat = 0.0
            p_value = 1.0
        else:
            dm_stat = d_mean / np.sqrt(hac_var / n)
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))

        result = {
            "dm_statistic": round(dm_stat, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "n_obs": n,
        }

        logger.info("DM test: stat=%.4f, p=%.4f — %s",
                     dm_stat, p_value,
                     "Significant" if result["significant"] else "Not significant")
        return result

    # ------------------------------------------------------------------
    # Residual diagnostics
    # ------------------------------------------------------------------

    def residual_diagnostics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, Any]:
        """
        Perform residual analysis: normality, autocorrelation, heteroscedasticity.

        Returns
        -------
        dict
            Diagnostic test results.
        """
        from scipy import stats

        residuals = actuals - predictions
        n = len(residuals)

        # Normality: Jarque-Bera
        jb_stat, jb_pval = stats.jarque_bera(residuals)

        # Ljung-Box for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residuals, lags=min(10, n // 5), return_df=True)
        lb_pval = lb_result["lb_pvalue"].min()

        # Heteroscedasticity: Breusch-Pagan (simplified)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            import statsmodels.api as sm
            X = np.arange(n).reshape(-1, 1)
            X = sm.add_constant(X)
            bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X)
        except Exception:
            bp_stat, bp_pval = np.nan, np.nan

        result = {
            "jarque_bera": {"statistic": round(jb_stat, 4), "p_value": round(jb_pval, 4)},
            "ljung_box": {"min_p_value": round(lb_pval, 4)},
            "breusch_pagan": {"statistic": round(bp_stat, 4) if not np.isnan(bp_stat) else None,
                              "p_value": round(bp_pval, 4) if not np.isnan(bp_pval) else None},
            "residual_mean": round(float(residuals.mean()), 4),
            "residual_std": round(float(residuals.std()), 4),
        }

        logger.info("Residual diagnostics: JB p=%.4f, LB min_p=%.4f", jb_pval, lb_pval)
        return result

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_confidence_intervals(
        self,
        metric_values: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict[str, float]:
        """
        Compute bootstrap confidence intervals for a metric.

        Parameters
        ----------
        metric_values : np.ndarray
            Per-observation metric values.
        confidence : float
            Confidence level.
        n_bootstrap : int
            Number of bootstrap samples.

        Returns
        -------
        dict
            lower, upper, mean, std of the metric.
        """
        rng = np.random.default_rng(seed)
        n = len(metric_values)
        boot_means = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            sample = rng.choice(metric_values, size=n, replace=True)
            boot_means[i] = sample.mean()

        alpha = 1 - confidence
        lower = np.percentile(boot_means, alpha / 2 * 100)
        upper = np.percentile(boot_means, (1 - alpha / 2) * 100)

        return {
            "mean": round(float(boot_means.mean()), 4),
            "std": round(float(boot_means.std()), 4),
            "lower": round(float(lower), 4),
            "upper": round(float(upper), 4),
            "confidence": confidence,
        }
