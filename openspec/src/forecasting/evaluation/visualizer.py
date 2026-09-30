"""
Evaluation visualizer: forecast plots, scatter, error distributions, Q-Q.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """
    Visualization helper for model evaluation results.

    Parameters
    ----------
    figsize : tuple
        Default figure size.
    """

    def __init__(self, figsize: tuple[int, int] = (14, 7)):
        self.figsize = figsize

    def plot_forecast(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        model_name: str = "Model",
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot actual vs predicted with optional prediction intervals."""
        fig, ax = plt.subplots(figsize=self.figsize)

        x = dates if dates is not None else np.arange(len(actuals))
        ax.plot(x, actuals, label="Actual", color="steelblue", linewidth=1.5)
        ax.plot(x, predictions, label=f"{model_name} Predicted", color="coral",
                linewidth=1.5, linestyle="--")

        if lower_bound is not None and upper_bound is not None:
            ax.fill_between(x, lower_bound, upper_bound, alpha=0.2, color="coral",
                           label="95% Prediction Interval")

        ax.set_title(title or f"Forecast vs Actual — {model_name}")
        ax.set_xlabel("Date" if dates is not None else "Index")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    def plot_scatter(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Predicted vs actual scatter plot with identity line."""
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.scatter(actuals, predictions, alpha=0.5, s=20, color="steelblue")

        # Identity line
        lims = [min(actuals.min(), predictions.min()),
                max(actuals.max(), predictions.max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="Identity")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predicted vs Actual — {model_name}")
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    def plot_error_distribution(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Error distribution histogram."""
        errors = actuals - predictions

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(errors, bins=50, edgecolor="white", alpha=0.7, color="steelblue")
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[0].set_title(f"Error Distribution — {model_name}")
        axes[0].set_xlabel("Error (Actual - Predicted)")
        axes[0].set_ylabel("Frequency")

        # Cumulative error
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1].plot(sorted_errors, cdf, color="steelblue")
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_title(f"Cumulative Error — {model_name}")
        axes[1].set_xlabel("Error")
        axes[1].set_ylabel("Cumulative Probability")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    def plot_qq(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Q-Q plot for residual normality assessment."""
        from scipy import stats as sp_stats

        residuals = actuals - predictions
        sorted_resid = np.sort(residuals)
        n = len(sorted_resid)
        theoretical = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(theoretical, sorted_resid, alpha=0.5, s=20, color="steelblue")

        # Reference line
        lims = [theoretical.min(), theoretical.max()]
        slope = np.std(sorted_resid)
        intercept = np.mean(sorted_resid)
        ax.plot(lims, [intercept + slope * x for x in lims], "r--", linewidth=1)

        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title(f"Q-Q Plot — {model_name}")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    def plot_comparison_table(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Render comparison table as a figure."""
        fig, ax = plt.subplots(figsize=(max(10, len(comparison_df.columns) * 1.5),
                                        max(3, len(comparison_df) * 0.5 + 1)))
        ax.axis("off")

        table = ax.table(
            cellText=comparison_df.round(4).values,
            colLabels=comparison_df.columns,
            rowLabels=comparison_df.index,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax.set_title("Model Comparison", fontsize=14, pad=20)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig
