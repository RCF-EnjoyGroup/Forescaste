"""
EDA visualizer: plots for distributions, decomposition, stationarity,
autocorrelation, seasonality, outliers, and correlation heatmaps.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.forecasting.eda.analyzer import DecompositionResult, EDAAnalyzer

logger = logging.getLogger(__name__)


class EDAVisualizer:
    """
    Visualization helper for EDA results.

    Parameters
    ----------
    analyzer : EDAAnalyzer
        The analyzer instance containing the data.
    figsize : tuple
        Default figure size.
    style : str
        Matplotlib style name.
    """

    def __init__(
        self,
        analyzer: EDAAnalyzer,
        figsize: tuple[int, int] = (14, 7),
        style: str = "seaborn-v0_8-whitegrid",
    ):
        self.analyzer = analyzer
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            logger.warning("Style '%s' not found, using default.", style)

    # ------------------------------------------------------------------
    # Distribution plots
    # ------------------------------------------------------------------

    def plot_distributions(
        self, column: str = "revenue", save_path: Optional[str] = None
    ) -> plt.Figure:
        """Histogram, boxplot, and KDE for a single column."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        data = self.analyzer.df[column].dropna()

        axes[0].hist(data, bins=50, edgecolor="white", alpha=0.7)
        axes[0].set_title(f"{column} — Histogram")
        axes[0].set_xlabel(column)
        axes[0].set_ylabel("Frequency")

        axes[1].boxplot(data, vert=True)
        axes[1].set_title(f"{column} — Boxplot")
        axes[1].set_ylabel(column)

        data.plot.kde(ax=axes[2])
        axes[2].set_title(f"{column} — KDE")
        axes[2].set_xlabel(column)

        fig.suptitle(f"Distribution Analysis: {column}", fontsize=14, y=1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Time series plot
    # ------------------------------------------------------------------

    def plot_time_series(
        self,
        column: str = "revenue",
        hotel_id: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot time series with optional hotel filter."""
        df = self.analyzer.df.copy()
        if hotel_id and "hotel_id" in df.columns:
            df = df[df["hotel_id"] == hotel_id]

        fig, ax = plt.subplots(figsize=self.figsize)
        if "hotel_id" in df.columns:
            for hid, group in df.groupby("hotel_id"):
                ax.plot(group.index, group[column], label=hid, alpha=0.7)
            ax.legend()
        else:
            ax.plot(df.index, df[column])

        ax.set_title(f"{column} Over Time" + (f" — {hotel_id}" if hotel_id else ""))
        ax.set_xlabel("Date")
        ax.set_ylabel(column)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Decomposition plot
    # ------------------------------------------------------------------

    def plot_decomposition(
        self, result: DecompositionResult, column: str = "revenue",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot trend, seasonal, and residual components."""
        fig, axes = plt.subplots(4, 1, figsize=self.figsize, sharex=True)

        # Observed
        series = self.analyzer.df[column].dropna()
        axes[0].plot(series, color="steelblue")
        axes[0].set_title(f"Observed — {column}")
        axes[0].set_ylabel(column)

        # Trend
        axes[1].plot(result.trend, color="coral")
        axes[1].set_title("Trend")
        axes[1].set_ylabel("Trend")

        # Seasonal
        axes[2].plot(result.seasonal, color="forestgreen")
        axes[2].set_title("Seasonal")
        axes[2].set_ylabel("Seasonal")

        # Residual
        axes[3].plot(result.residual, color="gray")
        axes[3].set_title("Residual")
        axes[3].set_ylabel("Residual")
        axes[3].set_xlabel("Date")

        fig.suptitle(
            f"Seasonal Decomposition ({result.model_type}, period={result.period})",
            fontsize=14, y=1.01,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # ACF / PACF plots
    # ------------------------------------------------------------------

    def plot_acf_pacf(
        self,
        column: str = "revenue",
        nlags: int = 60,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ACF and PACF side by side."""
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        series = self.analyzer.df[column].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        plot_acf(series, lags=nlags, ax=axes[0], alpha=0.05)
        axes[0].set_title(f"ACF — {column}")

        plot_pacf(series, lags=nlags, ax=axes[1], alpha=0.05, method="ywm")
        axes[1].set_title(f"PACF — {column}")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Seasonal heatmap
    # ------------------------------------------------------------------

    def plot_seasonal_heatmap(
        self,
        column: str = "revenue",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Month × Year heatmap of average revenue."""
        df = self.analyzer.df.copy()
        df["year"] = df.index.year
        df["month"] = df.index.month

        pivot = df.pivot_table(values=column, index="month", columns="year", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
        ax.set_title(f"Monthly Average {column} — Heatmap")
        ax.set_ylabel("Month")
        ax.set_xlabel("Year")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Outlier plot
    # ------------------------------------------------------------------

    def plot_outliers(
        self,
        column: str = "revenue",
        outlier_indices: Optional[pd.Index] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Highlight outliers on time series plot."""
        series = self.analyzer.df[column].dropna()

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(series.index, series, label=column, color="steelblue", alpha=0.8)

        if outlier_indices is not None and len(outlier_indices) > 0:
            # Align via boolean mask: with duplicated index labels (e.g. several
            # hotels per date) .loc would return more rows than indices.
            mask = series.index.isin(outlier_indices)
            ax.scatter(
                series.index[mask],
                series.values[mask],
                color="red", zorder=5, s=30, label="Outliers",
            )

        ax.set_title(f"{column} with Outliers Highlighted")
        ax.set_xlabel("Date")
        ax.set_ylabel(column)
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Annotated correlation heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
        )
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig
