"""
Hotel Revenue Forecasting Pipeline
==================================

A comprehensive time series forecasting pipeline for predicting hotel revenue
and room revenue from financial data.

Modules:
    - data: Data loading and validation
    - eda: Exploratory data analysis
    - preprocessing: Feature engineering and data preparation
    - models: Statistical, ML, and deep learning forecasting models
    - evaluation: Model evaluation and comparison
    - forecasting: Future forecast generation
"""

__version__ = "0.1.0"
__author__ = "Enjoy Costa Rica - Revenue Analytics"

__all__ = [
    "DataLoader",
    "EDAAnalyzer",
    "PreprocessingPipeline",
    "Evaluator",
    "FutureForecaster",
]


def __getattr__(name: str):
    """Lazily resolve optional submodules to avoid import-time dependency issues."""
    if name == "DataLoader":
        from src.forecasting.data import DataLoader
        return DataLoader
    if name == "EDAAnalyzer":
        from src.forecasting.eda import EDAAnalyzer
        return EDAAnalyzer
    if name == "PreprocessingPipeline":
        from src.forecasting.preprocessing import PreprocessingPipeline
        return PreprocessingPipeline
    if name == "Evaluator":
        from src.forecasting.evaluation import Evaluator
        return Evaluator
    if name == "FutureForecaster":
        from src.forecasting.forecasting import FutureForecaster
        return FutureForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
