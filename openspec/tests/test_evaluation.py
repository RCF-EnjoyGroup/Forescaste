"""Unit tests for evaluation module."""

import numpy as np
import pytest

from src.forecasting.evaluation.evaluator import Evaluator


@pytest.fixture
def evaluator():
    return Evaluator()


@pytest.fixture
def sample_predictions():
    rng = np.random.default_rng(42)
    actuals = rng.uniform(50000, 150000, 100)
    predictions = actuals + rng.normal(0, 5000, 100)
    return actuals, predictions


def test_compute_metrics(evaluator, sample_predictions):
    """Test core metric computation."""
    actuals, predictions = sample_predictions
    result = evaluator.compute_metrics(actuals, predictions, model_name="TestModel")

    assert result.model_name == "TestModel"
    assert result.n_obs == 100
    assert "MAE" in result.metrics
    assert "RMSE" in result.metrics
    assert "MAPE" in result.metrics
    assert "sMAPE" in result.metrics
    assert "MASE" in result.metrics
    assert "WAPE" in result.metrics

    # MAE and RMSE should be positive
    assert result.metrics["MAE"] > 0
    assert result.metrics["RMSE"] > 0


def test_compute_metrics_with_zeros(evaluator):
    """Test metric computation with zero actual values."""
    actuals = np.array([0, 1, 2, 3, 4])
    predictions = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    result = evaluator.compute_metrics(actuals, predictions)

    # Should not raise division by zero
    assert not np.isnan(result.metrics["MAE"])
    assert not np.isnan(result.metrics["MAPE"])


def test_compute_metrics_empty(evaluator):
    """Test metric computation with empty arrays."""
    actuals = np.array([])
    predictions = np.array([])
    result = evaluator.compute_metrics(actuals, predictions)
    assert result.n_obs == 0


def test_compare_models(evaluator, sample_predictions):
    """Test model comparison."""
    actuals, preds_a = sample_predictions
    preds_b = preds_a + np.random.default_rng(1).normal(0, 1000, len(actuals))

    results = [
        evaluator.compute_metrics(actuals, preds_a, "Model_A"),
        evaluator.compute_metrics(actuals, preds_b, "Model_B"),
    ]

    comparison = evaluator.compare_models(results, primary_metric="MAE")
    assert comparison.best_model in ["Model_A", "Model_B"]
    assert "MAE" in comparison.rankings


def test_diebold_mariano_test(evaluator):
    """Test Diebold-Mariano test."""
    rng = np.random.default_rng(42)
    actuals = rng.uniform(50000, 150000, 100)
    pred_a = actuals + rng.normal(0, 5000, 100)
    pred_b = actuals + rng.normal(0, 10000, 100)

    result = evaluator.diebold_mariano_test(actuals, pred_a, pred_b)

    assert "dm_statistic" in result
    assert "p_value" in result
    assert "significant" in result
    assert 0 <= result["p_value"] <= 1


def test_residual_diagnostics(evaluator):
    """Test residual diagnostics."""
    rng = np.random.default_rng(42)
    actuals = rng.normal(100, 10, 100)
    predictions = actuals + rng.normal(0, 5, 100)

    result = evaluator.residual_diagnostics(actuals, predictions)

    assert "jarque_bera" in result
    assert "ljung_box" in result
    assert "residual_mean" in result
    assert "residual_std" in result


def test_bootstrap_confidence_intervals(evaluator):
    """Test bootstrap CI computation."""
    rng = np.random.default_rng(42)
    values = rng.normal(10, 2, 100)

    result = evaluator.bootstrap_confidence_intervals(values, confidence=0.95)

    assert "mean" in result
    assert "lower" in result
    assert "upper" in result
    assert result["lower"] < result["mean"] < result["upper"]
