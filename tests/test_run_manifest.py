"""Tests del run manifest (task 1.3).

Verifica que una corrida sintética escribe el manifiesto completo y que la
misma seed reproduce exactamente la misma corrida.
"""

import numpy as np
import pytest

from src.monitoring.run_manifest import (
    REQUIRED_FIELDS,
    ManifestError,
    RunManifest,
    fingerprint,
    read_manifest,
    same_configuration,
)


def _synthetic_run(seed: int, metrics: dict | None = None) -> tuple[RunManifest, dict]:
    """Corrida sintética determinista: la seed gobierna la serie generada."""
    rng = np.random.default_rng(seed)
    series = rng.normal(loc=100, scale=10, size=30)
    payload = {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "total": float(series.sum()),
    }
    manifest = RunManifest(
        dataset_version="synthetic-v1",
        model_version="prophet-0.01-10",
        training_period={"start": "2024-01-01", "end": "2025-12-31"},
        validation_period={"start": "2026-01-01", "end": "2026-03-31"},
        feature_config={"lags": [1, 7, 14, 28], "rollings": [7, 14, 28]},
        hyperparameters={"changepoint_prior": 0.01, "seasonality_prior": 10.0},
        metrics=metrics if metrics is not None else payload,
        seed=seed,
    )
    return manifest, payload


def test_manifest_escribe_campos_completos(tmp_path):
    manifest, payload = _synthetic_run(seed=42)
    path = manifest.write(tmp_path / "run1")

    stored = read_manifest(path)
    for field in REQUIRED_FIELDS:
        assert field in stored, f"falta campo requerido: {field}"
    assert stored["seed"] == 42
    assert stored["dataset_version"] == "synthetic-v1"
    assert stored["model_version"] == "prophet-0.01-10"
    assert stored["training_period"] == {"start": "2024-01-01", "end": "2025-12-31"}
    assert stored["validation_period"] == {"start": "2026-01-01", "end": "2026-03-31"}
    assert stored["feature_config"]["lags"] == [1, 7, 14, 28]
    assert stored["hyperparameters"]["changepoint_prior"] == 0.01
    assert stored["metrics"]["mean"] == pytest.approx(payload["mean"])
    assert stored["timestamp"]


def test_manifest_reproducible_misma_seed(tmp_path):
    m1, p1 = _synthetic_run(seed=42)
    m2, p2 = _synthetic_run(seed=42)

    assert p1 == p2, "misma seed debe producir la misma corrida sintética"
    assert fingerprint(p1) == fingerprint(p2)

    path1 = m1.write(tmp_path / "run1")
    path2 = m2.write(tmp_path / "run2")
    d1, d2 = read_manifest(path1), read_manifest(path2)
    assert same_configuration(d1, d2), "misma seed y config => misma configuración de corrida"


def test_manifest_seed_distinta_cambia_corrida():
    _, p1 = _synthetic_run(seed=42)
    _, p2 = _synthetic_run(seed=7)
    assert p1 != p2, "seeds distintas deben producir corridas distintas"


def test_manifest_incompleto_falla():
    broken = RunManifest(
        dataset_version=None,   # type: ignore[arg-type]
        model_version="m1",
        training_period={"start": "a", "end": "b"},
        validation_period={"start": "c", "end": "d"},
        feature_config={},
        hyperparameters={},
        metrics={"x": 1.0},
        seed=42,
    )
    with pytest.raises(ManifestError):
        broken.validate()


def test_manifest_en_disco_incompleto_falla(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text('{"seed": 42}', encoding="utf-8")
    with pytest.raises(ManifestError):
        read_manifest(path)
