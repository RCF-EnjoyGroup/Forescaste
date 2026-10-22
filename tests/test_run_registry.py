"""Tests del registro de estado entre corridas (spec ``monitoring``).

El registro es el único estado persistido: habilita el drift de métricas
contra la corrida anterior (10.1), el re-entrenamiento con estado real
(10.4), la histéresis del campeón y el tracking vintage. Un registro
ausente o corrupto JAMÁS rompe la corrida.
"""

import json

import pytest

from src.monitoring.run_registry import (
    REGISTRY_VERSION,
    RunRecord,
    append_record,
    latest_record,
    load_registry,
)


def _record(**kwargs) -> RunRecord:
    base = dict(
        run_id="futuro-20261016-0726",
        run_dir="outputs/runs/futuro-20261016-0726",
        mode="futuro",
        timestamp="2026-10-16T13:28:38+00:00",
        cutoff="2026-10-06",
        training_end="2026-10-06",
        metrics={"wmape": 0.15, "mape": 27.4, "bias": -0.02, "n_obs": 516},
        metrics_source="injected_backtest",
        champion="pickup_curve",
        model_by_property={"CORIN": "pickup_curve", "FIESTA": "lightgbm"},
    )
    base.update(kwargs)
    return RunRecord(**base)


def test_registro_inexistente_devuelve_vacio(tmp_path):
    registry = load_registry(tmp_path / "run_registry.json")
    assert registry["records"] == []
    assert registry["registry_version"] == REGISTRY_VERSION
    assert latest_record(registry) is None


def test_registro_corrupto_no_rompe(tmp_path):
    path = tmp_path / "run_registry.json"
    path.write_text("{no es json", encoding="utf-8")
    registry = load_registry(path)
    assert registry["records"] == []
    assert latest_record(registry, mode="futuro") is None


def test_append_y_latest_roundtrip(tmp_path):
    path = tmp_path / "run_registry.json"
    append_record(path, _record(run_id="real-1", mode="real", champion="pickup_curve"))
    append_record(path, _record(run_id="futuro-1", mode="futuro"))

    registry = load_registry(path)
    assert len(registry["records"]) == 2
    # latest sin filtro: el último insertado
    assert latest_record(registry)["run_id"] == "futuro-1"
    # latest por modo: respeta el filtro
    assert latest_record(registry, mode="real")["run_id"] == "real-1"
    assert latest_record(registry, mode="futuro")["run_id"] == "futuro-1"
    assert latest_record(registry, mode="inexistente") is None


def test_append_crea_directorios_padre(tmp_path):
    path = tmp_path / "nested" / "dir" / "run_registry.json"
    append_record(path, _record())
    assert path.exists()


def test_latest_preserva_campos_del_record(tmp_path):
    path = tmp_path / "run_registry.json"
    append_record(path, _record())
    latest = latest_record(load_registry(path), mode="futuro")
    assert latest["champion"] == "pickup_curve"
    assert latest["model_by_property"]["FIESTA"] == "lightgbm"
    assert latest["metrics"]["wmape"] == pytest.approx(0.15)
    assert latest["training_end"] == "2026-10-06"


def test_record_tolerante_a_campos_desconocidos():
    """El registro evoluciona: campos nuevos no rompen la lectura de viejos."""
    payload = _record().to_dict()
    payload["campo_del_futuro"] = "x"
    record = RunRecord.from_dict(payload)
    assert record.champion == "pickup_curve"


def test_serializacion_json_valida(tmp_path):
    """El archivo en disco es JSON estricto (NaN/inf jamás)."""
    path = tmp_path / "run_registry.json"
    rec = _record(
        metrics={"wmape": float("nan"), "mape": 27.4},
        interval_coverage={"global": {"coverage": 0.81, "nominal": 0.8}},
    )
    append_record(path, rec)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)  # json.load con NaN fallaría en estricto
    assert payload["records"][0]["metrics"]["wmape"] is None
