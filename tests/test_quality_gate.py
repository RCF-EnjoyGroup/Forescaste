"""Tests de detención del pipeline ante problemas críticos (task 2.4).

Un dataset críticamente inválido debe impedir entrenar y NO escribir forecast.
"""

import pandas as pd
import pytest

from src.config import EngineConfig
from src.data.pipeline import forecast_output_path, run_ingestion
from src.data.schema import SourceKind, detect_schema
from src.validation.gate import CriticalDataError, enforce
from src.validation.quality import validate_sources
from src.validation.report import build_report

HINTS = {
    "property": ["property"],
    "date": ["date"],
    "capacity": ["capacity"],
    "demand": ["room_nights"],
    "pickup": ["pickup_rn"],
    "snapshot": ["snapshot_date"],
    "lead": ["lead_days"],
}


def _cfg(tmp_path) -> EngineConfig:
    return EngineConfig(
        random_seed=42,
        forecast_horizon=90,
        percentiles=(10, 50, 90),
        retrain_frequency="monthly",
        min_history_days=365,
        sources={},
        column_hints=HINTS,
    )


def _frames(capacity: pd.DataFrame, demand: pd.DataFrame, pickup: pd.DataFrame):
    return {
        "capacity": (capacity, detect_schema(capacity, SourceKind.CAPACITY, HINTS)),
        "demand": (demand, detect_schema(demand, SourceKind.DEMAND, HINTS)),
        "pickup": (pickup, detect_schema(pickup, SourceKind.PICKUP, HINTS)),
    }


def _ok_frames():
    dates = [str(d.date()) for d in pd.date_range("2026-01-01", periods=5)]
    capacity = pd.DataFrame(
        [("corin", d, 100) for d in dates], columns=["property", "date", "capacity"]
    )
    demand = pd.DataFrame(
        [("corin", d, 50) for d in dates], columns=["property", "date", "room_nights"]
    )
    stay = pd.to_datetime(["2026-03-01"] * 5)
    snap = pd.to_datetime(
        ["2025-12-01", "2025-12-31", "2026-01-30", "2026-02-14", "2026-02-28"]
    )
    pickup = pd.DataFrame(
        {
            "property": ["corin"] * 5,
            "stay_date": stay,
            "snapshot_date": snap,
            "lead_days": (stay - snap).days,
            "pickup_rn": [20, 35, 48, 55, 60],
        }
    )
    return _frames(capacity, demand, pickup)


def _critical_frames():
    """Duplicados + RN negativos + lead negativo => errores críticos."""
    frames = _ok_frames()
    cap_df = frames["capacity"][0]
    cap_df = pd.concat([cap_df, cap_df.iloc[[0]]], ignore_index=True)  # duplicado
    dem_df = frames["demand"][0].copy()
    dem_df.loc[0, "room_nights"] = -20  # RN negativo
    pick_df = frames["pickup"][0].copy()
    pick_df.loc[0, "lead_days"] = -10  # lead negativo
    return _frames(cap_df, dem_df, pick_df)


def _fake_training_and_forecast(sources, run_dir):
    """Simula el paso de entrenamiento+forecast: solo corre si el gate pasa."""
    report = run_ingestion(config=_cfg(run_dir), sources=sources)  # lanza si crítico
    # Si llegamos aquí, entrenar y escribir forecast:
    pd.DataFrame({"property": ["corin"], "stay_date": ["2026-03-01"], "forecast_rn": [60]}).to_csv(
        forecast_output_path(run_dir), index=False
    )
    return report


def test_dataset_critico_impide_entrenar_y_no_escribe_forecast(tmp_path):
    with pytest.raises(CriticalDataError) as exc:
        _fake_training_and_forecast(_critical_frames(), tmp_path)

    assert not forecast_output_path(tmp_path).exists(), "no debe escribirse forecast"
    msg = str(exc.value)
    assert "entrenamiento detenido" in msg
    assert "duplicate_property_date" in msg or "negative_rn" in msg


def test_dataset_critico_mensaje_identifica_problema(tmp_path):
    with pytest.raises(CriticalDataError) as exc:
        run_ingestion(config=_cfg(tmp_path), sources=_critical_frames())
    report = exc.value.report
    assert report.status == "fail"
    assert report.error_count >= 3
    assert "corin" in report.affected_properties
    for issue in report.errors:
        assert issue.check and issue.message


def test_dataset_solo_advertencias_continua(tmp_path):
    frames = _ok_frames()
    cap = frames["capacity"][0].copy()
    cap.loc[0, "capacity"] = 0  # warning, no error
    dem = frames["demand"][0].copy()
    dem.loc[0, "room_nights"] = 0  # ese día sin demanda: no hay rn>cap
    frames["capacity"] = (cap, detect_schema(cap, SourceKind.CAPACITY, HINTS))
    frames["demand"] = (dem, detect_schema(dem, SourceKind.DEMAND, HINTS))

    report = _fake_training_and_forecast(frames, tmp_path)
    assert report.status == "warnings"
    assert forecast_output_path(tmp_path).exists(), "advertencias no detienen el pipeline"


def test_dataset_limpio_pasa_el_gate(tmp_path):
    report = _fake_training_and_forecast(_ok_frames(), tmp_path)
    assert report.status == "pass"
    assert forecast_output_path(tmp_path).exists()


def test_enforce_no_lanza_sin_errores(tmp_path):
    frames = _ok_frames()
    report = build_report(frames, validate_sources(frames))
    enforce(report)  # no debe lanzar
