"""Tests del Data Quality Report (task 2.3).

Un dataset con problemas debe producir el reporte con los conteos esperados.
"""

import json

import pandas as pd

from src.data.schema import SourceKind, detect_schema
from src.validation.quality import validate_sources
from src.validation.report import build_report

HINTS = {
    "property": ["property"],
    "date": ["date"],
    "capacity": ["capacity"],
    "demand": ["room_nights", "rn"],
    "pickup": ["pickup_rn"],
    "snapshot": ["snapshot_date"],
    "lead": ["lead_days"],
}


def _sources(capacity: pd.DataFrame, demand: pd.DataFrame):
    return {
        "capacity": (capacity, detect_schema(capacity, SourceKind.CAPACITY, HINTS)),
        "demand": (demand, detect_schema(demand, SourceKind.DEMAND, HINTS)),
    }


def _cap(n_days=5, value=100, dup_last=False):
    dates = [str(d.date()) for d in pd.date_range("2026-01-01", periods=n_days)]
    rows = [("corin", d, value) for d in dates]
    if dup_last:
        rows.append(rows[-1])
    return pd.DataFrame(rows, columns=["property", "date", "capacity"])


def _dem(n_days=5, value=50, negative_first=False):
    dates = [str(d.date()) for d in pd.date_range("2026-01-01", periods=n_days)]
    rows = [("corin", d, value) for d in dates]
    if negative_first:
        rows[0] = ("corin", dates[0], -10)
    return pd.DataFrame(rows, columns=["property", "date", "room_nights"])


def test_dataset_con_problemas_conteos_esperados():
    sources = _sources(_cap(dup_last=True), _dem(negative_first=True))
    report = build_report(sources, validate_sources(sources))

    # registros: 6 capacity (5 + 1 dup) y 5 demand
    assert report.record_counts == {"capacity": 6, "demand": 5}
    assert report.total_records == 11

    # 1 error por duplicado + 1 error por RN negativo
    assert report.error_count == 2
    assert report.status == "fail"
    assert report.is_critical

    checks = report.checks_summary()
    assert checks["duplicate_property_date"]["error"] == 1
    assert checks["negative_rn"]["error"] == 1

    assert report.affected_properties == ["corin"]
    assert "2026-01-01" in report.affected_dates  # fecha del RN negativo
    assert "2026-01-05" in report.affected_dates  # fecha del duplicado


def test_dataset_solo_advertencias_status_warnings():
    # capacidad 0 con demanda 0 ese día => warning, sin rn_over_capacity
    cap = pd.DataFrame(
        [("corin", "2026-01-01", 0), ("corin", "2026-01-02", 100)],
        columns=["property", "date", "capacity"],
    )
    dem = pd.DataFrame(
        [("corin", "2026-01-01", 0), ("corin", "2026-01-02", 50)],
        columns=["property", "date", "room_nights"],
    )
    sources = _sources(cap, dem)
    report = build_report(sources, validate_sources(sources))

    assert report.error_count == 0
    assert report.warning_count >= 1
    assert report.status == "warnings"
    assert not report.is_critical


def test_dataset_limpio_status_pass():
    sources = _sources(_cap(), _dem())
    report = build_report(sources, validate_sources(sources))
    assert report.status == "pass"
    assert report.error_count == report.warning_count == 0
    assert report.affected_properties == []
    assert report.affected_dates == []


def test_reporte_json_estructura():
    sources = _sources(_cap(dup_last=True), _dem(negative_first=True))
    report = build_report(sources, validate_sources(sources))

    payload = json.loads(report.to_json())
    assert payload["status"] == "fail"
    assert payload["errors"] == 2
    assert payload["warnings"] == payload["warnings"]  # clave presente
    assert set(payload["records"].keys()) == {"total", "by_source"}
    assert payload["affected_properties"] == ["corin"]
    assert isinstance(payload["issues"], list) and payload["issues"]
    for issue in payload["issues"]:
        assert {"check", "severity", "message", "records", "properties", "dates"} <= set(issue)


def test_reporte_write(tmp_path):
    sources = _sources(_cap(), _dem())
    report = build_report(sources, validate_sources(sources))
    path = report.write(tmp_path / "dq" / "report.json")
    assert path.exists()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["status"] == "pass"


def test_summary_message_identifica_problema():
    sources = _sources(_cap(dup_last=True), _dem(negative_first=True))
    report = build_report(sources, validate_sources(sources))
    msg = report.summary_message()
    assert "duplicate_property_date" in msg
    assert "negative_rn" in msg
    assert "status=fail" in msg
