"""Tests de validación de calidad (task 2.2).

Un test por caso del spec data-ingestion, verificando detección y
clasificación error/advertencia.
"""

import pandas as pd

from src.data.schema import SourceKind, detect_schema
from src.validation.quality import (
    check_capacity_change,
    check_duplicates,
    check_invalid_lead,
    check_missing_dates,
    check_negative_capacity,
    check_negative_demand,
    check_pickup_inconsistency,
    check_rn_over_capacity,
    check_zero_capacity,
    validate_source,
)

HINTS = {
    "property": ["property", "hotel_id", "propiedad"],
    "date": ["date", "stay_date", "fecha"],
    "capacity": ["capacity", "inventory"],
    "demand": ["room_nights", "rooms_sold", "rn"],
    "pickup": ["pickup_rn", "current_rn"],
    "snapshot": ["snapshot_date", "as_of"],
    "lead": ["lead_days", "lead"],
}


def _cap(df):
    return detect_schema(df, SourceKind.CAPACITY, HINTS)


def _dem(df):
    return detect_schema(df, SourceKind.DEMAND, HINTS)


def _pick(df):
    return detect_schema(df, SourceKind.PICKUP, HINTS)


def _capacity(rows):
    return pd.DataFrame(rows, columns=["property", "date", "capacity"])


def _demand(rows):
    return pd.DataFrame(rows, columns=["property", "date", "room_nights"])


def test_duplicados_error():
    df = _capacity(
        [
            ("corin", "2026-01-01", 100),
            ("corin", "2026-01-01", 100),   # duplicado
            ("corin", "2026-01-02", 100),
        ]
    )
    issues = check_duplicates(df, _cap(df))
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].check == "duplicate_property_date"
    assert issues[0].records == 2
    assert issues[0].properties == ("corin",)
    assert issues[0].dates == ("2026-01-01",)


def test_negativos_rn_error():
    df = _demand([("corin", "2026-01-01", -5), ("corin", "2026-01-02", 10)])
    issues = check_negative_demand(df, _dem(df))
    assert len(issues) == 1 and issues[0].severity == "error"
    assert issues[0].records == 1


def test_negativos_capacidad_error():
    df = _capacity([("corin", "2026-01-01", -100)])
    issues = check_negative_capacity(df, _cap(df))
    assert len(issues) == 1 and issues[0].severity == "error"


def test_capacidad_cero_advertencia():
    df = _capacity([("corin", "2026-01-01", 0), ("corin", "2026-01-02", 100)])
    issues = check_zero_capacity(df, _cap(df))
    assert len(issues) == 1 and issues[0].severity == "warning"


def test_fechas_faltantes_advertencia():
    df = _capacity(
        [
            ("corin", "2026-01-01", 100),
            ("corin", "2026-01-05", 100),   # 3 días de hueco
        ]
    )
    issues = check_missing_dates(df, _cap(df))
    assert len(issues) == 1 and issues[0].severity == "warning"
    assert issues[0].records == 3
    assert issues[0].dates == ("2026-01-02", "2026-01-03", "2026-01-04")


def test_cambio_capacidad_advertencia():
    df = _capacity(
        [
            ("corin", "2026-01-01", 100),
            ("corin", "2026-01-02", 40),    # -60%
        ]
    )
    issues = check_capacity_change(df, _cap(df))
    assert len(issues) == 1 and issues[0].severity == "warning"
    assert issues[0].check == "unexpected_capacity_change"


def test_lead_negativo_error():
    df = pd.DataFrame(
        {
            "property": ["corin", "corin"],
            "stay_date": pd.to_datetime(["2026-03-01", "2026-03-01"]),
            "snapshot_date": pd.to_datetime(["2026-04-01", "2026-02-01"]),  # primero negativo
            "lead_days": [-31, 28],
            "pickup_rn": [10, 20],
        }
    )
    issues = check_invalid_lead(df, _pick(df))
    assert any(i.check == "negative_lead" and i.severity == "error" for i in issues)


def test_lead_no_numerico_error():
    df = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": pd.to_datetime(["2026-03-01"]),
            "snapshot_date": pd.to_datetime(["2026-01-01"]),
            "lead_days": ["treinta"],
            "pickup_rn": [10],
        }
    )
    issues = check_invalid_lead(df, _pick(df))
    assert any(i.check == "invalid_lead" and i.severity == "error" for i in issues)


def test_snapshots_duplicados_error():
    from src.validation.quality import check_duplicate_snapshots

    df = pd.DataFrame(
        {
            "property": ["corin"] * 3,
            "stay_date": pd.to_datetime(["2026-03-01"] * 3),
            "snapshot_date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-02-01"]),
            "lead_days": [59, 59, 28],
            "pickup_rn": [10, 12, 30],
        }
    )
    issues = check_duplicate_snapshots(df, _pick(df))
    assert len(issues) == 1 and issues[0].severity == "error"
    assert issues[0].records == 2


def test_pickup_inconsistente_error():
    df = pd.DataFrame(
        {
            "property": ["corin", "corin"],
            "stay_date": pd.to_datetime(["2026-03-01", "2026-03-01"]),
            "snapshot_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "lead_days": [10, 28],          # el primero debía ser 59
            "pickup_rn": [10, 30],
        }
    )
    issues = check_pickup_inconsistency(df, _pick(df))
    assert len(issues) == 1 and issues[0].severity == "error"
    assert issues[0].records == 1


def test_rn_sobre_capacidad_baja_frecuencia_advertencia():
    """1 registro sobre capacidad entre 200 => 0.5% < umbral 1% => warning."""
    dates = [str(d.date()) for d in pd.date_range("2026-01-01", periods=200)]
    ddf = _demand([("corin", dates[0], 110)] + [("corin", d, 50) for d in dates[1:]])
    cdf = _capacity([("corin", d, 100) for d in dates])
    issues = check_rn_over_capacity(ddf, _dem(ddf), cdf, _cap(cdf), 0.01)
    assert len(issues) == 1
    assert issues[0].check == "rn_over_capacity"
    assert issues[0].severity == "warning", "1/200 = 0.5% < 1% => advertencia"
    assert issues[0].records == 1


def test_rn_sobre_capacidad_alta_frecuencia_error():
    dates = pd.date_range("2026-01-01", periods=10)
    ddf = _demand([("corin", str(d.date()), 200) for d in dates])
    cdf = _capacity([("corin", str(d.date()), 100) for d in dates])
    issues = check_rn_over_capacity(ddf, _dem(ddf), cdf, _cap(cdf), 0.01)
    assert len(issues) == 1 and issues[0].severity == "error"
    assert issues[0].records == 10
    assert issues[0].properties == ("corin",)


def test_fuente_valida_sin_issues():
    df = _capacity([("corin", "2026-01-01", 100), ("corin", "2026-01-02", 100)])
    assert validate_source(df, _cap(df)) == []


def test_fuente_vacia_error():
    df = _capacity([])
    issues = validate_source(df, _cap(df))
    assert len(issues) == 1
    assert issues[0].check == "empty_source" and issues[0].severity == "error"


def test_todos_los_checks_detectan_en_dataset_sucio():
    """Un dataset con varios problemas => cada caso detectado y clasificado."""
    df = _capacity(
        [
            ("corin", "2026-01-01", 100),
            ("corin", "2026-01-01", 100),    # duplicado -> error
            ("corin", "2026-01-02", -5),     # negativo -> error
            ("corin", "2026-01-06", 0),      # cero -> warning (+ hueco 03-05)
        ]
    )
    issues = validate_source(df, _cap(df))
    checks = {i.check for i in issues}
    assert "duplicate_property_date" in checks
    assert "negative_capacity" in checks
    assert "zero_capacity" in checks
    assert "missing_dates" in checks
    assert all(i.severity in ("error", "warning") for i in issues)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    assert errors and warnings, "el dataset debe producir ambos tipos de hallazgo"
