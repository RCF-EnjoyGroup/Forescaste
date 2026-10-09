"""Tests de detección automática de esquema (task 2.1).

Verifica que se reconocen esquemas alternativos sin asumir nombres de columna
y que una estructura desconocida lanza un error claro.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.schema import SchemaDetectionError, SourceKind, detect_schema

HINTS = {
    "property": ["property", "property_id", "hotel", "hotel_id", "propiedad"],
    "date": ["date", "stay_date", "business_date", "fecha"],
    "capacity": ["capacity", "available_rooms", "inventory", "oficial_inventory"],
    "demand": ["room_nights", "rooms_sold", "rn", "demand"],
    "pickup": ["pickup", "pickup_rn", "booked_rn", "current_rn"],
    "snapshot": ["snapshot_date", "snap_date", "forecast_date", "as_of"],
    "lead": ["lead_days", "lead", "dias_anticipacion"],
}


def _capacity_v1() -> pd.DataFrame:
    """Esquema canónico: property / date / capacity."""
    dates = pd.date_range("2026-01-01", periods=10)
    return pd.DataFrame(
        {
            "property": ["corin"] * 10,
            "date": dates,
            "capacity": [100] * 10,
            "room_type": ["STD"] * 10,  # columna extra no mapeada
        }
    )


def _capacity_v2() -> pd.DataFrame:
    """Esquema alternativo: hotel_code / business_date / oficial_inventory."""
    dates = pd.date_range("2026-01-01", periods=10)
    return pd.DataFrame(
        {
            "hotel_code": ["marina"] * 10,
            "business_date": dates,
            "oficial_inventory": [80] * 10,
            "notes": ["ok"] * 10,
        }
    )


def _demand_v1() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=10)
    return pd.DataFrame({"property": ["fiesta"] * 10, "stay_date": dates, "rooms_sold": np.arange(10)})


def _demand_v2() -> pd.DataFrame:
    """Fechas como strings ISO, propiedad y métrica con otros nombres."""
    return pd.DataFrame(
        {
            "propiedad": ["lirel"] * 5,
            "fecha": ["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"],
            "rn": [10, 12, 9, 15, 11],
        }
    )


def _pickup_v1() -> pd.DataFrame:
    stay = pd.to_datetime(["2026-03-01"] * 5)
    snap = pd.to_datetime(["2025-12-01", "2025-12-31", "2026-01-30", "2026-02-14", "2026-02-28"])
    return pd.DataFrame(
        {
            "property": ["corin"] * 5,
            "stay_date": stay,
            "snapshot_date": snap,
            "lead_days": (stay - snap).days,
            "pickup_rn": [20, 35, 48, 55, 60],
        }
    )


def _pickup_v2() -> pd.DataFrame:
    """Snapshot con otro nombre y lead derivable."""
    return pd.DataFrame(
        {
            "hotel_id": ["sjo"] * 3,
            "date": ["2026-04-01", "2026-04-01", "2026-04-01"],
            "as_of": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "current_rn": [15, 25, 40],
        }
    )


def test_capacity_esquema_canonico():
    s = detect_schema(_capacity_v1(), SourceKind.CAPACITY, HINTS)
    assert s.roles["property"] == "property"
    assert s.roles["date"] == "date"
    assert s.roles["capacity"] == "capacity"
    assert "room_type" in s.unmapped_columns, "columna extra no debe romper la carga"


def test_capacity_esquema_alternativo():
    s = detect_schema(_capacity_v2(), SourceKind.CAPACITY, HINTS)
    assert s.roles == {"property": "hotel_code", "date": "business_date", "capacity": "oficial_inventory"}
    assert "notes" in s.unmapped_columns


def test_demand_esquema_alternativo_strings():
    s = detect_schema(_demand_v2(), SourceKind.DEMAND, HINTS)
    assert s.roles["property"] == "propiedad"
    assert s.roles["date"] == "fecha"
    assert s.roles["demand"] == "rn"


def test_demand_esquema_canonico():
    s = detect_schema(_demand_v1(), SourceKind.DEMAND, HINTS)
    assert s.roles["date"] == "stay_date"
    assert s.roles["demand"] == "rooms_sold"


def test_pickup_esquema_canonico():
    s = detect_schema(_pickup_v1(), SourceKind.PICKUP, HINTS)
    assert s.roles["snapshot"] == "snapshot_date"
    assert s.roles["lead"] == "lead_days"
    assert s.roles["pickup"] == "pickup_rn"


def test_pickup_lead_derivable_desde_fechas():
    s = detect_schema(_pickup_v2(), SourceKind.PICKUP, HINTS)
    assert s.roles["snapshot"] == "as_of"
    assert s.roles["lead"] == "__derived__"

    df = s.coerce(_pickup_v2())
    assert "lead_days" in df.columns
    # 2026-01-01 -> 2026-04-01 = 90 días; 2026-02-01 -> 2026-04-01 = 59 días
    assert list(df["lead_days"]) == [90, 59, 31]


def test_estructura_desconocida_falla():
    bad = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
    with pytest.raises(SchemaDetectionError) as exc:
        detect_schema(bad, SourceKind.DEMAND, HINTS)
    msg = str(exc.value)
    assert "demand" in msg and "mínimas" in msg
    assert "col_a" in msg, "el error debe listar las columnas disponibles"


def test_fuente_vacia_falla():
    with pytest.raises(SchemaDetectionError):
        detect_schema(pd.DataFrame(), SourceKind.CAPACITY, HINTS)


def test_coerce_normaliza_dtypes():
    s = detect_schema(_demand_v2(), SourceKind.DEMAND, HINTS)
    df = s.coerce(_demand_v2())
    assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
    assert pd.api.types.is_numeric_dtype(df["rn"])
    assert df["propiedad"].tolist() == ["lirel"] * 5
