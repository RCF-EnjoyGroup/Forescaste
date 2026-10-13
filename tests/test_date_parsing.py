"""Tests del cambio fix-capacity-date-format (parseo flexible de fechas).

El capacity real llega en DD/MM/YYYY (estándar del negocio): el parser
default de pandas descarta ~60% de las fechas (día > 12 no puede ser
mes) y cruza día/mes en las ambiguas ("05/01/2025" = 5-ene se lee como
5-may). ``parse_dates_flexible`` elige por columna el formato que
resuelve más fechas; normalización (core), coerción de esquema y checks
de calidad lo usan. Además: filas completamente vacías = higiene
(warning), y grupos con fechas ilegibles = error sin crash.
"""

import pandas as pd
import pytest

from src.data.dates import parse_dates_flexible
from src.data.schema import SourceKind, _is_date_like, detect_schema
from src.features.events import EventsError, load_events
from src.forecasting.core import _normalize_capacity
from src.validation.quality import (
    check_empty_rows,
    check_missing_dates,
    check_rn_over_capacity,
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


# ---------------------------------------------------------------- parser


def test_parser_iso_fast_path_identico_al_default():
    s = pd.Series(["2026-01-07", "2026-01-13", "2026-12-31"])
    out = parse_dates_flexible(s)
    assert out.tolist() == [
        pd.Timestamp("2026-01-07"),
        pd.Timestamp("2026-01-13"),
        pd.Timestamp("2026-12-31"),
    ]


def test_parser_prefer_dayfirst_desempata_ambiguousos():
    """Columna 100% ambigua: el conteo empata, prefer_dayfirst rompe a DD/MM."""
    s = pd.Series(["05/10/2026", "12/11/2026"])
    out = parse_dates_flexible(s, prefer_dayfirst=True)
    assert out.iloc[0] == pd.Timestamp("2026-10-05")
    assert out.iloc[1] == pd.Timestamp("2026-11-12")
    # sin la preferencia, el default gana (compatibilidad MM/DD)
    sin_pref = parse_dates_flexible(s)
    assert sin_pref.iloc[0] == pd.Timestamp("2026-05-10")


def test_parser_prefer_dayfirst_no_toca_iso():
    s = pd.Series(["2026-10-05", "2026-11-12"])
    out = parse_dates_flexible(s, prefer_dayfirst=True)
    assert out.tolist() == [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-11-12")]


def test_parser_ddmm_resuelve_todo_sin_cruce():
    """DD/MM real: día > 12 parsea y las ambiguas van con día primero."""
    s = pd.Series(["01/01/2025", "05/01/2025", "13/01/2025", "31/12/2025"])
    out = parse_dates_flexible(s)
    assert out.notna().all(), "ninguna fecha se descarta"
    assert out.iloc[1] == pd.Timestamp("2025-01-05"), "05/01 = 5-ene, no 5-may"
    assert out.iloc[2] == pd.Timestamp("2025-01-13")
    assert out.iloc[3] == pd.Timestamp("2025-12-31")


def test_parser_ddmm_columna_real_del_capacity():
    """Patrón del capacity real: 12 días ambiguos + salto a día 13."""
    dias = [f"{d:02d}/01/2025" for d in range(1, 32)]
    out = parse_dates_flexible(pd.Series(dias))
    assert out.notna().all()
    assert out.iloc[12] == pd.Timestamp("2025-01-13")
    assert out.iloc[30] == pd.Timestamp("2025-01-31")


def test_parser_mmdd_puro_no_se_tuerce():
    """Un source MM/DD puro resuelve 100% con el default: gana por empate."""
    s = pd.Series(["01/13/2026", "05/13/2026", "12/31/2026"])
    out = parse_dates_flexible(s)
    assert out.notna().all()
    assert out.iloc[0] == pd.Timestamp("2026-01-13")
    assert out.iloc[2] == pd.Timestamp("2026-12-31")


def test_parser_datetime_passthrough():
    s = pd.Series(pd.date_range("2026-01-01", periods=3))
    assert parse_dates_flexible(s).equals(s)


def test_parser_serie_vacia_todo_nat():
    out = parse_dates_flexible(pd.Series([None, None]))
    assert out.isna().all()


def test_parser_ilegibles_quedan_nat():
    out = parse_dates_flexible(pd.Series(["no-es-fecha", "2026-01-07"]))
    assert out.iloc[0] is pd.NaT
    assert out.iloc[1] == pd.Timestamp("2026-01-07")


# ------------------------------------------------------- normalización core


def _capacity_df(fechas, capacidades=(100,)):
    n = len(fechas)
    caps = (list(capacidades) * n)[:n]
    return pd.DataFrame(
        {"property": ["corin"] * n, "business_date": fechas, "available_rooms": caps}
    )


def test_normalize_capacity_ddmm_fechas_correctas():
    raw = _capacity_df(["05/01/2025", "13/01/2025", "31/12/2025"])
    schema = detect_schema(raw, SourceKind.CAPACITY, HINTS)
    out = _normalize_capacity(raw, schema)
    assert len(out) == 3, "el día 13 ya no se descarta"
    assert out["stay_date"].tolist() == [
        pd.Timestamp("2025-01-05"),
        pd.Timestamp("2025-01-13"),
        pd.Timestamp("2025-12-31"),
    ]


def test_normalize_capacity_iso_sigue_igual():
    raw = _capacity_df(["2026-01-07", "2026-01-08"])
    schema = detect_schema(raw, SourceKind.CAPACITY, HINTS)
    out = _normalize_capacity(raw, schema)
    assert out["stay_date"].tolist() == [
        pd.Timestamp("2026-01-07"),
        pd.Timestamp("2026-01-08"),
    ]


def test_normalize_capacity_sin_perder_el_60_por_ciento():
    """Replica del bug: 31 días DD/MM → el default solo resolvía 12."""
    fechas = [f"{d:02d}/01/2025" for d in range(1, 32)]
    raw = _capacity_df(fechas)
    schema = detect_schema(raw, SourceKind.CAPACITY, HINTS)
    assert len(_normalize_capacity(raw, schema)) == 31


# ------------------------------------------------------------ esquema


def test_is_date_like_ddmm():
    """Columna DD/MM reconocida como fecha (antes: <90% parseaba → False)."""
    s = pd.Series([f"{d:02d}/01/2025" for d in range(1, 21)])
    assert _is_date_like(s)


def test_coerce_esquema_capacity_ddmm():
    raw = _capacity_df(["13/01/2025", "31/01/2025"])
    schema = detect_schema(raw, SourceKind.CAPACITY, HINTS)
    out = schema.coerce(raw)
    assert out["business_date"].tolist() == [
        pd.Timestamp("2025-01-13"),
        pd.Timestamp("2025-01-31"),
    ]


# --------------------------------------------------------- calidad: vacías


def test_check_empty_rows_filtra_y_reporta():
    """Filas ',,' del capacity: warning, no duplicados espurios, no crash."""
    raw = pd.DataFrame(
        {
            "property": ["corin", "nan", "nan"],
            "business_date": ["13/01/2025", None, None],
            "available_rooms": [54, None, None],
        }
    )
    schema = detect_schema(raw.drop(index=[1, 2]), SourceKind.CAPACITY, HINTS)
    issues, clean = check_empty_rows(raw)
    assert len(issues) == 1 and issues[0].check == "empty_rows"
    assert issues[0].severity == "warning"
    assert issues[0].records == 2
    assert len(clean) == 1 and clean["property"].iloc[0] == "corin"


def test_validate_source_filas_vacias_no_generan_errores():
    """La fuente real (con basura) valida sin crash y sin error de gate."""
    raw = pd.DataFrame(
        {
            "property": ["corin", "corin", "nan", "nan"],
            "business_date": ["01/01/2025", "02/01/2025", None, None],
            "available_rooms": [54, 54, None, None],
        }
    )
    schema = detect_schema(raw.iloc[:2], SourceKind.CAPACITY, HINTS)
    issues = validate_source(raw, schema)
    assert not [i for i in issues if i.severity == "error"], (
        "las filas vacías son higiene, no corrupción"
    )
    empt = [i for i in issues if i.check == "empty_rows"]
    assert len(empt) == 1 and empt[0].records == 2
    assert not [i for i in issues if i.check == "duplicate_property_date"]


def test_check_empty_rows_fuente_limpia_no_emite():
    raw = _capacity_df(["13/01/2025"])
    issues, clean = check_empty_rows(raw)
    assert issues == [] and len(clean) == 1


# ------------------------------------------------- calidad: ilegibles


def test_check_missing_dates_grupo_ilegible_error_sin_crash():
    """Propiedad con fechas presentes pero ilegibles: error, no excepción."""
    fechas_corin = [f"{d:02d}/01/2025" for d in range(1, 21)]  # 20 válidas
    raw = pd.DataFrame(
        {
            "property": ["corin"] * 20 + ["malo", "malo"],
            "business_date": fechas_corin + ["???", "tampoco"],
            "available_rooms": [54] * 22,
        }
    )
    schema = detect_schema(raw, SourceKind.CAPACITY, HINTS)
    issues = check_missing_dates(raw, schema)
    ilegibles = [i for i in issues if "ilegibles" in i.message]
    assert len(ilegibles) == 1
    assert ilegibles[0].severity == "error"
    assert ilegibles[0].properties == ("malo",)
    assert not [i for i in issues if i.properties == ("corin",)], (
        "corin sigue completa: sin huecos espurios"
    )


def test_check_missing_dates_capacity_ddmm_sin_huecos_falsos():
    """31 días DD/MM consecutivos: antes el 61% ausente generaba huecos."""
    fechas = [f"{d:02d}/01/2025" for d in range(1, 32)]
    raw = _capacity_df(fechas)
    schema = detect_schema(raw, SourceKind.CAPACITY, HINTS)
    issues = check_missing_dates(raw, schema)
    assert issues == [], "calendario completo: cero huecos"


# ------------------------------------------------- cruce demanda/capacidad


def test_rn_over_capacity_join_ddmm_sin_falsos_positivos():
    """El cruce usa las fechas correctas: RN 40 <= cap 54 no es violación."""
    capacity = pd.DataFrame(
        {
            "property": ["corin", "corin"],
            "business_date": ["05/01/2026", "13/01/2026"],
            "available_rooms": [54, 54],
        }
    )
    demand = pd.DataFrame(
        {
            "property": ["corin", "corin"],
            "stay_date": ["2026-01-05", "2026-01-13"],
            "room_nights": [40, 40],
        }
    )
    c_schema = detect_schema(capacity, SourceKind.CAPACITY, HINTS)
    d_schema = detect_schema(demand, SourceKind.DEMAND, HINTS)
    issues = check_rn_over_capacity(demand, d_schema, capacity, c_schema)
    assert issues == [], "el join correcto elimina los falsos positivos"


def test_rn_over_capacity_ddmm_detecta_violacion_real():
    capacity = pd.DataFrame(
        {
            "property": ["corin", "corin"],
            "business_date": ["05/01/2026", "13/01/2026"],
            "available_rooms": [54, 54],
        }
    )
    demand = pd.DataFrame(
        {
            "property": ["corin", "corin"],
            "stay_date": ["2026-01-05", "2026-01-13"],
            "room_nights": [40, 60],  # 60 > 54 el 13-ene (día>12, antes invisible)
        }
    )
    c_schema = detect_schema(capacity, SourceKind.CAPACITY, HINTS)
    d_schema = detect_schema(demand, SourceKind.DEMAND, HINTS)
    issues = check_rn_over_capacity(demand, d_schema, capacity, c_schema)
    assert len(issues) == 1
    assert issues[0].records == 1
    assert "2026-01-13" in issues[0].dates


# ------------------------------------------------- detección con DD/MM


def test_detect_schema_capacity_ddmm_encuentra_roles():
    """La inspección detecta los roles aunque las fechas vengan DD/MM."""
    raw = _capacity_df(["13/01/2025", "31/01/2025"], capacidades=(54,))
    schema = detect_schema(raw, SourceKind.CAPACITY, HINTS)
    assert schema.date_col == "business_date"
    assert schema.col("capacity") == "available_rooms"


# ------------------------------------------------------ fuente de eventos


def test_events_loader_acepta_ddmm(tmp_path):
    """Si Revenue llena events.csv en DD/MM (hábito del negocio), no se cruza."""
    p = tmp_path / "events.csv"
    p.write_text(
        "event_name,event_date,property\nFeria,05/10/2026,CORIN\n",
        encoding="utf-8",
    )
    out = load_events(p)
    assert out["event_date"].iloc[0] == pd.Timestamp("2026-10-05")


def test_events_loader_iso_sigue_valido(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text("event_name,event_date\nCongreso,2026-10-05\n", encoding="utf-8")
    out = load_events(p)
    assert out["event_date"].iloc[0] == pd.Timestamp("2026-10-05")


def test_events_loader_basura_detiene(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text("event_name,event_date\nCongreso,noviembre\n", encoding="utf-8")
    with pytest.raises(EventsError, match="ilegible"):
        load_events(p)
