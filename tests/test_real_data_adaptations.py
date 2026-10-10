"""Adaptaciones a las fuentes REALES (task 10.5).

Los CSVs reales difieren del esquema asumido en cuatro puntos que estas
pruebas bloquean para siempre:

1. pickup SIN columna snapshot (se deriva ``stay - lead``);
2. alias de propiedad entre fuentes (``Hotel Royal Corin`` ≡ ``CORIN``);
3. ``rooms_booked`` es on-the-books ACUMULADO (el target es ``actual - booked``);
4. el booked al corte se toma del snapshot más reciente ≤ corte por estancia.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig, load_config
from src.data.schema import SourceKind, detect_schema
from src.forecasting.runner import (
    _apply_aliases,
    _booked_requests,
    _normalize_pickup,
    _remaining_pickup_curves,
    run_end_to_end,
)

HINTS = load_config(env={}).column_hints


def _pickup_real_style() -> pd.DataFrame:
    """Esquema real: property, stay_date, lead_days, rooms_booked (sin snapshot)."""
    stays = pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"])
    rows = []
    for stay in stays:
        for lead, booked in [(5, 40.0), (3, 45.0), (1, 48.0)]:
            rows.append(
                {
                    "property": "FIESTA",
                    "stay_date": str(stay.date()),
                    "lead_days": lead,
                    "rooms_booked": booked,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1. esquema


def test_pickup_real_sin_snapshot_deriva_de_lead():
    """El pickup real no trae snapshot: se deriva stay - lead, no se rechaza."""
    df = _pickup_real_style()
    s = detect_schema(df, SourceKind.PICKUP, HINTS)
    assert s.roles["pickup"] == "rooms_booked"
    assert s.roles["lead"] == "lead_days"
    assert s.roles["snapshot"] == "__derived__"

    out = _normalize_pickup(df, s)
    row = out[(out["stay_date"] == "2026-03-10") & (out["lead_days"] == 5)].iloc[0]
    assert row["snapshot_date"] == pd.Timestamp("2026-03-05")
    assert row["pickup_rn"] == 40.0


# ----------------------------------------------------------------- 2. alias


def test_alias_propiedad_cruza_fuentes():
    """'Hotel Royal Corin' (demanda) y 'CORIN' (capacity) son el mismo hotel."""
    cfg = load_config(env={})
    assert cfg.property_aliases.get("Hotel Royal Corin") == "CORIN"

    dem = pd.DataFrame(
        {
            "property": ["Hotel Royal Corin", "Hotel Royal Corin"],
            "stay_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "room_nights": [50.0, 55.0],
        }
    )
    schema = detect_schema(dem, SourceKind.DEMAND, cfg.column_hints)
    out = _apply_aliases(dem, schema, cfg.property_aliases)
    assert set(out["property"]) == {"CORIN"}


# ------------------------------------------------- 3. curvas restantes (no 2x)


def test_curvas_usan_pickup_restante_no_acumulado():
    """El target es actual - booked; entrenar con el acumulado infla ~2×."""
    pick = pd.DataFrame(
        {
            "property": ["P"] * 3,
            "stay_date": pd.to_datetime(["2026-03-01"] * 3),
            "snapshot_date": pd.to_datetime(["2026-02-20", "2026-02-25", "2026-02-28"]),
            "lead_days": [9, 4, 1],
            "pickup_rn": [30.0, 45.0, 48.0],  # on-the-books acumulado
        }
    )
    demand = pd.DataFrame(
        {
            "property": ["P"],
            "stay_date": pd.to_datetime(["2026-03-01"]),
            "room_nights": [50.0],
        }
    )
    curves = _remaining_pickup_curves(pick, demand, pd.Timestamp("2026-03-01"))
    # restante = 50 - booked: 20, 5, 2 (NO el acumulado 30/45/48)
    assert sorted(curves["pickup_rn"]) == [2.0, 5.0, 20.0]


# --------------------------------------------------- 4. booked as-of el corte


def test_booked_requests_toma_snapshot_mas_reciente():
    """Para cada estancia futura: el booked del snapshot más reciente ≤ corte."""
    pick = pd.DataFrame(
        {
            "property": ["P"] * 4,
            "stay_date": pd.to_datetime(["2026-03-10"] * 4),
            "snapshot_date": pd.to_datetime(
                ["2026-03-01", "2026-03-04", "2026-03-06", "2026-03-09"]
            ),
            "lead_days": [9, 6, 4, 1],
            "pickup_rn": [10.0, 25.0, 30.0, 33.0],
        }
    )
    dates = pd.DatetimeIndex([pd.Timestamp("2026-03-10")])
    # corte 2026-03-06 → lead necesario 4 → toma el snapshot del 06 (booked 30)
    req = _booked_requests(pick, ["P"], dates, pd.Timestamp("2026-03-06"))
    assert req["current_booked_rn"].iloc[0] == 30.0
    assert req["lead_days"].iloc[0] == 4


def test_booked_requests_solo_usa_snapshots_previos_al_corte():
    """Un snapshot posterior al corte no puede usarse (as-of)."""
    pick = pd.DataFrame(
        {
            "property": ["P"] * 2,
            "stay_date": pd.to_datetime(["2026-03-10"] * 2),
            "snapshot_date": pd.to_datetime(["2026-03-07", "2026-03-09"]),
            "lead_days": [3, 1],
            "pickup_rn": [30.0, 33.0],
        }
    )
    dates = pd.DatetimeIndex([pd.Timestamp("2026-03-10")])
    req = _booked_requests(pick, ["P"], dates, pd.Timestamp("2026-03-06"))
    # ambos snapshots son > corte → cold start (NaN → el caller usa la media)
    assert req["current_booked_rn"].isna().all()


# ----------------------------------------------------- 5. runner end-to-end


def _mini_sources():
    """Fuentes mínimas estilo real: alias + pickup sin snapshot + acumulado."""
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    rng = np.random.default_rng(7)
    dem = pd.DataFrame(
        {
            "property": ["Hotel Royal Corin"] * len(dates),
            "stay_date": dates,
            "rooms_sold": np.clip(60 + rng.normal(0, 8, len(dates)), 0, None).round(),
        }
    )
    cap = pd.DataFrame(
        {
            "property": ["CORIN"] * len(dates),
            "business_date": dates,
            "available_rooms": [90] * len(dates),
        }
    )
    rows = []
    final = dem.set_index("stay_date")["rooms_sold"]
    for stay in dates[::5]:
        for lead in [30, 14, 7, 3, 1]:
            snap = stay - pd.Timedelta(days=lead)
            if snap > pd.Timestamp("2026-03-31"):
                continue
            total = float(final.get(stay, 55.0))
            booked = total * (1 - np.exp(-lead / 25))
            rows.append(
                {
                    "property": "Hotel Royal Corin",
                    "stay_date": stay,
                    "lead_days": lead,
                    "rooms_booked": max(booked, 0.0),
                }
            )
    pick = pd.DataFrame(rows)
    return dem, cap, pick


def test_runner_end_to_end_con_esquema_real(tmp_path):
    """La cadena completa pasa con alias + pickup sin snapshot + acumulado."""
    dem, cap, pick = _mini_sources()
    result = run_end_to_end(
        demand=dem,
        capacity=cap,
        pickup=pick,
        run_dir=tmp_path / "real_style",
        cutoff=pd.Timestamp("2026-03-31"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=False,
    )
    assert result.checks.all_passed, result.checks.failures()
    ft = result.forecast_table
    assert set(ft["property"]) == {"CORIN"}  # alias aplicado
    assert (ft["final_forecast_rn"] <= ft["capacity"] + 1e-9).all()
    # sin doble conteo: el forecast no puede superar ~2× el nivel histórico
    assert ft["final_forecast_rn"].max() < 2 * dem["rooms_sold"].max()
