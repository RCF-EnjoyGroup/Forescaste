"""Tests del cambio booked-floor (piso de reservas en libros).

El piso de reservas es una restricción operacional de la capa de
constraints: ``final = min(max(raw, current_booked_rn), capacity)`` cuando
existe on-the-books; sin booked → ``final = min(raw, capacity)``. La
capacidad siempre manda sobre el piso (booked > capacidad → alerta de
inconsistencia de datos, no forecast imposible).
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.constraints import (
    BookedFloorAlert,
    apply_capacity_constraint,
)


def _constrained_frame(raw, booked, capacity=100.0):
    n = len(raw)
    return pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": pd.date_range("2026-07-01", periods=n),
            "raw_forecast_rn": raw,
            "current_booked_rn": booked,
            "capacity": [capacity] * n,
        }
    )


def test_piso_eleva_forecast_bajo_booked():
    df = _constrained_frame(raw=[80.0], booked=[95.0])
    out, alerts = apply_capacity_constraint(df)
    assert out["final_forecast_rn"].iloc[0] == pytest.approx(95.0)
    assert out["raw_forecast_rn"].iloc[0] == pytest.approx(80.0), "raw intacto"
    assert out["booked_floor_applied"].iloc[0]
    floor = [a for a in alerts if isinstance(a, BookedFloorAlert)]
    assert len(floor) == 1 and floor[0].shortfall == pytest.approx(15.0)


def test_piso_no_aplica_cuando_raw_mayor():
    df = _constrained_frame(raw=[98.0], booked=[95.0])
    out, alerts = apply_capacity_constraint(df)
    assert out["final_forecast_rn"].iloc[0] == pytest.approx(98.0)
    assert not out["booked_floor_applied"].iloc[0]
    assert not [a for a in alerts if isinstance(a, BookedFloorAlert)]


def test_sin_booked_no_hay_piso():
    df = _constrained_frame(raw=[80.0], booked=[np.nan])
    out, _ = apply_capacity_constraint(df)
    assert out["final_forecast_rn"].iloc[0] == pytest.approx(80.0)
    assert not out["booked_floor_applied"].iloc[0]


def test_booked_sobre_capacidad_manda_el_techo():
    """booked 120 > capacity 100: el piso no puede romper el inventario físico."""
    df = _constrained_frame(raw=[80.0], booked=[120.0], capacity=100.0)
    out, alerts = apply_capacity_constraint(df)
    assert out["final_forecast_rn"].iloc[0] == pytest.approx(100.0)
    over = [a for a in alerts if isinstance(a, BookedFloorAlert) and np.isnan(a.raw_forecast)]
    assert len(over) == 1, "debe alertar booked_over_capacity"


def test_piso_y_techo_encadenados():
    """raw 80 < booked 95 < cap 100 → final = 95 (piso aplica, techo no)."""
    df = _constrained_frame(raw=[80.0, 105.0], booked=[95.0, 92.0], capacity=100.0)
    out, alerts = apply_capacity_constraint(df)
    assert out["final_forecast_rn"].iloc[0] == pytest.approx(95.0)  # piso
    assert out["final_forecast_rn"].iloc[1] == pytest.approx(100.0)  # techo
    assert out["booked_floor_applied"].tolist() == [True, False]


def test_validate_forecast_table_acepta_piso():
    from src.reporting.forecast_table import validate_forecast_table

    df = _constrained_frame(raw=[80.0], booked=[95.0])
    out, _ = apply_capacity_constraint(df)
    out["forecast_date"] = pd.Timestamp("2026-06-30")
    out["lead_days"] = 1
    out["model_name"] = "pickup_curve"
    out["model_version"] = "0.1.0"
    out["forecast_run_timestamp"] = "2026-06-30T00:00:00+00:00"
    out["p10_forecast"] = out["final_forecast_rn"] - 5
    out["p50_forecast"] = out["final_forecast_rn"]
    out["p90_forecast"] = out["final_forecast_rn"] + 5
    problems = validate_forecast_table(out)
    assert problems == [], f"la validacion debe aceptar min(max(raw,booked),cap): {problems}"


def test_validate_forecast_table_con_nan_booked():
    """NaN en booked → sin piso → esperado = min(raw, cap) (no NaN)."""
    from src.reporting.forecast_table import validate_forecast_table

    df = _constrained_frame(raw=[80.0, 90.0], booked=[95.0, np.nan])
    out, _ = apply_capacity_constraint(df)
    out["forecast_date"] = pd.Timestamp("2026-06-30")
    out["lead_days"] = [1, 2]
    out["model_name"] = ["pickup_curve"] * 2
    out["model_version"] = ["0.1.0"] * 2
    out["forecast_run_timestamp"] = ["2026-06-30T00:00:00+00:00"] * 2
    out["p10_forecast"] = out["final_forecast_rn"] - 5
    out["p50_forecast"] = out["final_forecast_rn"]
    out["p90_forecast"] = out["final_forecast_rn"] + 5
    problems = validate_forecast_table(out)
    assert problems == [], f"NaN en booked no debe romper la validacion: {problems}"


def test_e2e_forecast_jamas_bajo_booked(tmp_path):
    """Integración: la tabla final de la corrida nunca queda bajo el booked."""
    from tests.test_remediation import (
        _folds_synthetic, _pickup_2props, _demand_2props, _capacity_2props,
    )
    from src.forecasting.runner import run_end_to_end
    from src.monitoring.run_manifest import read_manifest

    dem = _demand_2props()
    cap = _capacity_2props()
    pick = _pickup_2props(dem)
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "floor_e2e",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=_folds_synthetic("pickup_curve"),
    )
    assert result.checks.all_passed, (
        f"checks fallidos: {result.checks.failures()} details={result.checks.details}"
    )
    ft = result.forecast_table
    if "current_booked_rn" in ft.columns:
        booked = pd.to_numeric(ft["current_booked_rn"], errors="coerce")
        bajo = booked.notna() & (ft["final_forecast_rn"] < booked - 1e-9)
        assert not bajo.any(), (
            f"{int(bajo.sum())} registros con forecast < booked — el piso debe prevenirlo"
        )
    stored = read_manifest(result.manifest_path)
    assert "n_booked_floor_applied" in stored
    assert "n_booked_over_capacity" in stored
