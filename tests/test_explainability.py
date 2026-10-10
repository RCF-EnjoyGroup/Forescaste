"""Tests de explicabilidad (tasks 8.1–8.3)."""

import numpy as np
import pandas as pd
import pytest

from src.explainability.explainer import (
    PredictionExplanation,
    classify_driver,
    explain_from_importance,
    global_importance,
)


def _importance():
    return {
        "current_booked_rn": 0.55,
        "pickup_hist_mean_same_lead": 0.25,
        "day_of_week": 0.12,
        "capacity": 0.08,
    }


def _values():
    return {
        "current_booked_rn": 54.0,
        "pickup_hist_mean_same_lead": 17.0,
        "day_of_week": 5.0,
        "capacity": 100.0,
    }


# --------------------------- 8.1 ----------------------------------------


def test_explicacion_por_prediccion_poblada():
    expl = explain_from_importance(
        property="corin",
        stay_date="2026-03-07",
        forecast_rn=82.0,
        feature_values=_values(),
        feature_importance=_importance(),
        method="lightgbm_importance",
    )
    assert expl.property == "corin"
    assert expl.stay_date == "2026-03-07"
    assert expl.forecast_rn == 82.0
    assert len(expl.drivers) == 4
    assert expl.method == "lightgbm_importance"
    # cada driver con sus campos
    for d in expl.drivers:
        assert {"feature", "importance", "value", "contribution", "driver"} <= set(d)

    expl.validate()  # las contribuciones suman el forecast


def test_top_drivers_ordenados_por_abs():
    expl = explain_from_importance(
        property="corin",
        stay_date="2026-03-07",
        forecast_rn=82.0,
        feature_values=_values(),
        feature_importance=_importance(),
    )
    top = expl.top_drivers(2)
    assert len(top) == 2
    order = [d["contribution"] for d in top]
    assert abs(order[0]) >= abs(order[1])


def test_importancia_global_agregada():
    e1 = explain_from_importance(
        property="corin", stay_date="2026-03-07", forecast_rn=82.0,
        feature_values=_values(), feature_importance=_importance(),
    )
    e2 = explain_from_importance(
        property="marina", stay_date="2026-03-08", forecast_rn=60.0,
        feature_values=_values(), feature_importance=_importance(),
    )
    glob = global_importance([e1, e2])
    assert not glob.empty
    assert {"feature", "driver", "mean_abs_contribution", "n"} <= set(glob.columns)
    assert (glob["n"] == 2).all()
    # ordenado por importancia media
    assert glob["mean_abs_contribution"].is_monotonic_decreasing
    # la feature más importante es la de mayor peso
    assert glob.iloc[0]["feature"] == "current_booked_rn"


def test_importancia_global_vacia():
    assert global_importance([]).empty


def test_clasificacion_drivers():
    assert classify_driver("current_booked_rn") == "current_bookings"
    assert classify_driver("pickup_hist_mean") == "pickup"
    assert classify_driver("lead_days") == "pickup"
    assert classify_driver("day_of_week") == "seasonality"
    assert classify_driver("season") == "seasonality"
    assert classify_driver("capacity") == "capacity"
    assert classify_driver("unknown_zzz") == "other"


def test_falla_sin_importancia():
    with pytest.raises(ValueError, match="importancia"):
        explain_from_importance(
            property="x", stay_date="2026-01-01", forecast_rn=10.0,
            feature_values={}, feature_importance={},
        )


# --------------------------- 8.2 ----------------------------------------


def test_capacidad_como_contribucion_negativa():
    """8.2: cuando hay cap, la restricción aparece como contribución negativa."""
    expl = explain_from_importance(
        property="corin",
        stay_date="2026-03-07",
        forecast_rn=100.0,          # final YA capado
        feature_values=_values(),
        feature_importance=_importance(),
        capacity=100.0,
        raw_forecast=120.0,         # modelo esperaba 120
    )

    assert expl.has_capacity_constraint
    constraint = next(
        d for d in expl.drivers if d["driver"] == "capacity_constraint"
    )
    assert constraint["contribution"] < 0, "la restricción debe ser negativa"
    assert constraint["contribution"] == pytest.approx(-20.0)
    assert constraint["business_label"] == "Restricción de capacidad"
    expl.validate()  # sigue sumando el forecast


def test_sin_capacidad_no_hay_constraint():
    expl = explain_from_importance(
        property="corin", stay_date="2026-03-07", forecast_rn=82.0,
        feature_values=_values(), feature_importance=_importance(),
        capacity=100.0, raw_forecast=82.0,  # sin violación
    )
    assert not expl.has_capacity_constraint
    assert all(d["driver"] != "capacity_constraint" for d in expl.drivers)


# --------------------------- 8.3 ----------------------------------------


def test_resumen_negocio_legible():
    expl = explain_from_importance(
        property="corin",
        stay_date="2026-03-07",
        forecast_rn=100.0,        # final ya capado en la capacidad
        feature_values=_values(),
        feature_importance=_importance(),
        capacity=100.0,
        raw_forecast=120.0,       # modelo esperaba 120 > capacidad 100
    )
    summary = expl.business_summary()

    assert summary[0].startswith("Forecast RN: 100")
    joined = "\n".join(summary)
    # términos de negocio presentes (no solo técnicos)
    assert "Reservas actuales" in joined
    assert "Pickup" in joined
    assert "Restricción de capacidad" in joined
    # con signo
    assert any("+" in s for s in summary[1:])


def test_contribuciones_sumen_el_forecast():
    rng = np.random.default_rng(1)
    for _ in range(20):
        imp = dict(zip(_importance().keys(), rng.random(4)))
        imp = {k: v / sum(imp.values()) for k, v in imp.items()}
        forecast = float(rng.uniform(20, 150))
        expl = explain_from_importance(
            property="p", stay_date="2026-01-01", forecast_rn=forecast,
            feature_values=_values(), feature_importance=imp,
        )
        expl.validate()  # levanta ValueError si no suman


def test_validate_detecta_desajuste():
    expl = explain_from_importance(
        property="corin", stay_date="2026-03-07", forecast_rn=82.0,
        feature_values=_values(), feature_importance=_importance(),
    )
    expl.forecast_rn = 999.0  # corromper
    with pytest.raises(ValueError, match="no suman"):
        expl.validate()
