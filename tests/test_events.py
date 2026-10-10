"""Tests del cambio add-event-features.

- Carga del archivo de eventos: válido / ausente (None) / inválido (duro).
- Calendario combinado: ventanas globales y con scoping por propiedad;
  ``is_holiday`` nunca marca eventos; la distancia ML los cubre.
- Curva: un evento scoped solo condiciona a SU propiedad.
- e2e: manifiesto declara la fuente activa o ausente.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig, load_config
from src.features.events import EventsError, load_events
from src.features.holidays import build_holiday_calendar, calendar_with_events
from src.forecasting.runner import run_end_to_end
from src.models.baselines import PickupCurveModel
from src.monitoring.run_manifest import read_manifest


# --------------------------------------------------------------------------
# 1. carga
# --------------------------------------------------------------------------

def _write_events(tmp_path, rows: list[dict]) -> object:
    path = tmp_path / "events.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_carga_eventos_valida(tmp_path):
    path = _write_events(
        tmp_path,
        [
            {"event_name": "Congreso Tech", "event_date": "2026-05-10", "property": "CORIN"},
            {"event_name": "Festival Sol", "event_date": "2026-09-12", "property": None},
        ],
    )
    events = load_events(path)
    assert events is not None and len(events) == 2
    assert events["event_date"].tolist() == [
        pd.Timestamp("2026-05-10"), pd.Timestamp("2026-09-12")
    ]
    assert events.loc[0, "property"] == "CORIN"
    assert events.loc[1, "property"] is None


def test_carga_eventos_ausente(tmp_path):
    assert load_events(tmp_path / "no_existe.csv") is None


def test_carga_eventos_fecha_ilegible_falla(tmp_path):
    path = _write_events(
        tmp_path, [{"event_name": "Roto", "event_date": "12-mayo-202x", "property": None}]
    )
    with pytest.raises(EventsError, match="fecha ilegible"):
        load_events(path)


def test_carga_eventos_duplicados_falla(tmp_path):
    path = _write_events(
        tmp_path,
        [
            {"event_name": "Congreso Tech", "event_date": "2026-05-10", "property": "CORIN"},
            {"event_name": "Congreso Tech", "event_date": "2026-05-10", "property": "CORIN"},
        ],
    )
    with pytest.raises(EventsError, match="duplicados"):
        load_events(path)


def test_carga_eventos_columnas_alternativas(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        {"evento": ["Concierto X"], "fecha": ["2026-07-04"], "propiedad": ["FIESTA"]}
    ).to_csv(path, index=False)
    events = load_events(path)
    assert events["event_name"].iloc[0] == "Concierto X"
    assert events["property"].iloc[0] == "FIESTA"


# --------------------------------------------------------------------------
# 2. calendario combinado
# --------------------------------------------------------------------------

def _events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_name": "Congreso Tech", "event_date": pd.Timestamp("2026-05-10"), "property": "CORIN"},
            {"event_name": "Festival Sol", "event_date": pd.Timestamp("2026-09-12"), "property": None},
        ]
    )


def test_calendario_con_eventos_globales_y_scoped():
    base = build_holiday_calendar(range(2026, 2027), ("CR", "US"), window_days=4)
    combined = calendar_with_events(base, _events_frame())

    # el evento global (Festival Sol 12-sep) entra al mapa global; 10-sep le
    # pertenece (2 días) pese a Independencia CR (15-sep, a 5 días)
    assert combined.window_key(pd.Timestamp("2026-09-10")) == "festival_sol"
    # el evento scoped solo existe para SU propiedad
    assert combined.window_key(pd.Timestamp("2026-05-12"), "CORIN") == "congreso_tech"
    assert combined.window_key(pd.Timestamp("2026-05-12"), "FIESTA") is None
    # feriado del calendario público sigue disponible
    assert combined.window_key(pd.Timestamp("2026-04-03")) == "semana_santa"
    # las fechas de eventos entran a dates (distancia ML) pero NO a is_holiday
    assert combined.days_to_nearest(pd.Timestamp("2026-09-12")) == 0.0
    assert combined.days_to_nearest(pd.Timestamp("2026-05-10")) == 0.0
    assert not combined.is_holiday(pd.Timestamp("2026-05-10"))
    assert not combined.is_holiday(pd.Timestamp("2026-09-12"))
    assert combined.is_holiday(pd.Timestamp("2026-12-25"))
    # describe reporta totales
    desc = combined.describe()
    assert desc["n_dates"] > desc["n_holiday_dates"]
    assert desc["n_scoped_properties"] == 1


def test_calendario_sin_eventos_queda_igual():
    base = build_holiday_calendar(range(2026, 2027), ("CR", "US"), window_days=4)
    assert calendar_with_events(base, None) is base


# --------------------------------------------------------------------------
# 3. curva con scoping
# --------------------------------------------------------------------------

def _curves_with_event():
    """Congreso Tech (10-may-2026, scoped a CORIN): pickup 95; resto de
    temporada shoulder (mayo): 50; baja (julio): 20."""
    rows = []
    stays = (
        # congreso (ventana ±4 del 10-may: 6–14 may) — CORIN
        [("CORIN", "2026-05-08", 95.0), ("CORIN", "2026-05-10", 95.0), ("CORIN", "2026-05-13", 95.0)]
        + [("CORIN", "2025-05-09", 95.0), ("CORIN", "2025-05-11", 95.0), ("CORIN", "2025-05-13", 95.0)]
        # shoulder normal CORIN (fuera de ventana)
        + [("CORIN", "2026-05-20", 50.0), ("CORIN", "2026-05-25", 50.0), ("CORIN", "2026-04-15", 50.0),
           ("CORIN", "2026-04-20", 50.0), ("CORIN", "2025-05-20", 50.0), ("CORIN", "2025-05-25", 50.0)]
        # FIESTA en las MISMAS fechas del congreso (pickup normal 50)
        + [("FIESTA", "2026-05-08", 50.0), ("FIESTA", "2026-05-10", 50.0), ("FIESTA", "2026-05-13", 50.0),
           ("FIESTA", "2025-05-09", 50.0), ("FIESTA", "2025-05-11", 50.0), ("FIESTA", "2025-05-13", 50.0)]
    )
    for prop, stay_s, pickup in stays:
        rows.append(
            {"property": prop, "stay_date": pd.Timestamp(stay_s),
             "lead_days": 60, "pickup_rn": pickup}
        )
    return pd.DataFrame(rows)


def test_evento_scoped_no_contamina_otra_propiedad():
    base = build_holiday_calendar(range(2025, 2027), ("CR", "US"), window_days=4)
    combined = calendar_with_events(base, _events_frame())
    model = PickupCurveModel(
        n_bins=2, season_conditional=True, holiday_conditional=True,
        min_stays_per_bucket=3, min_stays_per_holiday_bucket=3,
        season_min_lead=30, holiday_calendar=combined,
    ).fit(_curves_with_event())

    # CORIN dentro de la ventana del congreso (lead ≥ umbral) → curva del evento
    req_corin = pd.DataFrame(
        [{"property": "CORIN", "stay_date": pd.Timestamp("2026-05-11"),
          "lead_days": 60, "current_booked_rn": 10.0}]
    )
    corin = model.predict(req_corin)["expected_pickup_rn"].iloc[0]
    assert corin == pytest.approx(95.0, abs=0.1)

    # FIESTA en la MISMA fecha → SIN curva del evento (season/property ~50)
    req_fiesta = pd.DataFrame(
        [{"property": "FIESTA", "stay_date": pd.Timestamp("2026-05-11"),
          "lead_days": 60, "current_booked_rn": 10.0}]
    )
    fiesta = model.predict(req_fiesta)["expected_pickup_rn"].iloc[0]
    assert fiesta == pytest.approx(50.0, abs=0.1), (
        "un evento scoped a CORIN no puede condicionar la curva de FIESTA"
    )
    # el bucket del evento solo existe para CORIN
    assert set(p for p, _ in model.holiday_curves_) == {"CORIN"}


# --------------------------------------------------------------------------
# 4. config + e2e
# --------------------------------------------------------------------------

def test_config_events_path():
    cfg = load_config()
    assert cfg.events_path == "data/events.csv"


def _events_config(events_path: str) -> EngineConfig:
    base = load_config()
    return EngineConfig(
        random_seed=42, forecast_horizon=30, percentiles=(10, 50, 90),
        retrain_frequency="monthly", min_history_days=365,
        sources=base.sources, column_hints=base.column_hints,
        excluded_properties=[], pickup_curve={},
        holidays={"countries": ["CR", "US"], "window_days": 4},
        events_path=events_path,
    )


def test_e2e_con_y_sin_eventos(tmp_path):
    from tests.test_long_horizon import (
        _capacity_3props, _demand_3props, _pickup_3props,
    )

    dem = _demand_3props()
    cap = _capacity_3props()
    pick = _pickup_3props(dem)
    events_path = _write_events(
        tmp_path, [{"event_name": "Congreso Tech", "event_date": "2025-03-15",
                     "property": "corin"}]
    )

    base_kwargs = dict(
        demand=dem, capacity=cap, pickup=pick,
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
    )
    result_on = run_end_to_end(
        run_dir=tmp_path / "on", config=_events_config(str(events_path)), **base_kwargs
    )
    assert result_on.checks.all_passed, (
        f"checks fallidos: {result_on.checks.failures()} details={result_on.checks.details}"
    )
    stored_on = read_manifest(result_on.manifest_path)
    assert stored_on["events"]["source"] == str(events_path)
    assert stored_on["events"]["n_events"] == 1
    assert stored_on["events"]["n_scoped"] == 1
    assert stored_on["holiday_calendar"]["n_dates"] > stored_on["holiday_calendar"]["n_holiday_dates"]

    result_off = run_end_to_end(
        run_dir=tmp_path / "off",
        config=_events_config(str(tmp_path / "no_existe.csv")), **base_kwargs
    )
    assert result_off.checks.all_passed
    stored_off = read_manifest(result_off.manifest_path)
    assert stored_off["events"] == {"source": None}
    # inertes-sin-datos: el evento (15-mar-2025) queda FUERA del horizonte
    # (jul-2025) → el forecast on/off debe ser idéntico
    assert np.allclose(
        result_on.forecast_table["final_forecast_rn"].to_numpy(),
        result_off.forecast_table["final_forecast_rn"].to_numpy(),
    )


def test_manifiesto_ausencia_de_eventos_sin_configurar(tmp_path):
    from tests.test_long_horizon import (
        _capacity_3props, _demand_3props, _pickup_3props,
    )

    result = run_end_to_end(
        demand=_demand_3props(), capacity=_capacity_3props(),
        pickup=_pickup_3props(_demand_3props()),
        run_dir=tmp_path / "no_events",
        cutoff=pd.Timestamp("2025-06-30"),
        sample_data=True,
        config=_events_config(str(tmp_path / "ausente.csv")),
    )
    stored = read_manifest(result.manifest_path)
    assert stored["events"] == {"source": None}
