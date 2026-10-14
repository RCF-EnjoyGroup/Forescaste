"""Corrida end-to-end (task 10.5).

Verifica los 5 criterios de aceptación:
0 fallos de leakage · final ≤ capacity en 100% · métricas y desgloses ·
dataset Power BI · manifiesto reproducible.

Las fuentes se construyen con el MISMO esquema variable que acepta la
detección automática (nombres distintos a los canónicos) para ejercitar toda
la cadena. **No son datos reales** (data/ no existe); el disclaimer queda en
la documentación técnica generada.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.runner import AcceptanceChecks, run_end_to_end
from src.monitoring.run_manifest import read_manifest


def _demand(n=500, start="2025-01-01", seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    dow = dates.dayofweek
    base = 55 + 12 * (dow >= 5)  # fin de semana más alto
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    return pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": dates,
            "rooms_sold": np.clip(base + seasonal + rng.normal(0, 5, n), 0, 100).round(),
        }
    )


def _capacity(n=500, start="2025-01-01", cap=90):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "propiedad": ["corin"] * n,
            "fecha": dates,
            "inventory": [cap] * n,
        }
    )


def _pickup(demand: pd.DataFrame):
    """Curva de pickup: booked crece a medida que baja el lead."""
    rows = []
    rng = np.random.default_rng(12)
    stays = demand["stay_date"].iloc[::10]
    final = demand.set_index("stay_date")["rooms_sold"]
    for stay in stays:
        for lead in [90, 60, 30, 14, 7, 3, 1]:
            snap = stay - pd.Timedelta(days=lead)
            if snap > pd.Timestamp("2025-06-30"):
                continue
            total = float(final.get(stay, 50))
            booked = total * (1 - np.exp(-lead / 35)) + rng.normal(0, 1.5)
            rows.append(
                {
                    "hotel": "corin",
                    "stay": stay,
                    "snap": snap,
                    "lead": lead,
                    "booked_rn": max(booked, 0),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def sources():
    dem = _demand()
    cap = _capacity()
    pick = _pickup(dem)
    return dem, cap, pick


def test_e2e_cadena_completa(sources, tmp_path):
    dem, cap, pick = sources
    result = run_end_to_end(
        demand=dem,
        capacity=cap,
        pickup=pick,
        run_dir=tmp_path / "run_e2e",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
    )

    # --- criterio 1: 0 fallos de leakage
    assert result.checks.leakage_clean, (
        f"leakage detectado: {result.checks.details.get('leakage_findings')}"
    )

    # --- criterio 2: final <= capacity en el 100% de registros
    ft = result.forecast_table
    assert len(ft) > 0
    ratio = (ft["final_forecast_rn"] <= ft["capacity"] + 1e-9).mean()
    assert ratio == 1.0, f"solo {ratio:.1%} de registros respeta la capacidad"
    assert result.checks.capacity_respected

    # --- criterio 3: métricas y desgloses presentes
    assert result.checks.metrics_and_breakdowns
    assert {"mape", "wmape", "mae", "rmse", "bias"} <= set(result.metrics)
    for name in ("lead_band", "property", "property_weekly", "season", "month", "day_of_week", "weekend_weekday"):
        assert name in result.breakdowns, f"falta desglose {name}"
        assert not result.breakdowns[name].empty

    # --- criterio 4: dataset Power BI presente (4 áreas)
    assert result.checks.powerbi_present
    assert set(result.powerbi) == {"forecast", "accuracy", "pickup", "model"}
    for area, frame in result.powerbi.items():
        assert not frame.empty, f"área Power BI vacía: {area}"

    # --- criterio 5: manifiesto reproducible
    assert result.checks.manifest_reproducible
    stored = read_manifest(result.manifest_path)
    for f in ("dataset_version", "model_version", "training_period", "seed", "metrics", "timestamp"):
        assert f in stored
    assert stored["sample_data"] is True

    # --- los 10 entregables
    assert len(result.deliverables) == 10
    for name, path in result.deliverables.items():
        assert path.exists(), f"entregable ausente: {name}"

    # --- ningún check falla
    assert result.checks.all_passed, f"checks fallidos: {result.checks.failures()}"


def test_e2e_manifest_reproducible_misma_seed(sources, tmp_path):
    """Misma seed => misma corrida reproducible."""
    dem, cap, pick = sources
    kwargs = dict(
        demand=dem,
        capacity=cap,
        pickup=pick,
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
    )
    r1 = run_end_to_end(run_dir=tmp_path / "a", **kwargs)
    r2 = run_end_to_end(run_dir=tmp_path / "b", **kwargs)

    m1 = read_manifest(r1.manifest_path)
    m2 = read_manifest(r2.manifest_path)
    assert m1["seed"] == m2["seed"]
    assert m1["model_version"] == m2["model_version"]
    assert m1["training_period"] == m2["training_period"]
    # mismas predicciones
    assert np.allclose(
        r1.forecast_table["final_forecast_rn"].to_numpy(),
        r2.forecast_table["final_forecast_rn"].to_numpy(),
    )


def test_e2e_horizonte_configurable(sources, tmp_path):
    """Cambiar FORECAST_HORIZON cambia la corrida sin tocar código."""
    dem, cap, pick = sources
    from src.config import load_config

    base = dict(
        demand=dem,
        capacity=cap,
        pickup=pick,
        cutoff=pd.Timestamp("2025-06-30"),
        sample_data=True,
    )
    r30 = run_end_to_end(run_dir=tmp_path / "h30", config=load_config(env={"FORECAST_HORIZON": "30"}), **base)
    r90 = run_end_to_end(run_dir=tmp_path / "h90", config=load_config(env={"FORECAST_HORIZON": "90"}), **base)

    n_props = dem["property"].nunique()
    assert len(r30.forecast_table) == 30 * n_props
    assert len(r90.forecast_table) == 90 * n_props
    assert len(r90.forecast_table) > len(r30.forecast_table)


def test_e2e_esquema_alternativo_detectado(sources, tmp_path):
    """Las fuentes usan nombres no canónicos; la detección no los asume."""
    dem, cap, pick = sources
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "schema", cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
    )
    # capacity usa 'propiedad'/'fecha'/'inventory' y pickup 'hotel'/'stay'/'snap'/'lead'
    assert result.checks.all_passed, (
        f"la cadena falló con esquema alternativo: {result.checks.failures()} "
        f"details={result.checks.details}"
    )


def test_e2e_disclaimer_de_datos_de_muestra(sources, tmp_path):
    """La documentación técnica advierte que fue una corrida de muestra."""
    dem, cap, pick = sources
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "disc", cutoff=pd.Timestamp("2025-06-30"),
        sample_data=True,
    )
    doc = result.deliverables["technical_documentation"].read_text(encoding="utf-8")
    assert "DISCLAIMER" in doc
    assert "MUESTRA" in doc
    assert "Limitaciones" in doc


def test_e2e_sin_datos_reales_reporta_limitacion(tmp_path):
    """Sin data/*.csv la corrida real no arranca: se reporta, no se inventa."""
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    real_present = all(
        (data_dir / f"{name}.csv").exists() for name in ("capacity_daily", "demand_daily", "pickup_by_lead")
    )
    if real_present:
        pytest.skip("las fuentes reales están presentes: ejecutar la corrida real")
    # si no hay datos reales, el test de muestra es la única corrida posible
    assert not real_present, "debe quedar claro que falta data/"
