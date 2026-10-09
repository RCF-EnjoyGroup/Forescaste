"""Tests del config central (task 1.4).

Verifica que cambiar el horizonte (por config o por variable de entorno) sin
tocar código se refleja en la corrida de prueba.
"""

import numpy as np
import yaml

from src.config import load_config


def _write_config(path, horizon=90, freq="monthly", min_hist=365, seed=42):
    payload = {
        "random_seed": seed,
        "log_level": "INFO",
        "forecast": {"horizon_days": horizon, "percentiles": [10, 50, 90]},
        "retraining": {"frequency": freq, "min_history_days": min_hist},
        "data": {
            "sources": {
                "capacity": "data/capacity_daily.csv",
                "demand": "data/demand_daily.csv",
                "pickup": "data/pickup_by_lead.csv",
            },
            "column_hints": {"property": ["property", "hotel_id"]},
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_config_por_defecto_del_proyecto():
    cfg = load_config()
    assert cfg.forecast_horizon in (30, 60, 90)
    assert cfg.random_seed == 42
    assert cfg.retrain_frequency == "monthly"
    assert cfg.min_history_days == 365
    assert set(cfg.sources) == {"capacity", "demand", "pickup"}


def test_cambiar_horizonte_en_config_sin_tocar_codigo(tmp_path):
    """Horizonte 30 vs 90: dos configs distintas, cero cambios de código."""
    cfg30 = load_config(_write_config(tmp_path / "a.yaml", horizon=30))
    cfg90 = load_config(_write_config(tmp_path / "b.yaml", horizon=90))

    assert cfg30.forecast_horizon == 30
    assert cfg90.forecast_horizon == 90

    # La corrida de prueba (fechas de estancia generadas) refleja el horizonte:
    def run(cfg, as_of="2026-10-09"):
        start = np.datetime64(as_of)
        return list(start + np.arange(1, cfg.forecast_horizon + 1))

    assert len(run(cfg30)) == 30
    assert len(run(cfg90)) == 90


def test_variable_de_entorno_sobreescribe_horizonte(tmp_path):
    path = _write_config(tmp_path / "c.yaml", horizon=90)
    cfg = load_config(path, env={"FORECAST_HORIZON": "60"})
    assert cfg.forecast_horizon == 60

    cfg_default = load_config(path, env={})
    assert cfg_default.forecast_horizon == 90, "sin env var se respeta el YAML"


def test_env_sobreescribe_retraining_y_seed(tmp_path):
    path = _write_config(tmp_path / "d.yaml", freq="monthly", min_hist=365, seed=42)
    cfg = load_config(
        path,
        env={"RETRAIN_FREQUENCY": "weekly", "MIN_HISTORY": "120", "RANDOM_SEED": "7"},
    )
    assert cfg.retrain_frequency == "weekly"
    assert cfg.min_history_days == 120
    assert cfg.random_seed == 7


def test_horizonte_invalido_falla(tmp_path):
    path = _write_config(tmp_path / "e.yaml", horizon=0)
    try:
        load_config(path, env={})
    except ValueError as exc:
        assert "positivo" in str(exc)
    else:
        raise AssertionError("horizon_days=0 debió fallar")
