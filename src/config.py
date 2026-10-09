"""Configuración central del motor.

Carga ``config.yaml`` (raíz del proyecto) y permite sobreescribir valores por
variables de entorno (``FORECAST_HORIZON``, ``RETRAIN_FREQUENCY``,
``MIN_HISTORY``, ``RANDOM_SEED``) de modo que cambiar el horizonte u otra
política NO requiere tocar código (spec ``reporting`` / task 1.4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_ENV_MAP = {
    "FORECAST_HORIZON": ("forecast_horizon", int),
    "RETRAIN_FREQUENCY": ("retrain_frequency", str),
    "MIN_HISTORY": ("min_history_days", int),
    "RANDOM_SEED": ("random_seed", int),
}


@dataclass(frozen=True)
class EngineConfig:
    """Vista inmutable de la configuración del motor."""

    random_seed: int
    forecast_horizon: int
    percentiles: tuple[int, ...]
    retrain_frequency: str
    min_history_days: int
    sources: dict[str, str]
    column_hints: dict[str, list[str]]
    log_level: str = "INFO"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def source_paths(self) -> dict[str, Path]:
        base = DEFAULT_CONFIG_PATH.parent
        return {k: (base / v if not Path(v).is_absolute() else Path(v)) for k, v in self.sources.items()}


def _coerce_env(value: str, kind: type) -> Any:
    return kind(value)


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> EngineConfig:
    """Carga la configuración desde YAML + overrides de entorno.

    Parameters
    ----------
    path:
        Ruta alternativa al ``config.yaml`` por defecto.
    env:
        Mapping de overrides estilo entorno (por defecto ``os.environ``);
        se inyecta en tests para no depender del entorno real.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    env = os.environ if env is None else env
    forecast = dict(data.get("forecast") or {})
    retraining = dict(data.get("retraining") or {})

    values: dict[str, Any] = {
        "random_seed": int(data.get("random_seed", 42)),
        "forecast_horizon": int(forecast.get("horizon_days", 90)),
        "percentiles": tuple(int(p) for p in forecast.get("percentiles", [10, 50, 90])),
        "retrain_frequency": str(retraining.get("frequency", "monthly")).lower(),
        "min_history_days": int(retraining.get("min_history_days", 365)),
    }

    for env_name, (key, kind) in _ENV_MAP.items():
        if env_name in env and env[env_name] != "":
            values[key] = _coerce_env(env[env_name], kind)

    if values["forecast_horizon"] <= 0:
        raise ValueError(f"forecast_horizon debe ser positivo, recibido: {values['forecast_horizon']}")
    if not values["percentiles"]:
        raise ValueError("forecast.percentiles no puede estar vacío")

    data_cfg = dict(data.get("data") or {})
    sources = dict(data_cfg.get("sources") or {})
    hints = {k: list(v) for k, v in (data_cfg.get("column_hints") or {}).items()}

    return EngineConfig(
        **values,
        sources=sources,
        column_hints=hints,
        log_level=str(data.get("log_level", "INFO")),
        raw=data,
    )
