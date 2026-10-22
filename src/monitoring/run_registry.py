"""Registro de estado entre corridas (spec ``monitoring``).

El manifiesto es por-corrida e inmutable; el registro es el ÚNICO estado
persistido entre corridas y habilita el monitoreo con memoria:

- drift de métricas (10.1): la corrida compara sus métricas contra las de
  la corrida anterior registrada (comparación real, no placeholder);
- política de re-entrenamiento (10.4): ``days_since_last`` y
  ``new_records`` reflejan el estado real del registro;
- estabilidad del campeón: la selección por propiedad retiene al
  incumbente salvo evidencia material (histéresis entre corridas);
- tracking vintage: puntuación del forecast publicado anterior contra los
  reales que se materializaron desde entonces.

El archivo es append-only y se escribe atómicamente (temp + rename); un
registro corrupto o ausente JAMÁS rompe la corrida (devuelve vacío).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any

REGISTRY_VERSION = 1
REGISTRY_FILENAME = "run_registry.json"
# modos registrados: "real" (verificación) y "futuro" (pronóstico publicado)
OPERATIONAL_MODE = "futuro"


@dataclass
class RunRecord:
    """Estado mínimo de una corrida para que la SIGUIENTE monitoree contra ella."""

    run_id: str
    run_dir: str
    mode: str  # "real" | "futuro" | otro
    timestamp: str  # ISO-8601 UTC
    cutoff: str  # fecha de corte (YYYY-MM-DD)
    training_end: str  # último día con el que entrenó (= cutoff)
    metrics: dict[str, Any] = field(default_factory=dict)
    metrics_source: str = ""  # horizon_actuals | injected_backtest | none
    champion: str = ""
    model_by_property: dict[str, str] = field(default_factory=dict)
    interval_coverage: dict[str, Any] = field(default_factory=dict)
    vintage_tracking: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _sanitize(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def _sanitize(obj: Any) -> Any:
    """NaN/inf → None: el registro es JSON estricto, sin depender del caller."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def load_registry(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Lee el registro; ausente o corrupto → registro vacío (jamás rompe)."""
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"registry_version": REGISTRY_VERSION, "records": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return {"registry_version": REGISTRY_VERSION, "records": []}
    return payload


def latest_record(
    registry: dict[str, Any], mode: str | None = None
) -> dict[str, Any] | None:
    """Último registro (opcionalmente del modo pedido); None si no hay."""
    records = registry.get("records") or []
    if mode is not None:
        records = [r for r in records if str(r.get("mode")) == mode]
    return dict(records[-1]) if records else None


def append_record(path: str | os.PathLike[str], record: RunRecord) -> dict[str, Any]:
    """Añade una corrida al registro con escritura atómica y lo devuelve.

    Single-process documentado: el motor corre una corrida a la vez; la
    escritura usa temp+rename para que un corte a mitad nunca deje un
    JSON a medias.
    """
    path = os.fspath(path)
    registry = load_registry(path)
    registry.setdefault("records", []).append(record.to_dict())
    registry["registry_version"] = REGISTRY_VERSION
    _atomic_write(path, registry)
    return registry


def _atomic_write(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)) or ".",
        prefix="registry-", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
