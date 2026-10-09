"""Run manifest — reproducibilidad de cada corrida (spec ``monitoring``).

Cada ejecución del motor escribe un manifiesto JSON con: versión del dataset,
versión del modelo, periodo de entrenamiento, periodo de validación,
configuración de features, hiperparámetros, métricas, timestamp y random seed
(D10 del design: los directorios de corrida se inmutan).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "run_manifest.json"

REQUIRED_FIELDS = (
    "dataset_version",
    "model_version",
    "training_period",
    "validation_period",
    "feature_config",
    "hyperparameters",
    "metrics",
    "seed",
    "timestamp",
)


class ManifestError(ValueError):
    """El manifiesto no cumple el contrato de campos requeridos."""


@dataclass
class RunManifest:
    """Contrato de reproducibilidad de una corrida."""

    dataset_version: str
    model_version: str
    training_period: dict[str, str]
    validation_period: dict[str, str]
    feature_config: dict[str, Any]
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    seed: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        missing = [f for f in REQUIRED_FIELDS if getattr(self, f, None) is None]
        if missing:
            raise ManifestError(f"Run manifest incompleto, faltan campos: {missing}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload

    def write(self, run_dir: str | Path) -> Path:
        """Escribe el manifiesto en ``run_dir`` (inmutable tras escribirse)."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / MANIFEST_FILENAME
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, ensure_ascii=False)
        return path


def read_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ManifestError(f"Manifest en disco incompleto, faltan campos: {missing}")
    return payload


def fingerprint(obj: Any) -> str:
    """Huella estable (SHA-256) de un objeto para comparar corridas idénticas."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def same_configuration(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Dos manifiestos representan la misma corrida si su configuración coincide.

    Ignora ``timestamp`` (inherente a cada ejecución) y ``metrics`` (resultado).
    """
    ignore = {"timestamp", "metrics"}
    return fingerprint({k: v for k, v in a.items() if k not in ignore}) == fingerprint(
        {k: v for k, v in b.items() if k not in ignore}
    )
