"""Regresión del crash del modo futuro — openspec 2026-11-10-fix-futuro-registry-access.

`cfg.monitoring` es un MAPA (src/config.py): la ruta del registro se lee
con `.get`, jamás por atributo (acceso por atributo crasheaba todo
`--futuro` desde el commit c255eb4). El helper también filtra por modo
(memoria por línea, openspec 2026-11-10-registry-mode-memory).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.forecasting.run_real import _incumbente_previo


class _CfgDict:
    """Config-like: monitoring como MAPA (como src.config)."""

    def __init__(self, monitoring: dict) -> None:
        self.monitoring = monitoring


def _escribir_registro(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"registry_version": 1, "records": records}),
                    encoding="utf-8")


def _rec(run_id: str, mode: str, cutoff: str) -> dict:
    return {"run_id": run_id, "mode": mode, "cutoff": cutoff,
            "timestamp": "2026-11-05T12:00:00+00:00",
            "model_by_property": {"MARINA": f"modelo-{run_id}"}}


def test_lee_ruta_como_mapa_no_como_atributo(tmp_path):
    path = tmp_path / "registry.json"
    _escribir_registro(path, [_rec("futuro-1105", "futuro", "2026-11-05")])
    cfg = _CfgDict({"registry_path": str(path)})
    prev = _incumbente_previo(cfg, mode="futuro")
    assert prev is not None and prev["run_id"] == "futuro-1105"
    # contra el código viejo esto reventaba con AttributeError


def test_filtra_por_modo_memoria_por_linea(tmp_path):
    path = tmp_path / "registry.json"
    _escribir_registro(path, [
        _rec("real-1104", "real", "2026-11-04"),
        _rec("futuro-1105", "futuro", "2026-11-05"),
    ])
    cfg = _CfgDict({"registry_path": str(path)})
    prev = _incumbente_previo(cfg, mode="futuro")
    assert prev is not None and prev["run_id"] == "futuro-1105", (
        "el modo futuro encadena al último futuro, no al último registro"
    )


def test_sin_ruta_memoria_vacia():
    assert _incumbente_previo(_CfgDict({})) is None
    assert _incumbente_previo(_CfgDict({"registry_path": None})) is None


def test_registro_inexistente_memoria_vacia(tmp_path):
    cfg = _CfgDict({"registry_path": str(tmp_path / "no_existe.json")})
    assert _incumbente_previo(cfg) is None, "un archivo ausente jamás rompe la corrida"


def test_registro_corrupto_memoria_vacia(tmp_path):
    path = tmp_path / "roto.json"
    path.write_text("{no es json", encoding="utf-8")
    cfg = _CfgDict({"registry_path": str(path)})
    assert _incumbente_previo(cfg) is None


def test_config_real_del_proyecto_es_mapa():
    from src.config import load_config

    cfg = load_config()
    assert isinstance(cfg.monitoring, dict), (
        "si algún día monitoring deja de ser mapa, este test guía la lectura"
    )
    assert "registry_path" in cfg.monitoring
