"""Memoria por línea de corrida — openspec 2026-11-10-registry-mode-memory.

`previous_run` (histéresis, drift, retraining) se toma del último
registro del MISMO modo: la verificación real y la publicación operativa
futura comparten archivo pero jamás comparten memoria. El caso que
motivó el cambio: el brazo B del A/B de bias del 5-nov heredó sarimax de
una corrida futura posterior y el A/B quedó confundido.
"""

from __future__ import annotations

from src.monitoring.run_registry import latest_record


def _registro(*records: tuple[str, str]) -> dict:
    """Registro sintético: (run_id, mode) en orden de ejecución."""
    return {
        "registry_version": 1,
        "records": [
            {
                "run_id": run_id,
                "mode": mode,
                "timestamp": f"2026-11-0{i + 1}T12:00:00+00:00",
                "cutoff": "2026-11-04",
                "model_by_property": {"MARINA": f"modelo-{run_id}"},
            }
            for i, (run_id, mode) in enumerate(records)
        ],
    }


def test_memoria_toma_el_ultimo_de_su_modo():
    reg = _registro(
        ("real-1104", "real"),
        ("futuro-1104", "futuro"),
        ("real-1105", "real"),
    )
    # la corrida real pide su propia línea: la última real, NO la futura
    prev = latest_record(reg, mode="real")
    assert prev is not None and prev["run_id"] == "real-1105"


def test_futura_no_contamina_la_real():
    # orden real: el futuro se ejecutó DESPUÉS de la última real —
    # exactamente el caso que confundió el A/B de bias del 5-nov
    reg = _registro(
        ("real-1104", "real"),
        ("futuro-1104", "futuro"),
    )
    prev = latest_record(reg, mode="real")
    assert prev is not None and prev["run_id"] == "real-1104"
    assert prev["model_by_property"]["MARINA"] == "modelo-real-1104"


def test_sin_registro_de_su_modo_memoria_vacia():
    reg = _registro(("futuro-1104", "futuro"))
    assert latest_record(reg, mode="real") is None, (
        "sin corrida real previa: memoria vacía, nunca herencia cruzada"
    )


def test_linea_futura_intacta():
    reg = _registro(
        ("real-1104", "real"),
        ("futuro-1104", "futuro"),
        ("real-1105", "real"),
    )
    prev = latest_record(reg, mode="futuro")
    assert prev is not None and prev["run_id"] == "futuro-1104"


def test_ad_hoc_aislado():
    reg = _registro(
        ("real-1104", "real"),
        ("futuro-1104", "futuro"),
    )
    assert latest_record(reg, mode="ad hoc") is None, (
        "las corridas sueltas no heredan de las líneas productivas"
    )
    assert latest_record(reg) is not None, "el registro sigue consultable"


def test_registro_vacio():
    assert latest_record({"records": []}, mode="real") is None
