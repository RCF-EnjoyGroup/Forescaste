# -*- coding: utf-8 -*-
"""Evidencia del change registry-mode-memory (openspec 2026-11-10).

Recorre el registro REAL y, para cada corrida, muestra el previous_run
bajo la regla vieja (último registro de cualquier modo) vs la nueva
(último registro del mismo modo). El caso documentado: real-20261105-1442
(brazo B del A/B de bias) heredaba 'futuro-20261104-1525' (MARINA=sarimax
— el incumbente que confundió el A/B); con la regla nueva hereda
real-20261104-1514 (MARINA=lightgbm, comparable).

Uso:  python -X utf8 scripts/ab_registry_memory.py
"""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY = "outputs/monitoring/run_registry.json"


def main() -> int:
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    records = reg.get("records") or []
    if not records:
        print("registro vacío")
        return 2

    def ultimo(hasta: int, mode: str | None) -> dict | None:
        pool = records[:hasta]
        if mode is not None:
            pool = [r for r in pool if str(r.get("mode")) == mode]
        return pool[-1] if pool else None

    cambios = 0
    print("%-26s %-8s | %-30s | %-30s" % ("corrida", "modo", "old (cualquier modo)", "new (mismo modo)"))
    print("-" * 100)
    for i, r in enumerate(records):
        old = ultimo(i, None)
        new = ultimo(i, str(r.get("mode")))
        old_id = str((old or {}).get("run_id", "—"))
        new_id = str((new or {}).get("run_id", "—"))
        marca = "  <-- CAMBIA" if old_id != new_id else ""
        if old_id != new_id:
            cambios += 1
        mf_old = (old or {}).get("model_by_property", {}).get("MARINA", "?")
        mf_new = (new or {}).get("model_by_property", {}).get("MARINA", "?")
        print("%-26s %-8s | %-20s M=%-8s | %-20s M=%-8s%s" % (
            r.get("run_id", "?"), str(r.get("mode", "?")),
            old_id, mf_old, new_id, mf_new, marca))
    print()
    print(f"corridas cuya memoria cambia con la regla nueva: {cambios} de {len(records)}")
    print("el caso del A/B de bias (real-20261105-1442): old= futuro-20261104-1525 (sarimax),")
    print("new= real-20261104-1514 (lightgbm) — brazos comparables en modelo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
