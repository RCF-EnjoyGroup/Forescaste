"""Comparación de la corrida nueva contra el baseline (pre-mejoras).

Uso:
    python scripts/compare_runs.py <run_nueva> <run_baseline>
"""

import json
import sys
from pathlib import Path


def _load(run: str, name: str) -> dict:
    p = Path(run) / name
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main(new_run: str, base_run: str) -> int:
    acc_new = _load(new_run, "forecast_accuracy_report.json")
    acc_base = _load(base_run, "forecast_accuracy_report.json")
    man_new = _load(new_run, "run_manifest.json")
    man_base = _load(base_run, "run_manifest.json")

    print("=" * 74)
    print("GLOBAL")
    rows = [("wmape", "WMAPE"), ("mape", "MAPE"), ("bias", "Bias")]
    for key, label in rows:
        n = acc_new["global"].get(key)
        b = acc_base["global"].get(key)
        fmt = (lambda v: f"{v:+.4f}") if key == "bias" else (lambda v: f"{v:.4f}")
        print(f"  {label:8} base={fmt(b) if b is not None else '—':>10}  nuevo={fmt(n) if n is not None else '—':>10}")

    print("=" * 74)
    print("COBERTURA P10-P90 (nominal 80%)")
    for label, acc in (("base", acc_base), ("nuevo", acc_new)):
        cov = acc.get("interval_coverage") or {}
        g = cov.get("global") or {}
        print(f"  {label}: global={g.get('coverage', float('nan')):.3f}", end="")
        for row in cov.get("by_property") or []:
            print(f" | {row['property']}={row['coverage']:.2f}", end="")
        print()
    nuevo_cov = (acc_new.get("interval_coverage") or {}).get("by_property") or []
    base_cov = (acc_base.get("interval_coverage") or {}).get("by_property") or []
    if nuevo_cov and base_cov:
        desv_n = sum(abs(r["coverage"] - 0.80) for r in nuevo_cov) / len(nuevo_cov)
        desv_b = sum(abs(r["coverage"] - 0.80) for r in base_cov) / len(base_cov)
        print(f"  desviación media |cov−0.80| por propiedad: base={desv_b:.3f} → nuevo={desv_n:.3f}")

    print("=" * 74)
    print("MONITOREO")
    for label, man in (("base", man_base), ("nuevo", man_new)):
        mon = man.get("monitoring") or {}
        alerts = mon.get("drift_alerts") or []
        print(f"  {label}: n_drift_alerts={mon.get('n_drift_alerts')}")
        for a in alerts:
            print(f"    - [{a.get('severity')}] {a.get('message')}")
        print(f"    retraining: {mon.get('retraining_decision', {}).get('reason')}")
        if mon.get("retraining_state"):
            print(f"    estado: {mon.get('retraining_state')}")

    print("=" * 74)
    print("NUEVO EN ESTA CORRIDA")
    for key in ("bias_correction", "cold_start_properties", "vintage_tracking",
                "forecast_revision", "previous_run"):
        v = man_new.get(key)
        if key == "bias_correction" and isinstance(v, dict):
            print(f"  bias_correction: {v.get('n_corrected')}/{v.get('n_cells')} celdas corregidas")
            for c in (v.get("cells") or []):
                if c.get("applied"):
                    print(f"    - {c['property']} {c['lead_band']}: bias={c['rel_bias']:+.3f} → factor={c['factor']:.3f}")
        else:
            print(f"  {key}: {v}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
