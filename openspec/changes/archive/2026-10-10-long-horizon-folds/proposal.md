## Why

La selección de campeón (global y por propiedad) se decide con folds de 30 días — un rango donde las curvas estacional/de feriado están gateadas (`season_min_lead: 30`) y los candidatos son indistinguibles. El resultado es una selección **ciega al régimen largo** (leads 30–90): LIREL empeoró con la curva estacional (0.150→0.188 en la ventana de verificación) y aun así los folds le siguieron eligiendo pickup_curve, porque a leads 1–30 las variantes son idénticas. La selección debe optimizar la métrica operativa real: el WMAPE del horizonte completo de 90 días.

## What Changes

- **Ventanas de fold de 30d → 90d** (`benchmark.fold_horizon_days`, default 90): los cortes trimestrales existentes hacen que las 4 ventanas **tesian exactamente el año anterior al corte** — cada estación evaluada una vez, sin solape entre folds ni con la ventana de reporte `(cutoff, cutoff+90]` (el último fold termina en `cutoff`).
- **Cobertura estacional completa**: con 30d los folds muestreaban 120 días del año; con 90d cubren 360 — incluida la temporada alta, que antes caía en a lo sumo un fold parcial.
- **DM con corrección a pasos reales**: la significancia Diebold-Mariano usa el largo de la ventana de fold derivado del panel de errores (90 pasos, antes 30) y el panel crece de ~82 a ~340 pares por modelo.
- **Consistencia de curva benchmark↔producción**: los folds entrenan pickup_curve con `horizon=90` — mismos bins [1, 91] que el forecast operativo (antes los folds usaban bins [1, 31], una curva distinta de la desplegada).
- **Runtime acotado**: los modelos estadísticos no cambian de costo (un fit por fold, solo más pasos de predicción); los GBM hacen rollout recursivo de 90 pasos (3× que antes) — medido en la corrida real antes de cerrar el default.

## Capabilities

### New Capabilities

<!-- Ninguna. -->

### Modified Capabilities

- `evaluation`: MODIFIED — Validación temporal exclusiva: las ventanas de fold cubren el horizonte operativo completo (leads 1–`fold_horizon_days`) y tesian el año anterior al corte; MODIFIED — Significancia estadística de la selección: la corrección HAC usa el largo real de la ventana de fold.

## Impact

- **Código**: `config.yaml` sección `benchmark` + `src/config.py`; `src/forecasting/run_real.py` (default 90 + wiring); `src/reporting/benchmark.py` (`dm_horizon` threaded); `src/forecasting/runner.py` (derivación del DM-horizon desde el panel).
- **Comportamiento observable**: la selección por propiedad puede cambiar (LIREL/MARINA son los candidatos naturales a moverse); el benchmark MD reporta WMAPE sobre ventanas de 90d; el panel DM del manifiesto crece.
- **Riesgo**: la media de WMAPE por fold ahora pondera los leads largos (donde todos los modelos son peores) — es la métrica operativa correcta, pero los números del benchmark no son comparables 1:1 con corridas anteriores; el runtime del benchmark crece (medir).
