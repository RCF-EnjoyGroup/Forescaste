# Design — long-horizon-folds

## Context

La selección de campeón se entrena con folds de 30d. Las curvas
estacional/feriado operan a leads ≥ 30 (`season_min_lead`), así que los folds
evalúan exactamente el rango donde los candidatos son idénticos: la selección
es ciega al régimen largo. LIREL es el caso visible (0.150→0.188 con la curva
estacional en la ventana de verificación, y aun así elegido pickup_curve).

## Goals / Non-Goals

**Goals** — Ventanas de fold = horizonte operativo (90d) con tiling exacto del
año anterior al corte; DM con corrección HAC al largo real; benchmark y
producción entrenan la MISMA curva (bins [1, 91]); runtime medido antes de
cerrar; seleccionable por config para afinar costo/evidencia por entorno.

**Non-Goals** — Cambiar el mecanismo de selección (composite, parsimonia) ni
los criterios de desempate; folds ponderados por lead; validación de cobertura
de bandas (otra palanca).

## Decisions

### D1 — Ventanas de 90d con cortes trimestrales existentes: tiling exacto
`fold_cutoffs = cutoff − 90·(n_folds..1)` con ventana `(c, c+90]`: las 4
ventanas tesian `[cutoff−360, cutoff]` sin solaparse entre sí. El último fold
termina EN cutoff; la ventana de reporte empieza en cutoff+1 → separación
selección/reporte se preserva (el test existente lo verifica con
`max(stay) ≤ cutoff − (SPACING − HORIZON) = cutoff`).
Alternativa descartada: añadir folds largos ADEMÁS de los cortos — ventanas
solapadas = evidencia correlacionada y doble conteo del rango 1–30.

### D2 — Seleccionar sobre la métrica operativa, no la de corto lead
La media de WMAPE por fold de 90d pondera todos los leads por volumen real de
estancias — el mismo mix que consume el negocio en el horizonte operativo.
Costo honesto: los números del benchmark no son comparables con corridas
previas (los WMAPEs suben porque ahora incluyen 60–89). Documentado en el
proposal; la comparabilidad histórica se lee por banda de lead del accuracy
report, que no cambia de metodología.

### D3 — Curva de fold = curva de producción
Con `horizon=90` en el fold, pickup_curve se entrena con leads ≤ 90 (bins
[1, 91]) — idéntico al forecast operativo. Antes los folds usaban bins [1, 31]:
el benchmark evaluaba una curva con resolución distinta de la desplegada. El
panel ML de los GBM también gana (leads ≤ 90 en `_booked_training_panel`).

### D4 — DM al largo real, derivado del panel
`_dm_horizon_from_panel(error_panel)`: mediana de fechas distintas por fold
(~90). Vacío → default 30. Threaded `run_benchmark(dm_horizon=...) →
select_champion(dm_horizon=...)`. Con ventanas de 90d el panel DM crece de
~82 a ~340 pares por modelo — la significancia deja de ser anécdota.

### D5 — Config `benchmark` para el trade-off costo/evidencia
`benchmark: {fold_horizon_days: 90, n_folds: 4}` en `config.yaml`: sin tocar
código, un entorno puede bajar a 30/4 (barato, corto lead) o subir a 90/8 (dos
años de evidencia, ~2× runtime). El default 90/4 = un año completo tesiano,
cada estación evaluada una vez.

## Risks / Trade-offs

- [Runtime: rollout ML 3× más largo por fold] → medir en la corrida real; si
  el benchmark supera ~10 min, el config permite bajar sin tocar código.
- [Selección por propiedad más volátil (cada fold ahora pesa más)] → el
  `min_folds` de 3 y el fallback a campeón global siguen protegiendo; la
  volatilidad es el precio de ver el régimen real.
- [WMAPEs del benchmark suben vs corridas previas] → esperado y honesto:
  ahora incluyen la banda 30–89; documentado, no es regresión.
- [GBM empeoran en rollouts de 90 pasos y bajan del tablero] → eso ES la
  verdad operativa de un modelo autoregresivo a 90 pasos; el bake-off la
  estaba ocultando.

## Migration Plan

1. Config + defaults + DM threading → tests (tiling, cobertura 12 meses, largo
   de ventana, DM-horizon) → corrida real con timing → comparar selecciones
   (watch item LIREL) → suite → archive. Rollback: `benchmark.fold_horizon_days: 30`.
