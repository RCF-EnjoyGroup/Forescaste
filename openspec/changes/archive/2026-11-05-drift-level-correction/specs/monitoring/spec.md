# monitoring — delta

## MODIFIED Requirements

### Requirement: Tracking signal del forecast publicado

#### Scenario: Corrección de nivel por drift (knob, default por A/B)
- **WHEN** `monitoring.level_correction.enabled` está activo y la corrida es `futuro` con una publicación previa graduable
- **THEN** el desvío de nivel por propiedad se estima SOLO de los errores del forecast publicado previo sobre la ventana reciente de estancias (`window_days`), con `rel = Σ(fc−act)/Σ|act|` y `ts = Σerr/media|err|`
- **THEN** el peso de la corrección es suave y sin estado: `min(|ts|/ts_trigger, 1)` — nunca un interruptor que parpadee con el umbral
- **THEN** la corrección aplica shrinkage por tamaño de muestra (`n/(n+shrinkage)`), clip `±max_rel`, mínimo de observaciones (`min_obs`) y umbral de materialidad (0.5%)
- **THEN** la aplicación es multiplicativa al punto ANTES de la corrección por banda y del piso de reservas/techo de capacidad; el dataset publicado lleva `level_correction_factor` y el manifiesto el detalle por propiedad (n, rel, ts, peso, factor, razón) — ninguna corrección silenciosa
- **THEN** en modo verificación NO se aplica (el horizonte con reales no se corrige con información del propio horizonte); la razón queda en el manifiesto
- **THEN** con `enabled: false` el sistema no escribe la columna ni cambia ningún número
- **THEN** el default OFF quedó fijado por el A/B replay del 2026-11-05: la señal (errores de la publicación previa) solo observa leads cortos ya calibrados por el piso de reservas — cuando disparó, sobre-corrigió (cobertura FIESTA 90.8%→81.7%) y el drift real (leads 14-59) queda fuera de su alcance temporal; la maquinaria queda como knob a la espera de la variante de señal de publicación antigua
