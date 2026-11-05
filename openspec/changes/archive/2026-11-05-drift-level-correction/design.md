# Design — drift-level-correction

## Por qué el forecast publicado y no los folds

Los folds del backtest mezclan regímenes del año — su media no distingue
un bias permanente de un drift reciente (el caso FIESTA: consistencia 50%
entre estaciones, gate bloquea, y el A/B de celdas estacionales probó que
refinar la granularidad no resuelve un régimen nuevo). El forecast
PUBLICADO, en cambio, es el artefacto operativo: sus errores contra
reales materializados miden exactamente el desvío de nivel vigente, y
están disponibles as-of (publicación anterior, estancias ya vividas). Es
walk-forward puro: cada corrección usa solo información ≤ su corte.

## La señal y el peso suave

`vintage_detail(publicación_previa, reales, through=último real)` produce
el error por (propiedad, estancia) del forecast que el negocio vio.
Sobre la ventana reciente (`window_days=60` de fecha de estancia):

- `rel = Σ(fc − act) / Σ|act|` — magnitud y signo del desvío;
- `ts = Σerr / media|err|` — la misma tracking signal del monitoreo
  (umbral clásico |TS|≤4), acotada por tamaño de muestra por construcción.

**Peso suave** `w = min(|ts|/trigger, 1)` en lugar de histéresis con
estado: evita el parpadeo alrededor del umbral sin depender del orden de
las corridas ni del estado del registro (la debilidad que la revisión
externa ya señaló en `previous_run`). El control fino lo da el shrinkage
por tamaño de muestra, no un interruptor.

## Implementación

### `src/forecasting/core.py`

- `level_correction_factors(detail, *, trigger_ts, window_days, min_obs,
  shrinkage, max_rel) -> (factors, report)`: filtra la ventana, calcula
  (n, rel, ts) por propiedad, factor = `1 − rel · w · n/(n+shrinkage)`
  clipado a `1±max_rel`; `|factor−1| < 0.005` → 1.0 con razón. Devuelve
  el detalle completo (nada silencioso).
- `apply_level_factors(frame, factors)`: multiplicativo sobre el punto,
  escribe la columna `level_correction_factor` (patrón idéntico a
  `apply_bias_factors`).

### `src/forecasting/runner.py`

En `run_end_to_end`, justo después de `build_forecasts` (antes de la
corrección por banda y de las restricciones): si el knob está activo Y
`run_mode == "futuro"` Y hay publicación previa operativa → calcula los
factores con `vintage_detail(published_previa, dem, through=último real)`
y aplica. El reporte completo viaja al manifiesto (`level_correction`).
El dataset publicado hereda la columna (mismo patrón que
`bias_correction_factor`).

En modo verificación: sin corrección, con la razón documentada en el
manifiesto (`level_correction.enabled` + `mode` — la evaluación del
horizonte con reales debe permanecer sin contaminar).

### `config.yaml`

```yaml
monitoring:
  level_correction:
    enabled: <A/B>      # corrección de nivel por drift del publicado
    ts_trigger: 4.0     # |TS| que activa el peso completo
    window_days: 60     # ventana de estancias del error publicado
    min_obs: 10         # mínimo de estancias para estimar
    shrinkage: 10       # n/(n+shrinkage)
    max_rel: 0.20       # clip ±20%
```

## Validación

`scripts/ab_level_correction.py` — replay walk-forward sobre el registro:
pares (publicación anterior → publicación siguiente) en orden temporal;
factor estimado solo con errores ≤ corte de la siguiente; aplicado
(multiplicando punto y banda p10/p90 por f) al horizonte publicado;
puntuado contra reales materializados de la data vigente. Tabla por
propiedad: WMAPE, sesgo, cobertura baseline vs corregido; veredicto
contra los 4 criterios de la proposal.

Salvedades documentadas: (a) los reales de puntuación usan la data
vigente (revisada feb→ago-15) — la comparación es pareada, ambos lados
ven la misma verdad; (b) la cobertura se aproxima escalando la banda
publicada por f (en producción la banda se recalibra alrededor del punto
corregido).

## Riesgos

- **Doble corrección** con la capa por banda (fold-based): ambas son
  multiplicativas; la de banda es de año completo y la de nivel reciente —
  si ambas disparan, componen. El A/B mide el efecto conjunto; si el
  compuesto sobre-corrigiera, el clip y el peso suave lo acotan.
- **Corrección que persigue ruido** en hoteles chicos: min_obs + shrinkage
  + TS acotada por muestra lo mitigan; el replay lo mide por propiedad.
- **Retrocompatibilidad**: knob OFF → cero cambios (factor ausente,
  columna no escrita, manifiesto con `enabled: false`).
