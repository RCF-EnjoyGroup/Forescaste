# Design — seasonal-calibration-cells

## La dimensión faltante y por qué es honesta

La temporada del stay es **conocible a-priori**: depende solo del calendario
(stay_date), no de datos que hijack del futuro. Un horizonte que se publica
hoy con stays en ago-oct sabía desde el diseño que aba ago (temporada baja)
y oct (shoulder). Usarla como dimensión de calibración no introduce
lookahead — a diferencia de usar los reales del horizonte jamás disponibles.

Definición compartida con el desglose de `metrics_by_seasonality` ya
publicado: 12/1/2 → high, 3/4/5/10/11 → shoulder, 6-9 → low.

## Arquitectura de la solución

### 1. `src/evaluation/breakdowns.py` (nuevo helper compartido)

`season_of_stay(dates) -> pd.Series[str]` — refactor del mapeo inline de
`metrics_by_seasonality` (comportamiento idéntico, fuente única).

### 2. `src/forecasting/core.py`

**`lead_bias_factors(..., seasonal_cells: bool = False)`**:

- Con `True`, DENTRO de cada celda (prop, banda) además se estima el sesgo
  por (prop, banda, temporada): misma fórmula
  (`rel = Σ(fc − act) / Σ|act|`), mismos gates por celda estacional
  (min_obs, consistencia de signo entre folds, shrinkage, clip).
- **Fallback del factor por fila del horizonte**: la celda estacional
  recomienda un factor; el factor FINAL por (prop, banda, temporada) es:
  celda estacional aplicable → estacional; si no → el factor de
  (prop, banda) si aplicable; si no → 1.0. El reporte registra por celda:
  factor estacional, factor de banda y la fuente aplicada
  (`season / band / none`) — nada silencioso.
- `apply_bias_factors` pasa a operar con llaves
  `(prop, banda, temporada)` mapeadas por fila (la temporada de cada
  stay_date del horizonte).

**`operative_residual_bands(..., seasonal_cells: bool = False)`**:

- Con `True`, las llaves del dict de residuos pasan de (prop, banda) a
  (prop, banda, temporada). El residuo por fila usa el factor FUENTE
  aplicable a ESA fila (mismo criterio de fallback: la banda calibra el
  pipeline completo que se publica).
- Ponderación de recencia por fold se mantiene idéntica.

**`build_intervals(..., residual_bands con llaves de 3-tupla)`**:

- `prop_aware` hoy requiere llaves 2-tupla; se extiende: 3-tupla →
  modo temporada (la temporada por fila deriva del stay_date con
  `season_of_stay`).
- Ladder nuevo: celda (prop, banda, temporada) ≥ min_residuals →
  pool (prop, banda) ≥ min → pool propiedad → pool banda → global.
  Cada fallback agregado a `notes` (documentado, no silencioso).
- Contratos intactos: cuantiles completos (−q90, −q10), proyección
  p10 ≤ p50 ≤ p90, recorte a [0, capacity], pesos de recencia.

### 3. `src/forecasting/runner.py`

Lee `bias_cfg["seasonal_cells"]` una vez y se lo pasa a
`lead_bias_factors` y `operative_residual_bands`. El manifiesto expone el
knob y el reporte extendido (celda × temporada con fuente aplicada).

### 4. `config.yaml`

```yaml
bias_correction:
  min_obs: 20
  shrinkage: 30
  max_rel: 0.30
  seasonal_cells: <valor del A/B>   # celdas por temporada del stay
```

### 5. Validación A/B (`scripts/ab_seasonal_cells.py` + corridas reales)

- Knob OFF ya existe: baseline = `real-20261030-1147` (data/seed
  idénticos).
- Knob ON: re-corrida completa del pipeline real con el knob activo.
- Comparación sobre el horizonte con reales: cobertura por propiedad y
  celda, WMAPE/bias por propiedad, ancho de banda medio.
- Script reproducible que imprime la tabla comparativa A/B y el veredicto
  contra los criterios de aceptación de la proposal.

## Riesgos y mitigaciones

- **Celdas pequeñas estacionales** (`n < 20`): la escalera cae a
  (prop, banda) — el comportamiento actual — sin costo ni ganancia.
  Con 4 folds × 90d, una celda estacional de una banda de lead de 30 días
  acumula ~30-120 residuos por año: las celdas de FIESTA (volumen alto)
  sí forman celda propia; las propiedades pequeñas caen a fallback SIEMPRE
  declarado.
- **Sobre-ajuste de régimen**: la consistencia de signo entre folds DENTRO
  de la celda estacional es el mismo antivirus de siempre — si el sesgo
  gira incluso dentro de la temporada, no corrige.
- **Retrocompatibilidad**: con knob off, llaves 2-tupla, códigos y
  bandas bit a bit idénticos (test de regresión lo clava).
