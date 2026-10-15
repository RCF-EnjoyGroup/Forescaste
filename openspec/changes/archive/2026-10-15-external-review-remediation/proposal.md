## Why

Una revisión externa de ciencia de datos (segundo modelo de IA, 2026-10-15)
auditoró el motor contra el código y los entregables publicados y entregó 8
hallazgos priorizados. Cada afirmación se validó de forma independiente contra
`src/` y contra los datos reales antes de aceptarla; seis se confirmaron como
defectos reales y se remediaron, dos se diferenciaron con evidencia.

**Hallazgo crítico confirmado** — `per_property_backtest` (modo `--futuro`)
instancia `PickupCurveModel` hardcodeado: las métricas publicadas de la corrida
futura describían un modelo que NO era el operante (p. ej. en
`futuro-20261010-1652` MARINA operaba `xgboost` pero sus métricas — MAPE 24.3%,
bias −24.6% — eran de la curva de pickup). La garantía "el campeón opera" se
cumplía en el forecast pero no en la evaluación.

**Verificación numérica independiente de los demás hallazgos:**

| # | Afirmación | Verificado con | Veredicto |
|---|---|---|---|
| 1 | Backtest futuro no mide el campeón | `run_real.py:292` (hardcode) | Defecto — remediar |
| 2 | ML sin `lead_days` | `ML_FEATURES` (core.py:42): el panel la tiene pero no llega a la matriz | Defecto — remediar |
| 3 | Bins planos a leads cortos | Datos reales: pickup restante FIESTA lead1≈0 vs lead9≈+21 comparten bin `[1,10.9)` con `n_bins=10` | Defecto — remediar |
| 4 | Bias medido nunca corregido | What-if citado es in-sample (mejora la métrica que reporta) | Diferir — requiere diseño as-of |
| 5 | Rollout recursivo GBM parcheado | `predict_candidate` realimenta yhat como lags; blend lo mitiga | Diferir — cambio de investigación; `lead_days` ataca parcialmente la causa |
| 6 | Selección por propiedad inestable | FIESTA lightgbm↔seasonal_naive, MARINA lightgbm↔xgboost entre corridas de Oct-10/14; sin test de significancia por propiedad | Defecto — remediar con margen |
| 7 | seasonal_naive de 1 observación | `baselines.py:48` (`iloc[-1]`) | Defecto — remediar |
| 8 | FIESTA sub-cubre banda (65.6%) | Recomputado: global 78.3% con CORIN (66.7%, invisible para la revisión por el alias), FIESTA 65.6% | Defecto — publicar cobertura (la recalibración fina queda como trabajo futuro) |

## What Changes

- **El campeón opera TAMBIÉN la evaluación futura**: `per_property_backtest`
  se reescribe sobre `build_forecasts` (pipeline exacto de producción:
  fit/predict/blend) con `model_by_property` seleccionado **as-of** al inicio
  de la ventana de backtest (benchmark repetido con corte `cutoff − horizon`).
  Las métricas del modo futuro miden el pipeline completo (selección +
  forecast) sin ventaja retrospectiva; aplican piso de reservas y techo de
  capacidad — la métrica evalúa el forecast publicado. El manifiesto declara
  `eval_model_by_property`.
- **`lead_days` como feature ML de primera clase**: exógena (aritmética de
  calendario), sin fuga en panel (lead del snapshot) ni en horizonte
  (`stay − cutoff`). Los GBM dejan de distinguir leads solo a través del
  booked ruidoso y pueden aprender la forma de la curva de pickup.
- **Bordes de bins explícitos y densos** (`pickup_curve.bin_edges` en
  `config.yaml`): `[1, 3, 7, 14, 21, 30, 45, 60, 90]` con catch-all. A/B en la
  ventana de verificación: banda 1-2 bias **+1.85% → −0.01%**, WMAPE
  **3.1% → 1.8%**; global 12.79% → 12.73%.
- **Margen de materialidad del override por propiedad (5%)**: un ganador
  local solo desplaza al campeón global si lo supera por > 5% relativo del
  composite EN ESA propiedad; de lo contrario se retiene al global con el
  fallback documentando la retención. Mismo espíritu que
  `BLEND_ACCURACY_MARGIN`.
- **Baseline exigente**: `seasonal_naive` usa la media de las últimas 4
  ocurrencias del mismo DOW (parametrizable `dow_window`).
- **Cobertura empírica de la banda P10–P90** en `forecast_accuracy_report.json`
  (global, por propiedad, por banda de lead) cuando la ventana evaluada tiene
  reales; sin banda sobre la ventana se documenta el porqué.

## Capabilities

### Modified Capabilities
- `forecasting`: MODIFIED — La evaluación del modo futuro opera el campeón seleccionado as-of con las restricciones del forecast publicado; margen de materialidad del override por propiedad; bordes de bins explícitos en la curva de pickup; baseline estacional suavizado.
- `evaluation`: MODIFIED — Cobertura empírica de la banda P10–P90 como desglose de evaluación; el lead de la estancia es feature exógena de los candidatos ML.
- `reporting`: MODIFIED — El accuracy report publica `interval_coverage`; el manifiesto publica `eval_model_by_property`.

## Impact

- **Código**: `run_real.py` (per_property_backtest + flujo futuro), `runner.py` (eval_models, cobertura, doc técnica), `core.py` (lead feature), `baselines.py` (dow_window, bin_edges), `selection.py` (margen), `breakdowns.py` (interval_coverage), `config.py`/`config.yaml` (bin_edges), 9 tests nuevos.
- **Comportamiento observable**: las métricas del modo `--futuro` describen los modelos que operan; banda 1-2 sin sesgo de cierre; accuracy report con `interval_coverage`; el benchmark tiene un baseline más fuerte y ML más competitivos (catboost WMAPE 0.240 → 0.223).
- **Costo**: el modo futuro corre el benchmark dos veces (selección operativa + selección as-of de la evaluación): ~7 min → ~11 min.
- **Riesgo**: la selección por propiedad cambia (candidatos cambiaron: FIESTA→seasonal_naive, MARINA→catboost); el campeón global pickup_curve es estable y su empate DM con catboost se mantiene.
