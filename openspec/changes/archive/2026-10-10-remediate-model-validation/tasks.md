# Tasks — remediate-model-validation

## 1. Features íntegras (D3/D4)

- [x] 1.1 `src/features/asof.py`: `rolling_backward` aplica `shift(1)` antes de la ventana (excluye la propia fila); docstring documenta la semántica "solo días previos". Verificar: tests existentes de rollings actualizados y `python -m pytest tests/test_features.py -q` en verde.
- [x] 1.2 `src/features/demand.py`: reindexar cada grupo a frecuencia diaria antes de lags/rollings y devolver al índice original; gate de `lag_365`/`same_month_year` por span calendario (`max-min+1 ≥ 365`); `missing_features` pasa a dict `{feature: [propiedades]}`. Verificar: test nuevo con huecos donde `lag_7` de LAPAS-like data = valor de 7 días naturales atrás (no 7 filas), y test de atribución por propiedad.
- [x] 1.3 `src/forecasting/run_real.py`: corregir `_recursive_ml_predict` para construir la fila de features del día a predecir (no `tail(1)` del anterior); reproducir el caso lineal del hallazgo (lightgbm ya no predice plano). Verificar: test nuevo de rollout con rampa lineal dentro de tolerancia.
- [x] 1.4 `src/evaluation/metrics.py` (o leakage): `check_target_not_in_features(frame, target_col, feature_cols)` que recalcula rollings con shift(1) y falla si una feature coincide con la versión que incluye el target. Verificar: test unitario con feature contaminada → `LeakageError`; suite completa en verde.

## 2. Benchmark por propiedad con folds estacionales (D2/D5)

- [x] 2.1 `src/forecasting/run_real.py`: `real_fold_metrics` evalúa por propiedad Y portafolio (columna `property` en los fold_metrics; `PORTFOLIO` para el agregado); folds con cortes trimestrales `cutoff−360/−270/−180/−90` días, ventana de 30d, sin solape con la ventana de reporte. Verificar: test de que ningún fold solapa `(cutoff, cutoff+horizon]` y de que fold_metrics tiene 9 grupos (8 props + portafolio).
- [x] 2.2 `src/forecasting/run_real.py`: `_ml_frame` añade `current_booked_rn` (snapshot as-of por stay del panel pickup) a las features de entrenamiento y de predicción; grid del horizonte usa `_booked_requests`. Verificar: test de que `current_booked_rn` está en el frame y es as-of (snapshot ≤ corte del fold).
- [x] 2.3 `src/evaluation/metrics.py`: `diebold_mariano(loss_a, loss_b, horizon)` con corrección HAC h-pasos; `src/reporting/benchmark.py`: reporta significancia campeón vs subcampeón y declara empate estadístico en `reason` cuando p > 0.05 (desempate por parsimonia). Verificar: test de DM con series idénticas → no significativo; benchmark md incluye la declaración.
- [x] 2.4 Diagnóstico del pickup_curve de portafolio: corregir `_portfolio_pickup_forecast` o alinearlo con el pipeline por propiedad (grano y método del modelo desplegado); si queda fuera del tablero por diseño, registrar el motivo en el benchmark. Verificar: WMAPE del pickup_curve de portafolio consistente (mismo orden) con el por-propiedad; test de no-regresión.

## 3. Campeón operativo (D1)

- [x] 3.1 `src/forecasting/runner.py`: dispatcher `build_forecasts(model_name, ...)` con interfaz común para los 7 candidatos (pickup_curve actual; seasonal_naive; ets/sarimax por propiedad; GBMs por propiedad con features as-of); `run_end_to_end` corre el benchmark ANTES del forecast y pronostica con `benchmark.champion`; con `fold_metrics=None` usa `pickup_curve` + etiqueta `synthetic_placeholder`. Verificar: test de que el forecast usa el campeón inyectado (folds sintéticos con ganador ≠ pickup_curve → tabla con `model_name` = ganador).
- [x] 3.2 Selección por propiedad operativa: `run_end_to_end` invoca `select_per_property` con los fold_metrics por propiedad; cada propiedad se pronostica con su ganador si tiene evidencia suficiente, si no con el campeón global; fallback registrada en el manifiesto (`per_property_models`, `property_fallback`). Verificar: test con folds por propiedad donde dos propiedades ganan con modelos distintos → tabla final tiene `model_name` distinto por propiedad.
- [x] 3.3 Coherencia de campeón en entregables: `run_manifest` (`model_version`, `hyperparameters.model`) y `powerbi/model.csv` (7 filas + `is_champion` real) derivados del benchmark. Verificar: test de que manifiesto, `powerbi/model.csv` y `model_benchmark.md` declaran el mismo campeón.

## 4. Controles con dientes en producción (D6)

- [x] 4.1 `src/forecasting/runner.py`: `gate.enforce(dq_report)` tras `build_report` — status `fail` lanza `CriticalDataError` antes de features/entrenamiento. Verificar: test de corrida e2e con fuente inválida (RN negativos) → excepción y ningún entregable escrito.
- [x] 4.2 `src/forecasting/runner.py`: suite de leakage con `capacity` en `frames` (capacidad recortada al corte), `train`/`full`/`feature_fn` reales (check de features con dataset completo) y un rolling representativo (check rolling con futuro); split validado `{train_end: cutoff, val_start: cutoff+1}`; `check_target_not_in_features` sobre la matriz ML; manifiesto declara `capacity_usage: "horizon_cap_only"`. Verificar: test de inyección de capacidad futura en features → `LeakageError`; corrida real en verde.
- [x] 4.3 `src/features/capacity.py` + runner: features de capacidad dependientes de reservas se calculan tras el merge de `current_booked_rn` (reordenar pipeline) o su ausencia queda en `missing_features`. Verificar: corrida e2e reporta `capacity_remaining`/`capacity_utilization` presentes o documentadas.

## 5. Backtest real por propiedad + bandas honestas (D8/D7)

- [x] 5.1 Backtest multi-propiedad: ejecución por propiedad del framework `run_backtest` (la dimensión propiedad se conserva sin que unas actuals pisen a otras, sin ruptura de API). Verificar: tests de backtesting en verde + test nuevo de 2 propiedades con actuals distintos en la misma fecha.
- [x] 5.2 `src/forecasting/runner.py` + `run_real.py`: backtest muestreado (30 stay dates × leads 90/60/30/14/7/3/1 × propiedades) con el campeón congelando pickup as-of; resultado persistido en `backtesting_report.json` (global, por lead band, por propiedad). Verificar: corrida real produce reporte con métricas ≠ stub; test de formato.
- [x] 5.3 `src/forecasting/constraints.py`: `build_intervals` ancla `p50` al punto publicado (`final_forecast_rn`), `p10/p90` con `clip(0, capacity)`, y soporta residuos por banda de lead (dict banda → residuos, fallback global con `min_residuals=20`); `interval_source` veraz (default sintético → `synthetic_placeholder`). Verificar: tests de intervals actualizados (p50 == final en 100%, p90 ≤ capacity en 100%) + test de banda por lead con residuos crecientes.
- [x] 5.4 `src/forecasting/run_real.py`: residuos por banda de lead provienen del backtest de 5.2 (reemplaza residuo T-30 de ventana única); en modo verificación usa la ventana anterior al corte. Verificar: manifiesto de corrida real con `interval_source: real_backtest` y bandas de amplitud creciente por lead.

## 6. Monitoring + entregables íntegros (D9/D10)

- [x] 6.1 `src/forecasting/runner.py`: al final de la corrida, `detect_outliers` + `run_drift_suite` + `should_retrain`; manifiesto registra `monitoring: {outliers, drift_alerts, retraining_decision}`. Verificar: test de que el manifiesto e2e contiene la sección monitoring; corrida real la muestra.
- [x] 6.2 `forecast_accuracy_report.json`: `{global, by_property, by_lead_band, evaluation_coverage}` con propiedades excluidas y motivo; `powerbi/accuracy.csv` concatena desglose por lead band + por propiedad (columna `breakdown`). Verificar: test de cobertura (propiedad sin reales en ventana queda listada) + test de columnas de accuracy.csv.
- [x] 6.3 KPIs serializables: `_jsonable` convierte `NaN/inf → None` en manifiesto (JSON válido estricto). Verificar: `json.loads` del manifiesto de corrida real sin `NaN` literal; test unitario.
- [x] 6.4 Explicabilidad real: `_explain_forecast` deriva drivers de los valores de la propia fila (booked/pickup/capacidad) sin pesos fijos; SHAP solo si el campeón es ML (y con guard de costo). Verificar: test de que el reporte cambia cuando cambian los drivers de fila (no valores constantes).

## 7. Integración y verificación final

- [x] 7.1 Suite completa en verde: `python -m pytest -q` (197+ tests, incl. los actualizados que fijaban campeón hardcodeado, stub de backtest y residuo sintético). Verificar: 0 fallos.
- [x] 7.2 Corrida de verificación real: `python -m src.forecasting.run_real` — sale 0, campeón coherente en los 4 canales (benchmark, manifiesto, tabla, Power BI), `backtesting_report.json` con métricas reales, bandas `p90 ≤ capacity`, coverage reportada. Verificar: inspección de los entregables del run.
- [x] 7.3 Corrida futura real: `python -m src.forecasting.run_real --futuro` — sale 0, `metrics_source: injected_backtest`, coherencia de campeón, KPIs sin `NaN` literal. Verificar: inspección del manifiesto y entregables.
