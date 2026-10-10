# Design — remediate-model-validation

## Context

La validación de especialista encontró que la arquitectura de modelado tiene una fractura central: el runner productivo genera el forecast con `PickupCurveModel` hardcodeado (`runner.py:364`) mientras el benchmark (`run_benchmark` → `select_champion`) declara un campeón que nunca opera; `powerbi/model.csv` publica un tercer campeón hardcodeado (`runner.py:477`). Además, el benchmark evalúa a los 3 GBM con un rollout recursivo con off-by-one (`run_real.py:119`) y con rolling features que incluyen el target de la propia fila; los lags son posicionales (`asof.py:103` `shift`) y los huecos reales del CSV (LAPAS 345, LIRAK 212, LIREL 93 fechas) desplazan su semántica calendario; el gate de calidad y los checks avanzados de leakage no están cableados a la ruta productiva; las bandas se anclan al raw pre-cap con un shift global de residuos T-30; y los entregables backtesting/drift son stubs. Ver `proposal.md` para el detalle de motivación.

## Goals / Non-Goals

**Goals**
- El campeón del bake-off opera el forecast: un único campeón coherente en benchmark, manifiesto, tabla de salida y Power BI.
- Benchmark íntegro: rollout ML con features as-of del paso correcto, features de entrenamiento sin el target de la fila, señal de pickup para GBM, y candidato pickup_curve evaluado con el mismo grano que producción.
- Lags/rollings con semántica calendario bajo huecos.
- Folds de selección distribuidos estacionalmente y separados de la ventana de reporte; selección por propiedad con folds por propiedad y fallback declarado.
- Controles con dientes en la ruta productiva: gate de calidad `fail` detiene la corrida; suite de leakage con capacidad en el perímetro y checks de features activos; verificación de target en la matriz ML.
- Entregables íntegros: backtest real por propiedad, desglose de accuracy por propiedad persistido, cobertura de evaluación, KPIs serializables, explicabilidad derivada del modelo real.
- Bandas honestas: calibradas por banda de lead, ancladas al punto publicado, capadas a capacidad, con origen veraz.

**Non-Goals**
- Tuning exhaustivo de hiperparámetros (los candidatos siguen con parámetros fijos declarados; el tuning es un cambio futuro).
- Reconciliación top-down del portafolio (bottom-up ya la cierra en 0; queda como estaba).
- Rediseño del check de huecos de calidad de datos (el ruido de `missing_dates` en pickup/capacidad futura se documenta, no se rediseña en este cambio).
- Cambios en el contrato SQL de exportación (`README_SISTEMAS.md`).

## Decisions

### D1 — El campeón opera: dispatcher de modelos en el runner
`run_end_to_end` recibe `fold_metrics` reales (inyectados por `run_real.py`), corre `run_benchmark` ANTES de generar el forecast, y usa el campeón resultante vía un **dispatcher** `build_forecasts(model_name, ...)` en `runner.py` que soporta los 7 candidatos con una interfaz común (`fit(...)` → `predict(requests) -> DataFrame` con `forecast_rn`):
- `pickup_curve`: pipeline actual (curvas de pickup restante + cold start).
- `seasonal_naive`: `SeasonalNaive` sobre historia por propiedad.
- `ets` / `sarimax`: un modelo por propiedad entrenado con su serie diaria completa as-of.
- GBMs: un modelo por propiedad sobre features as-of (D3).
Alternativa descartada: dejar `pickup_curve` fijo y solo documentar — perpetúa la incoherencia que motiva el cambio. El manifiesto y `powerbi/model.csv` se llenan con `benchmark.champion` / `model_name` reales. Si `fold_metrics` es `None` (tests de muestra), el dispatcher usa `pickup_curve` y el manifiesto etiqueta `benchmark_source: synthetic_placeholder`, manteniendo compatibilidad con la suite existente.

### D2 — Benchmark per-property con selector de grano
`real_fold_metrics` se reestructura: para cada propiedad y el portafolio, evalúa los 7 candidatos en los mismos folds. Los `fold_metrics` ganan columna `property` (`PORTFOLIO` para el agregado), lo que activa `select_per_property` (hoy muerto por falta de la columna). **El campeón global se decide con la evidencia por propiedad** (media de WMAPEs por propiedad, cada una con el mismo peso): usar la fila agregada comete el efecto Simpson — el WMAPE del portafolio refleja el error de las propiedades grandes y oculta el de las chicas (hallazgo #7 del diagnóstico). La fila `PORTFOLIO` queda para el resumen de consola y como panel de errores para la prueba Diebold-Mariano (significancia a nivel de sistema). `by_property` alimenta el dispatcher: cada propiedad se pronostica con su ganador si tiene ≥ `MIN_FOLDS_PER_PROPERTY` folds, si no con el campeón global (fallback registrado en el manifiesto). El benchmark MD mantiene la tabla portafolio + campeón global.
Alternativa descartada: evaluar solo portafolio y elegir campeón único — el forecast es por propiedad y el efecto Simpson del agregado oculta el error real.

### D3 — Rollout ML corregido y features sin target
- **Off-by-one**: en `_recursive_ml_predict`, la fila de features se construye para el día `d` a predecir (`data` + fila candidata de `d` con `room_nights=NaN`), no `tail(1)` del día anterior. Con `dropna(subset=["lag_1"])` la fila candidata ya existe en el frame de features; su predicción reemplaza el NaN.
- **Target en rollings**: `rolling_backward` pasa a llamar a `series.shift(1).rolling(...)` (ventana `[i-w, i-1]`), excluyendo la propia fila. `lag()` ya es seguro por definición. Los rollings del reporte de features cambian de semántica de "incluye hoy" a "solo días previos" — documentado en docstring.
- **Señal de pickup**: `ML_FEATURES` añade `current_booked_rn` (on-the-books as-of del día de estancia, desde el panel de pickup) — la feature más predictiva del dominio. Para el frame de benchmark por propiedad, el grid del horizonte se une con el snapshot as-of (mismo `_booked_requests`), y la historia de entrenamiento une el on-the-books del snapshot más reciente por stay (todo con `snapshot_date ≤ cutoff` del fold).
- **Verificación**: nueva función `check_target_not_in_features(frame, target, features)`: para cada rolling feature, recalcula con shift(1) y exige coincidencia; se invoca dentro de `_ml_frame` y falla la corrida si la matriz contiene el target (spec leakage: features→matriz de entrenamiento).

### D4 — Lags calendario vía reindexación diaria
`add_demand_features` reindexa cada grupo de propiedad a frecuencia diaria (`reindex(date_range)` con `room_nights=NaN` en huecos) antes de aplicar lags/rollings, y devuelve el frame al índice original (los huecos re-obtienen NaN en features, sin cambiar el conteo de filas del dataset). `lag_7` vuelve a ser "mismo día la semana pasada" y `lag_365` el mismo día del año anterior. El gate de `LAG_365_MIN_DAYS` pasa de contar filas a medir el **span calendario** (`max-min+1 ≥ 365`). `same_month_year` se mantiene como lag 365 calendario (semántica ahora real); `missing_features` pasa de lista global a dict `{feature: [propiedades afectadas]}` con preservación del formato plano en el manifiesto (`"lag_365": ["SJOSL"]`).
Alternativa descartada: lags aritméticos por fecha sin reindexar (`shift` sobre DatetimeIndex) — pandas lo soporta, pero los rollings seguirían contando filas en vez de días; la reindexación resuelve ambos uniformemente y permite `min_periods` sobre días reales.

### D5 — Folds estacionales y separación selección/reporte
Los 4 folds × 30d se anclan en cortes `cutoff − 360d`, `−270d`, `−180d`, `−90d` (trimestrales), de modo que (a) cubren estaciones distintas del año (incluida temporada alta con histórico suficiente), (b) el último fold termina 90 días antes del corte → **cero solape** con la ventana de métricas publicada `(cutoff, cutoff+horizon]`, y (c) entrenamientos expanding estándar. Se añade `diebold_mariano(errors_a, errors_b, horizon)` en `evaluation/metrics.py` (pérdida absoluta, corrección de autocorrelación de h pasos, aproximación normal): el benchmark reporta si el campeón supera al subcampeón con significancia; si no (p > 0.05), el `reason` declara "empate estadístico" y el desempate usa parsimonia (baseline > candidato complejo).
Alternativa descartada: folds 90d completos — multiplica ×3 el costo del benchmark por propiedad para evidencia marginal; 30d por fold trimestral cubre 4 estaciones con costo razonado.

### D6 — Controles cableados a producción
- **Gate**: `run_end_to_end` llama `gate.enforce(dq_report)` después de `build_report`; `CriticalDataError` detiene la corrida antes de features/entrenamiento (status `fail`).
- **Leakage suite completa**: se pasa `capacity` (recortada al corte para features/entrenamiento) en `frames`; los checks `check_feature_whole_dataset` y `check_rolling_with_future` se invocan con `train=history`, `full=dem`, `feature_fn` de features de demanda y un rolling representativo. La capacidad post-corte se declara como exógena del techo (`capacity_usage: "horizon_cap_only"` en el manifiesto).
- **Split real**: el split validado pasa a ser `{train_end: cutoff, val_start: cutoff+1}` — la validación real es el horizonte del forecast.

### D7 — Bandas honestas
`build_intervals` cambia de firma semántica: `point_col="final_forecast_rn"` (ancla al punto publicado → `p50 = final` por construcción), y los residuos se calculan **por banda de lead** (`residuals_by_lead_band`: dict banda → cuantiles) desde el backtest por propiedad con cortes por lead (D8), con fallback al pool global por escasez de datos (`min_residuals=20`). Post-calibración: `p10 = clip(point + q10, 0, capacity)`, `p90 = clip(point + q90, 0, capacity)`. `interval_source` se decide en el punto de inyección (solo `real_backtest` cuando el caller pasó residuos reales; el default sintético se marca `synthetic_placeholder`).

### D8 — Backtest real por propiedad en la corrida productiva
`run_backtest` (snapshots.py) se extiende con dimensión propiedad: `BacktestRevision.property`, actuals indexados por `(property, stay_date)`, y métricas por propiedad. La corrida productiva ejecuta un backtest muestreado: N stay dates recientes (default 30) × leads (90/60/30/14/7/3/1) × propiedades, con `forecast_fn` del campeón por propiedad congelando pickup as-of — sobre el modelo desplegado, no una reimplementación. El resultado alimenta `backtesting_report.json` (métricas globales, por lead band y por propiedad) Y los residuos por lead band de D7, reemplazando el residuo T-30 de ventana única.

### D9 — Drift y outliers en la corrida
Al final de `run_end_to_end`: `detect_outliers(history)` + `run_drift_suite` (compara ventana reciente vs referencia histórica por propiedad, drift de curva de pickup) + `should_retrain` → `monitoring` en el manifiesto con `retraining_decision` (`retrain`/`skip` + motivo). Sin alertas activas externas: el manifiesto es el canal.

### D10 — Entregables íntegros
- `forecast_accuracy_report.json`: `{global, by_property, by_lead_band, evaluation_coverage: {evaluated_properties, excluded_properties: {prop: motivo}, n_obs, portfolio_properties}}`.
- `powerbi/accuracy.csv`: concat de desglose por lead band + por propiedad con columna `breakdown`.
- `powerbi/model.csv`: filas de los 7 candidatos del benchmark + `is_champion` real.
- KPIs: `_jsonable(kpis)` convierte `NaN/inf → None` antes de serializar el manifiesto.
- Explicabilidad: si el campeón es ML, `explain_with_shap` (limitado por costo); si es pickup_curve, drivers reales derivados de `current_booked_rn`/`expected_pickup_rn`/capacidad con pesos relativos calculados de los propios valores (no 0.55/0.30/0.15 fijos); si es estadístico, contribución dominante de nivel/estacionalidad declarada.
- `backtesting_report.json`: dict de D8 (no stub).

## Risks / Trade-offs

- [El campeón puede cambiar de corrida a corrida → series de forecast no comparables entre corridas] → el manifiesto registra campeón+benchmarks de la corrida; el KPI `forecast_vs_previous` documenta que compara modelos distintos cuando aplica.
- [Costo: benchmark por propiedad × 4 folds ≈ 8× hoy] → folds de 30d con SARIMAX/ETS por propiedad acotados por `maxiter`; si el runtime explota, `--quick` reduce propiedades evaluadas (solo top por volumen) — documentado, no bloqueante.
- [Rollings sin el propio valor cambian resultados de features existentes] → tests de features se actualizan a la semántica nueva; el cambio es intencional (el valor previo filtraba el target).
- [Bandas por lead band con pocas observaciones por banda] → fallback a pool global con `min_residuals`; la banda ancha (90+) hereda del pool si no hay evidencia.
- [Gate estricto puede detener corridas que antes pasaban] → es el comportamiento especificado desde el diseño original; `run_real.py` lo reporta con código ≠ 0 y mensaje claro.

## Migration Plan

1. Implementar en orden: D3/D4 (features) → D5 (folds) → D2 (benchmark por propiedad) → D1 (dispatcher campeón) → D6 (controles) → D8/D7 (backtest/bandas) → D9/D10 (monitoring/entregables).
2. Actualizar los tests que fijan el comportamiento viejo (campeón hardcodeado, stub de backtest, residuo sintético → `real_backtest`).
3. Correr `python -m pytest -q` (suite completa) y `python -m src.forecasting.run_real` (verificación) + `--futuro` antes de dar por cerrado el cambio.
4. Rollback: `git revert` del merge; sin migración de datos (los artefactos de corridas viejas conservan su formato).
