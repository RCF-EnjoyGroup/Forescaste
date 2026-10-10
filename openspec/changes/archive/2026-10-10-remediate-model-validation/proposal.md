## Why

La validación de especialista sobre el motor de forecasting detectó una fractura central: **la corrida productiva siempre opera `pickup_curve` hardcodeado** (el peor candidato del bake-off según su propia evidencia: WMAPE 0.4452 vs 0.1234 de sarimax) mientras el benchmark declara un campeón (sarimax) que jamás genera el forecast. Además, el benchmark está corrupto (rollout ML con off-by-one y target dentro de las rolling features de entrenamiento), los lags son posicionales en vez de calendario (degradados silenciosamente por los huecos reales de LAPAS/LIRAK/LIREL), las bandas P10/P90 están mal calibradas y violan la capacidad, y varios controles prometidos (gate de calidad, suite leakage completa, backtest T-90…T-1, drift) no están cableados a la ruta productiva. Sin esta remediación, las métricas publicadas no son confiables y el sistema opera su peor modelo con la apariencia de un proceso de selección riguroso.

## What Changes

- **El campeón seleccionado pasa a operar el forecast**: la generación del forecast, el manifiesto y el dataset Power BI consumen el resultado real de `select_champion` (fin del hardcodeo de `pickup_curve`). La selección por propiedad (`select_per_property`) se evalúa con folds por propiedad y opera cuando hay evidencia suficiente.
- **Benchmark ML reparado**: se corrige el off-by-one del rollout recursivo (features as-of del día a predecir, no del anterior), las rolling features de entrenamiento excluyen el target (shift previo a la ventana), y los GBM reciben features de pickup/capacidad (`current_booked_rn` y derivadas).
- **Lags alineados a calendario**: los lags/rollings se computan sobre series reindexadas a frecuencia diaria por propiedad, de modo que `lag_7` sea "mismo día la semana pasada" y `lag_365` el mismo día del año anterior, incluso con huecos en las fuentes.
- **Folds con cobertura estacional**: los folds de selección se distribuyen a lo largo del año (incluyendo temporada alta dic–feb) y la ventana de selección no solapa la ventana de métricas publicada; se reporta separación explícita.
- **Bandas honestas**: los residuos para P10/P50/P90 se calculan por banda de lead, la banda se ancla al forecast publicado (post-cap), `p90 ≤ capacity` en todos los registros, y `interval_source` refleja el origen real de los residuos (corrige el default que siempre reporta `real_backtest`).
- **Controles cableados a producción**: el gate de calidad (`enforce`) y la suite de leakage completa (con capacidad en el perímetro as-of) se ejecutan y pueden detener la corrida productiva; un `fail` de calidad no publica forecast.
- **Entregables íntegros**: el backtest T-90…T-1 se ejecuta de verdad por propiedad y se persiste en `backtesting_report.json` (fin del stub); el desglose de accuracy por propiedad se persiste en el accuracy report y en Power BI; la cobertura de evaluación (propiedades dentro/fuera de la ventana) se reporta; las importancias de features dejan de estar hardcodeadas; los KPIs sin dato se serializan como `null` (no `NaN`).
- **Drift/monitoring activado**: la corrida productiva ejecuta detección de outliers y suite de drift, y registra la decisión de re-entrenamiento en el manifiesto.

## Capabilities

### New Capabilities

<!-- Ninguna: la remediación fortalece capacidades existentes, no introduce nuevas. -->

### Modified Capabilities

- `forecasting`: el campeón seleccionado SHALL operar el forecast (manifiesto, Power BI y tabla de salida coherentes con el benchmark); los intervalos SHALL anclarse al forecast publicado, calibrarse por banda de lead y respetar el techo de capacidad.
- `evaluation`: el benchmark SHALL evaluar candidatos íntegros (rollout sin off-by-one, features sin target, con señal de pickup) en folds distribuidos estacionalmente y separados de la ventana de reporte; la cobertura de evaluación SHALL reportarse explícitamente.
- `feature-engineering`: los lags y rollings SHALL alinearse a calendario (frecuencia diaria) por propiedad; las features faltantes SHALL reportarse con atribución por propiedad.
- `leakage-prevention`: la suite completa SHALL ejecutarse en la corrida productiva (incluida la capacidad en el perímetro as-of) y detener la corrida ante hallazgos.
- `data-ingestion`: el gate de calidad SHALL aplicarse en la ruta productiva: un status `fail` detiene la corrida antes de entrenar y no publica forecast.
- `backtesting`: el backtest T-90…T-1 SHALL ejecutarse por propiedad en la corrida productiva y persistirse con métricas en el entregable.
- `reporting`: los entregables SHALL declarar un único campeón coherente entre benchmark, manifiesto y Power BI; el desglose por propiedad SHALL persistirse; la explicabilidad SHALL derivarse del modelo real.
- `monitoring`: la corrida productiva SHALL ejecutar detección de outliers, suite de drift y decisión de re-entrenamiento, registrándolas en el manifiesto.

## Impact

- **Código**: `src/forecasting/runner.py` (selección operativa, bandas, manifiesto, entregables), `src/forecasting/run_real.py` (folds, rollout ML, benchmark), `src/features/demand.py` + `src/features/asof.py` (lags calendario), `src/backtesting/snapshots.py` (multi-propiedad, ejecución real), `src/monitoring/` (wiring), `src/reporting/` (coherencia de campeón, desglose por propiedad), `src/validation/` (gate cableado).
- **Comportamiento observable**: la tabla de forecast y el manifiesto pueden reportar un campeón distinto de `pickup_curve` (el modelo ganador del bake-off); las bandas cambian de forma (calibradas por lead, capadas a capacidad); `backtesting_report.json` deja de ser un stub.
- **Tests**: se actualizan los tests que asumían el hardcodeo (campeón pickup_curve, stub de backtest, residuo sintético) y se añaden tests para lags calendario, rollout corregido, gate productivo y coherencia de campeón.
- **Riesgos**: el WMAPE publicado puede cambiar al operar el campeón real (esperado y deseado); la corrida ahora puede fallar por gate de calidad o leakage (comportamiento especificado desde el inicio y hoy incumplido).
