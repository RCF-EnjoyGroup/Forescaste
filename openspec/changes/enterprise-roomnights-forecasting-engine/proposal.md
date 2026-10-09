## Why

Enjoy Group necesita un motor profesional de forecasting de Room Nights para Revenue Management y planificación corporativa que responda diariamente: "¿Cuántas Room Nights esperamos cerrar por propiedad y fecha, por qué lo creemos, qué tan confiable es el forecast y cómo cambia conforme entra pickup?". El proyecto se reinicia desde cero (el workspace quedó vacío y así se decidió), por lo que este cambio define desde la raíz el comportamiento exigido por el documento maestro *Enterprise Room Nights Forecasting Engine*: demanda primero, capacidad como restricción física, pickup como señal fundamental, cero leakage y evaluación temporal honesta — más las capas que faltaban: intervalos de incertidumbre, snapshots de revisión, drift, explainability, dataset para Power BI y KPIs ejecutivos.

## What Changes

- Se crea una base nueva (`openspec/` recién inicializado, `src/` modular, sin notebooks monolíticos) que implementa el motor completo sobre los tres archivos reales: `capacity_daily`, `demand_daily`, `pickup_by_lead`.
- **Target**: `Final Room Nights` a granularidad `Property × Stay Date`, con inspección automática de esquemas (sin asumir nombres de columnas).
- **Demand-first**: la demanda es el objetivo; Revenue queda fuera de esta versión (capa económica futura: `RN × ADR`).
- **Capacidad como restricción**: `Final = min(Raw, Capacity)` conservando `raw_forecast`, `capacity` y `final_forecast` por separado, con alerta de violación antes del cap.
- **Pickup**: curvas no lineales por lead time / propiedad / estacionalidad (`Current Bookings → Expected Pickup → Final RN`).
- **Champion/challenger** con 7 candidatos: Seasonal Naive (baseline obligatorio), Pickup Curve (baseline obligatorio), ETS, SARIMAX, LightGBM, XGBoost, CatBoost; selección de campeón por evidencia out-of-sample y selección distinta por propiedad.
- **Validación exclusivamente temporal**: expanding window / rolling origin + evaluación multi-step con datos congelados por snapshot; prohibido el split aleatorio.
- **Métricas completas**: MAPE (manejo explícito de `Actual = 0`), WMAPE (métrica corporativa principal), MAE, RMSE, Bias; desgloses por lead time, propiedad y estacionalidad.
- **Backtesting** de snapshots históricos (T-90 … T-1) y **revisión de forecast** (T-90 vs T-60 vs T-30 vs … vs Actual).
- **Intervalos** P10/P50/P90 siempre que el modelo lo permita.
- **Explainability** por fecha/propiedad (SHAP / importancia / contribuciones comprensibles para Revenue Management).
- **Salida estructurada** de forecast + **dataset listo para Power BI** + **KPIs ejecutivos**.
- **Monitoring**: drift de MAPE/WMAPE/bias, drift de curva de pickup y de distribución de demanda, con alertas; retraining configurable; reproducibilidad completa (dataset version, model version, seed, métricas, timestamp).
- **Data Quality Report** previo al entrenamiento con detención del pipeline ante problemas críticos; **pruebas automáticas de leakage** que fallan el pipeline si detectan lookahead.
- **Estándares de ingeniería**: estructura `src/` modular (data, validation, features, forecasting, models, evaluation, backtesting, explainability, monitoring, reporting), configurable, testeable; el notebook queda solo como interfaz de análisis.

## Capabilities

### New Capabilities
- `data-ingestion`: lectura e inspección de `capacity_daily`, `demand_daily`, `pickup_by_lead` (esquema detectado, no asumido), validación de calidad de datos y reporte de calidad con detención ante errores críticos.
- `leakage-prevention`: garantía de no-leakage: información disponible solo hasta la fecha de corte, features as-of seguras y pruebas automáticas que fallan el pipeline.
- `feature-engineering`: features hoteleras (calendario, demanda histórica con lags/rolling, capacidad, pickup con aceleración y ventanas) respetando orden temporal.
- `forecasting`: estrategia de modelos champion/challenger con los 7 candidatos, jerarquía propiedad/portafolio reconciliada, horizonte configurable, restricción de capacidad, ocupación implícita e intervalos P10/P50/P90.
- `evaluation`: validación rolling/expanding, evaluación multi-step por snapshot, métricas MAPE/WMAPE/MAE/RMSE/Bias con manejo de Actual=0, desgloses por lead time, propiedad y estacionalidad, y selección de campeón.
- `backtesting`: simulación de snapshots históricos (T-90 … T-1) y tracking de revisiones del forecast contra el Actual.
- `explainability`: explicación del forecast por fecha/propiedad con contribuciones de drivers, para modelos ML (SHAP/importancia) legibles para Revenue Managers.
- `reporting`: salida estructurada de forecast, dataset listo para Power BI, KPIs ejecutivos y los 10 entregables del proyecto.
- `monitoring`: drift de métricas/curvas/distribución con alertas, retraining configurable, snapshots de forecast y reproducibilidad de cada corrida.

### Modified Capabilities
<!-- Ninguno: OpenSpec se inicializó vacío en este reinicio; todas las capacidades son nuevas. -->

## Impact

- **Código nuevo**: `src/` completo (data, validation, features, forecasting, models, evaluation, backtesting, explainability, monitoring, reporting) + `tests/`; sin dependencia de notebooks para la lógica.
- **Configuración**: `FORECAST_HORIZON` (30/60/90+), `RETRAIN_FREQUENCY`, `MIN_HISTORY`, seeds y hiperparámetros centralizados y versionados.
- **Datos**: tres CSV/agregados reales como única fuente; el pipeline debe poder re-ejecutarse con datos actualizados sin tocar código.
- **Consumidores**: Revenue Management (forecast diario + explicación), gerencia (KPIs), Power BI (dataset), y el proceso de re-entrenamiento/monitoring.
- **Riesgo**: al ser un reinicio desde cero no hay compatibilidad que mantener; el riesgo principal es la validez temporal de las features (mitigado por las pruebas de leakage).
