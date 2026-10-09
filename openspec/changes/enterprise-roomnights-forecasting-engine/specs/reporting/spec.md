## Purpose

Define la salida estructurada del forecast, el dataset listo para Power BI, los KPIs ejecutivos y los entregables del proyecto.

## ADDED Requirements

### Requirement: Tabla final de forecast
El sistema SHALL generar una tabla de salida con al menos: `property`, `stay_date`, `forecast_date`, `lead_days`, `current_booked_rn`, `capacity`, `raw_forecast_rn`, `final_forecast_rn`, `p10_forecast`, `p50_forecast`, `p90_forecast`, `model_name`, `model_version` y `forecast_run_timestamp`.

#### Scenario: Columnas obligatorias
- **WHEN** se ejecuta el forecast
- **THEN** la tabla de salida contiene todas las columnas obligatorias con valores poblados

#### Scenario: Columnas extendidas
- **WHEN** la información lo permite
- **THEN** la salida agrega `historical_same_day_rn`, `expected_pickup_rn`, `pickup_rate`, `forecast_occupancy`, `forecast_bias` y `confidence_level`

### Requirement: Dataset listo para Power BI
El sistema SHALL producir un dataset preparado para Power BI que permita analizar forecast (Forecast RN, Actual RN, Variance, Variance %, Occupancy, Capacity), accuracy (MAPE, WMAPE, MAE, RMSE, Bias), pickup (Current RN, Expected Pickup, Pickup %, Lead Time) y modelo (Champion, Challenger, Model version).

#### Scenario: Cargas del dashboard
- **WHEN** el dataset se carga en Power BI
- **THEN** cada una de las áreas (forecast, accuracy, pickup, modelo) es analizable con sus campos

#### Scenario: Variance
- **WHEN** existen forecast y Actual para la misma propiedad/fecha
- **THEN** el dataset expone la diferencia absoluta y porcentual entre ambos

### Requirement: KPIs ejecutivos
El sistema SHALL producir como mínimo: Forecast Room Nights, Forecast Occupancy, Forecast Accuracy (WMAPE y MAPE), Forecast Bias, Expected Pickup, Capacity Utilization y Forecast vs Previous Forecast.

#### Scenario: Tablero de KPIs
- **WHEN** se ejecuta una corrida de forecast
- **THEN** los siete KPIs están calculados y disponibles para reporte

### Requirement: Reportes del proyecto
El sistema SHALL producir los entregables: Data Quality Report, EDA Report, Feature Engineering Report, Model Benchmark (Seasonal Naive, Pickup Curve, ETS, SARIMAX, LightGBM, XGBoost, CatBoost), Backtesting Report, Forecast Accuracy Report, Future Forecast Dataset, Model Explainability Report, dataset para Power BI y documentación técnica.

#### Scenario: Benchmark completo
- **WHEN** se ejecuta el Model Benchmark
- **THEN** la tabla de resultados incluye una fila por cada uno de los siete candidatos con MAPE, WMAPE, MAE, RMSE, Bias y estabilidad
- **THEN** el campeón se declara con evidencia out-of-sample

#### Scenario: Documentación de limitaciones
- **WHEN** se publica la documentación técnica
- **THEN** las limitaciones del modelo quedan documentadas explícitamente

### Requirement: Estándares de ingeniería
El sistema SHALL organizar el código de forma modular bajo `src/` (data, validation, features, forecasting, models, evaluation, backtesting, explainability, monitoring, reporting), reproducible, configurable y testeable; la lógica no MAY vivir en un único notebook.

#### Scenario: Notebook como interfaz
- **WHEN** se usa un notebook
- **THEN** este actúa como interfaz de análisis y presentación, delegando la lógica en `src/`

#### Scenario: Re-ejecución con datos actualizados
- **WHEN** llegan datos nuevos en las tres fuentes
- **THEN** el sistema se re-ejecuta con los datos actualizados sin modificar código manualmente
