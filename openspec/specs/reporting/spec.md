# reporting Specification

## Purpose

Define la salida estructurada del forecast, el dataset listo para Power BI, los KPIs ejecutivos y los entregables del proyecto.

## Requirements

### Requirement: Tabla final de forecast
El sistema SHALL generar una tabla de salida con al menos: `property`, `stay_date`, `forecast_date`, `lead_days`, `current_booked_rn`, `capacity`, `raw_forecast_rn`, `final_forecast_rn`, `p10_forecast`, `p50_forecast`, `p90_forecast`, `model_name`, `model_version` y `forecast_run_timestamp`.

#### Scenario: Columnas obligatorias
- **WHEN** se ejecuta el forecast
- **THEN** la tabla de salida contiene todas las columnas obligatorias con valores poblados

#### Scenario: Columnas extendidas
- **WHEN** la información lo permite
- **THEN** la salida agrega `historical_same_day_rn`, `expected_pickup_rn`, `pickup_rate`, `forecast_occupancy`, `forecast_bias` y `confidence_level`


### Requirement: Dataset listo para Power BI
El sistema SHALL producir un dataset preparado para Power BI que permita analizar forecast (Forecast RN, Actual RN, Variance, Variance %, Occupancy, Capacity), accuracy (MAPE, WMAPE, MAE, RMSE, Bias), pickup (Current RN, Expected Pickup, Pickup %, Lead Time) y modelo (Champion, Challenger, Model version); el desglose de accuracy SHALL incluir la dimensión propiedad y el área de modelo SHALL ser coherente con el benchmark de la corrida.

#### Scenario: Cargas del dashboard
- **WHEN** el dataset se carga en Power BI
- **THEN** cada una de las áreas (forecast, accuracy, pickup, modelo) es analizable con sus campos

#### Scenario: Variance
- **WHEN** existen forecast y Actual para la misma propiedad/fecha
- **THEN** el dataset expone la diferencia absoluta y porcentual entre ambos

#### Scenario: Accuracy por propiedad en Power BI
- **WHEN** la corrida evalúa múltiples propiedades
- **THEN** el dataset Power BI incluye el desglose de accuracy por propiedad además del desglose por banda de lead

#### Scenario: Coherencia del campeón en el dataset
- **WHEN** el área de modelo del dataset Power BI declara el campeón
- **THEN** coincide con el campeón del benchmark y del manifiesto de la misma corrida


### Requirement: KPIs ejecutivos
El sistema SHALL producir como mínimo: Forecast Room Nights, Forecast Occupancy, Forecast Accuracy (WMAPE y MAPE), Forecast Bias, Expected Pickup, Capacity Utilization y Forecast vs Previous Forecast; un KPI sin dato disponible SHALL serializarse como `null`.

#### Scenario: Tablero de KPIs
- **WHEN** se ejecuta una corrida de forecast
- **THEN** los siete KPIs están calculados y disponibles para reporte

#### Scenario: KPI sin dato disponible
- **WHEN** un KPI no tiene datos suficientes para calcularse (por ejemplo, no existe una corrida previa comparable)
- **THEN** se serializa como `null` con el motivo documentado, nunca como `NaN`


### Requirement: Reportes del proyecto
El sistema SHALL producir los entregables: Data Quality Report, EDA Report, Feature Engineering Report, Model Benchmark (Seasonal Naive, Pickup Curve, ETS, SARIMAX, LightGBM, XGBoost, CatBoost), Backtesting Report, Forecast Accuracy Report, Future Forecast Dataset, Model Explainability Report, dataset para Power BI y documentación técnica; cada entregable SHALL derivarse de los cálculos reales de la corrida.

#### Scenario: Benchmark completo
- **WHEN** se ejecuta el Model Benchmark
- **THEN** la tabla de resultados incluye una fila por cada uno de los siete candidatos con MAPE, WMAPE, MAE, RMSE, Bias y estabilidad
- **THEN** el campeón se declara con evidencia out-of-sample
- **WHEN** el benchmark proviene de datos sintéticos de reemplazo
- **THEN** el entregable lo etiqueta explícitamente

#### Scenario: Documentación de limitaciones
- **WHEN** se publica la documentación técnica
- **THEN** las limitaciones del modelo quedan documentadas explícitamente

#### Scenario: Accuracy Report completo
- **WHEN** se publica el Forecast Accuracy Report
- **THEN** incluye las métricas globales, el desglose por propiedad y la cobertura de evaluación (propiedades dentro y fuera de la ventana)

#### Scenario: Explicabilidad derivada del modelo real
- **WHEN** se publica el Model Explainability Report
- **THEN** las importancias se derivan del modelo que generó el forecast de la corrida o del método sustituto que aplique, nunca de valores fijos predefinidos
- **WHEN** el modelo campeón no produce importancias nativas
- **THEN** el reporte declara el método sustituto y su limitación

#### Scenario: Backtesting Report con contenido real
- **WHEN** se produce el Backtesting Report de una corrida productiva
- **THEN** contiene las métricas del backtest ejecutado, no una referencia a las pruebas del framework


### Requirement: Estándares de ingeniería
El sistema SHALL organizar el código de forma modular bajo `src/` (data, validation, features, forecasting, models, evaluation, backtesting, explainability, monitoring, reporting), reproducible, configurable y testeable; la lógica no MAY vivir en un único notebook.

#### Scenario: Notebook como interfaz
- **WHEN** se usa un notebook
- **THEN** este actúa como interfaz de análisis y presentación, delegando la lógica en `src/`

#### Scenario: Re-ejecución con datos actualizados
- **WHEN** llegan datos nuevos en las tres fuentes
- **THEN** el sistema se re-ejecuta con los datos actualizados sin modificar código manualmente


### Requirement: Vista HTML por propiedad
El sistema SHALL incluir en `report.html` el comportamiento por propiedad: forecast de cada propiedad con su banda de incertidumbre, techo de capacidad y reales cuando existan; accuracy por propiedad con su modelo operante; y las revisiones del backtest T-90..T-1 desglosadas por propiedad y por banda de lead. Las secciones SHALL ser defensivas ante datos ausentes y jamás romper la corrida.

#### Scenario: Forecast por propiedad
- **WHEN** la corrida produce forecast para varias propiedades
- **THEN** el reporte muestra una gráfica por propiedad con su P50, banda P10/P90 y techo de capacidad
- **THEN** el título de cada panel declara el modelo que opera esa propiedad
- **THEN** los reales se superponen cuando la ventana tiene observados

#### Scenario: Accuracy por propiedad
- **WHEN** la evaluación calcula el desglose por propiedad
- **THEN** el reporte muestra el WMAPE por propiedad y una tabla con propiedad, modelo, WMAPE, MAPE, bias y n_obs

#### Scenario: Revisiones del backtest por propiedad
- **WHEN** el backtest T-90..T-1 produce revisiones
- **THEN** el reporte muestra el WMAPE de las revisiones por propiedad y la convergencia del error por banda de lead (una serie por propiedad)
- **THEN** el `backtesting_report.json` incluye el desglose `by_property_lead_band`

#### Scenario: Datos ausentes
- **WHEN** falta alguno de los desgloses o el backtest está vacío
- **THEN** las secciones correspondientes se omiten con un mensaje y la corrida no falla
