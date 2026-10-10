## MODIFIED Requirements

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
