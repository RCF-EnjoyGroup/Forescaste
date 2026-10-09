## Purpose

Define el monitoreo operacional del motor: alertas de drift, retraining configurable, almacenamiento de snapshots de forecast y reproducibilidad total de cada corrida.

## ADDED Requirements

### Requirement: Detección de drift
El sistema SHALL monitorear deterioro de MAPE y WMAPE, drift de Bias, drift de features, drift de la curva de pickup y drift de la distribución de demanda.

#### Scenario: Drift de métricas
- **WHEN** la precisión o el bias se deterioran significativamente respecto a la referencia histórica
- **THEN** el sistema genera una alerta identificando la métrica y la magnitud del deterioro

#### Scenario: Drift de curva de pickup
- **WHEN** la curva de pickup por lead se desplaza respecto de la histórica
- **THEN** el sistema reporta el cambio para que sea revisado antes del próximo retraining

### Requirement: Detección de outliers sin eliminación automática
El sistema SHALL detectar eventos, picos anormales, caídas abruptas, cierres, cambios de inventario, datos faltantes y cambios estructurales, clasificándolos sin eliminarlos automáticamente.

#### Scenario: Clasificación de picos
- **WHEN** se detecta un pico anormal de demanda
- **THEN** el sistema lo clasifica (error de datos, evento real, grupo, temporada, cambio estructural)
- **THEN** los eventos reales se conservan en el entrenamiento

### Requirement: Retraining configurable
El sistema SHALL soportar retraining con frecuencia configurable (`RETRAIN_FREQUENCY`), horizonte configurable (`FORECAST_HORIZON`) e histórico mínimo configurable (`MIN_HISTORY`), sin re-entrenar innecesariamente cuando no hay información nueva suficiente.

#### Scenario: Sin información nueva suficiente
- **WHEN** los datos nuevos no alcanzan el mínimo configurado
- **THEN** el sistema omite el retraining y registra el motivo

#### Scenario: Retraining programado
- **WHEN** se cumple la frecuencia configurada con datos nuevos suficientes
- **THEN** el pipeline re-entrena y valida que el desempeño no se haya degradado

### Requirement: Reproducibilidad de cada corrida
El sistema SHALL guardar por cada ejecución: versión del dataset, versión del modelo, periodo de entrenamiento, periodo de validación, configuración de features, hiperparámetros, métricas, timestamp y random seed.

#### Scenario: Corrida reproducible
- **WHEN** se solicita reproducir una corrida previa
- **THEN** con los metadatos almacenados se regeneran los mismos resultados
- **THEN** el forecast es determinístico dada la misma configuración y seed
