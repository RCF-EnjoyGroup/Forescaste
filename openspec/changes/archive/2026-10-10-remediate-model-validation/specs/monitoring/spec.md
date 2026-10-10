## MODIFIED Requirements

### Requirement: Detección de drift
El sistema SHALL monitorear deterioro de MAPE y WMAPE, drift de Bias, drift de features, drift de la curva de pickup y drift de la distribución de demanda; el monitoreo SHALL ejecutarse como parte de la corrida productiva y sus resultados quedar registrados en el manifiesto.

#### Scenario: Drift de métricas
- **WHEN** la precisión o el bias se deterioran significativamente respecto a la referencia histórica
- **THEN** el sistema genera una alerta identificando la métrica y la magnitud del deterioro

#### Scenario: Drift de curva de pickup
- **WHEN** la curva de pickup por lead se desplaza respecto de la histórica
- **THEN** el sistema reporta el cambio para que sea revisado antes del próximo retraining

#### Scenario: Monitoreo en la corrida productiva
- **WHEN** termina una corrida productiva end-to-end
- **THEN** la detección de outliers y la suite de drift se ejecutaron sobre los datos de la corrida
- **THEN** sus resultados y la decisión de re-entrenamiento quedan registrados en el manifiesto
