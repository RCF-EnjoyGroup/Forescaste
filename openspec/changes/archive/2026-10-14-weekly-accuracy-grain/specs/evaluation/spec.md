## MODIFIED Requirements

### Requirement: Precisión por propiedad
El sistema SHALL calcular MAPE, WMAPE, MAE, RMSE y Bias por cada propiedad, junto con ranking, peores y mejores propiedades, bias y volatilidad del error. El sistema SHALL además calcular el desglose al grano SEMANAL por propiedad (suma de forecast y reales por semana antes de medir) y reportar el contexto de escala de los reales (media y percentil 10) para leer el MAPE con honestidad — el MAPE diario con reales de una sola noche infla el error aparente de propiedades chicas.

#### Scenario: Tabla por propiedad
- **WHEN** se evalúa un forecast multi-propiedad
- **THEN** existe una fila por propiedad con las cinco métricas
- **THEN** el ranking de precisión se reporta ordenado
- **THEN** cada fila incluye la media y el percentil 10 de los reales evaluados

#### Scenario: Precisión al grano semanal
- **WHEN** se evalúa un forecast multi-propiedad
- **THEN** existe un desglose por propiedad con las métricas medidas sobre semanas completas (forecast y reales sumados por semana)
- **THEN** el grano semanal cancela el ruido de días con reales diminutos: una propiedad cuyo MAPE diario se dispare por reales de pocas noches muestra su precisión operativa real

#### Scenario: Advertencia de MAPE por denominador chico
- **WHEN** el percentil 10 de los reales diarios de una propiedad evaluada cae por debajo de 10 RN
- **THEN** el reporte marca su MAPE diario como advertencia de lectura (el criterio es estadístico del frame evaluado, no una lista fija de propiedades)
