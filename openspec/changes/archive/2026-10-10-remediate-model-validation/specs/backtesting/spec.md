## MODIFIED Requirements

### Requirement: Backtesting de snapshots históricos
El sistema SHALL construir un framework de backtesting que, para una fecha de estancia futura, simule cómo habría sido el forecast cuando faltaban 90, 60, 30, 14, 7, 3 y 1 día; el backtest SHALL soportar múltiples propiedades y ejecutarse en la corrida productiva.

#### Scenario: Simulación por lead
- **WHEN** se ejecuta el backtest para una fecha de estancia
- **THEN** se generan forecasts en cada corte de lead configurado usando solo la información disponible en ese corte
- **THEN** cada forecast se compara con el Room Nights final observado

#### Scenario: Integridad del backtest
- **WHEN** se reconstruye un snapshot histórico
- **THEN** ningún dato posterior al corte del snapshot participa en features ni en targets de ese corte

#### Scenario: Backtest multi-propiedad
- **WHEN** el portafolio tiene varias propiedades
- **THEN** el backtest se ejecuta por propiedad y las comparaciones contra reales conservan la dimensión propiedad sin que unas pisen a otras

#### Scenario: Backtest en la corrida productiva
- **WHEN** se ejecuta una corrida productiva con reales observados disponibles
- **THEN** el backtest se ejecuta sobre una muestra representativa de fechas de estancia con cortes por lead
- **THEN** sus resultados se persisten en el Backtesting Report de la corrida

### Requirement: Evaluación honesta del backtest
El sistema SHALL reportar las mismas métricas (MAPE, WMAPE, MAE, RMSE, Bias) sobre los resultados del backtest, incluyendo el desglose por lead time y por propiedad.

#### Scenario: Métricas del backtest
- **WHEN** termina el backtesting
- **THEN** se publican las métricas globales y por banda de lead calculadas sobre los snapshots simulados

#### Scenario: Métricas por propiedad
- **WHEN** el backtest cubre varias propiedades
- **THEN** las métricas se reportan también por propiedad
