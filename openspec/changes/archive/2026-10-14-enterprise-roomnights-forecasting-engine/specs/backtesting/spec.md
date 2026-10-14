## Purpose

Define el framework de backtesting que simula snapshots históricos (T-90 … T-1) y el tracking de revisiones del forecast contra el Actual.

## ADDED Requirements

### Requirement: Backtesting de snapshots históricos
El sistema SHALL construir un framework de backtesting que, para una fecha de estancia futura, simule cómo habría sido el forecast cuando faltaban 90, 60, 30, 14, 7, 3 y 1 día.

#### Scenario: Simulación por lead
- **WHEN** se ejecuta el backtest para una fecha de estancia
- **THEN** se generan forecasts en cada corte de lead configurado usando solo la información disponible en ese corte
- **THEN** cada forecast se compara con el Room Nights final observado

#### Scenario: Integridad del backtest
- **WHEN** se reconstruye un snapshot histórico
- **THEN** ningún dato posterior al corte del snapshot participa en features ni en targets de ese corte

### Requirement: Tracking de revisiones del forecast
El sistema SHALL almacenar snapshots de forecast para comparar `Forecast T-90` vs `T-60` vs `T-30` vs `T-14` vs `T-7` vs `Actual`.

#### Scenario: Serie de revisiones
- **WHEN** se generan forecasts en múltiples anticipos para la misma fecha de estancia
- **THEN** cada snapshot queda persistido con su fecha de forecast (forecast_date) y lead_days
- **THEN** la evolución de la expectativa hacia el Actual es reconstruible

#### Scenario: Cambio de expectativa
- **WHEN** la expectativa entre dos revisiones cambia significativamente
- **THEN** el sistema reporta la variación para que Revenue Management pueda interpretarla

### Requirement: Evaluación honesta del backtest
El sistema SHALL reportar las mismas métricas (MAPE, WMAPE, MAE, RMSE, Bias) sobre los resultados del backtest, incluyendo el desglose por lead time.

#### Scenario: Métricas del backtest
- **WHEN** termina el backtesting
- **THEN** se publican las métricas globales y por banda de lead calculadas sobre los snapshots simulados
