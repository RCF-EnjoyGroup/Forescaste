## Purpose

Generate future revenue forecasts using the best selected model and provide actionable insights for revenue management.

## ADDED Requirements

### Requirement: Future Forecast Generation
The system SHALL generate forecasts for a specified future horizon using the best model.

#### Scenario: Forecast horizon configuration
- **WHEN** future forecasting is requested
- **THEN** the system accepts configurable forecast horizon (e.g., 30, 60, 90 days)
- **THEN** the system uses the best model selected from evaluation phase
- **THEN** the system generates point forecasts and prediction intervals

#### Scenario: Multi-hotel forecasting
- **WHEN** data contains multiple hotels
- **THEN** the system generates forecasts per hotel (if model supports it)
- **THEN** the system provides aggregated portfolio forecast
- **THEN** the system identifies top/bottom performing hotels in forecast

#### Scenario: Forecast with exogenous variables
- **WHEN** best model requires exogenous variables (SARIMAX, Prophet with regressors)
- **THEN** the system generates or obtains future values for exogenous variables
- **THEN** the system documents assumptions for exogenous variable projections

### Requirement: Forecast Visualization and Reporting
The system SHALL create publication-ready forecast visualizations and summary reports.

#### Scenario: Forecast plot generation
- **WHEN** forecast is generated
- **THEN** the system plots historical data (last N periods) + forecast with intervals
- **THEN** the system annotates key dates (holidays, events) on forecast plot
- **THEN** the system provides separate plots per hotel and aggregated view

#### Scenario: Forecast summary report
- **WHEN** forecast is generated
- **THEN** the system produces summary with: total forecasted revenue, growth vs last period, confidence intervals
- **THEN** the system highlights key drivers and assumptions
- **THEN** the system flags periods of high uncertainty

### Requirement: Model Retraining and Updating
The system SHALL support model retraining as new data becomes available.

#### Scenario: Incremental model update
- **WHEN** new data is available
- **THEN** the system supports retraining best model with expanded dataset
- **THEN** the system validates retrained model performance hasn't degraded
- **THEN** the system versions models and tracks performance over time

### Requirement: Assumptions and Limitations Documentation
The system SHALL document all forecasting assumptions and limitations.

#### Scenario: Assumptions documentation
- **WHEN** forecast is generated
- **THEN** the system documents: data granularity, forecast horizon, model assumptions, data quality caveats
- **THEN** the system documents: seasonality assumptions, trend assumptions, external factor assumptions
- **THEN** the system documents: limitations (no external data, no weather, no competitor data, etc.)

#### Scenario: Improvement recommendations
- **WHEN** forecast is generated
- **THEN** the system recommends potential improvements: external data sources, higher frequency data, hierarchical forecasting, scenario analysis