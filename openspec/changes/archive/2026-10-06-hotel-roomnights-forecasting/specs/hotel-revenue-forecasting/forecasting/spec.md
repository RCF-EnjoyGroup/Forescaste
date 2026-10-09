## MODIFIED Requirements

### Requirement: Future Forecast Generation
The system SHALL generate room-nights forecasts for a specified future horizon using the best selected model, reported in room-night units and respecting physical capacity.

#### Scenario: Forecast horizon configuration
- **WHEN** future forecasting is requested
- **THEN** the system accepts configurable forecast horizon (e.g., 30, 60, 90 days)
- **THEN** the system uses the best model selected from evaluation phase
- **THEN** the system generates point forecasts and prediction intervals in room-night units

#### Scenario: Capacity ceiling
- **WHEN** any forecasted daily room nights exceed the available capacity (available_rooms; portfolio level = sum across hotels)
- **THEN** the system caps the forecast at the available capacity
- **THEN** the system reports the dates and magnitude of the capping

#### Scenario: Implied occupancy reporting
- **WHEN** the future forecast is generated
- **THEN** the system reports implied occupancy (forecasted room nights / available rooms) alongside the forecast
- **THEN** the system flags implied occupancies that are implausible (e.g., above 100% before capping) as a model or data sanity warning

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
The system SHALL create publication-ready room-nights forecast visualizations and summary reports.

#### Scenario: Forecast plot generation
- **WHEN** forecast is generated
- **THEN** the system plots historical data (last N periods) + forecast with intervals in room-night units
- **THEN** the system annotates key dates (holidays, events) on forecast plot
- **THEN** the system provides separate plots per hotel and aggregated view

#### Scenario: Forecast summary report
- **WHEN** forecast is generated
- **THEN** the system produces summary with: total forecasted room nights, average implied occupancy, and per-period breakdown
- **THEN** the system highlights key drivers and assumptions
- **THEN** the system flags periods of high uncertainty
