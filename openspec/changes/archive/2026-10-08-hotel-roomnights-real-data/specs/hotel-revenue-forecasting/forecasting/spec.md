## MODIFIED Requirements

### Requirement: Future Forecast Generation
The system SHALL generate room-nights forecasts for a specified future horizon per property and for the portfolio, combining realized history with the current booking books, reported in room-night units and respecting inventory capacity.

#### Scenario: Forecast horizon configuration
- **WHEN** future forecasting is requested
- **THEN** the system accepts configurable forecast horizon (e.g., 90 days) and uses the best model selected from evaluation
- **THEN** the system generates point forecasts and conformal uncertainty bands in room-night units

#### Scenario: Books-informed forecast inputs
- **WHEN** the future forecast is generated for dates present in the booking books
- **THEN** the system uses the current on-the-books values at lead buckets >= the horizon (as-of-safe) as forward-known inputs
- **THEN** the as-of cutoff is reported so no snapshot after the forecast origin is used

#### Scenario: Capacity ceiling
- **WHEN** any forecasted daily room nights exceed the inventory capacity (per property; portfolio = sum across properties)
- **THEN** the system caps the forecast at the available capacity and reports the capped dates and magnitudes

#### Scenario: Implied occupancy reporting
- **WHEN** the future forecast is generated
- **THEN** the system reports implied occupancy (forecasted room nights / official inventory) and flags implausible values

#### Scenario: Multi-hotel forecasting
- **WHEN** data contains multiple properties
- **THEN** the system generates forecasts per property and an aggregated portfolio forecast
- **THEN** the sum of per-property forecasts is compared with the direct portfolio forecast and the gap is reported

#### Scenario: New-hotel cold start
- **WHEN** a property has no (or minimal) realized history (e.g., opened recently)
- **THEN** the system forecasts it through the global model using pickup, capacity and calendar signals
- **THEN** the property is flagged as cold-start in the report with an explicit confidence caveat

#### Scenario: Forecast with exogenous variables
- **WHEN** best model requires exogenous variables (SARIMAX, Prophet with regressors)
- **THEN** the system generates or obtains future values for exogenous variables
- **THEN** the system documents assumptions for exogenous variable projections

### Requirement: Forecast Visualization and Reporting
The system SHALL create publication-ready per-property room-nights forecast visualizations and summary reports, with capacity, occupancy and booking-book context.

#### Scenario: Forecast plot generation
- **WHEN** forecast is generated
- **THEN** the system plots historical realized demand + forecast with uncertainty bands in room-night units, alongside the capacity line
- **THEN** the system provides separate plots per property and an aggregated view

#### Scenario: Forecast summary report
- **WHEN** forecast is generated
- **THEN** the system produces a summary with total forecasted room nights per property, implied occupancy, current books coverage (share of the forecast already on the books), and uncertainty flags
- **THEN** cold-start properties and any capacity fallbacks are called out explicitly
