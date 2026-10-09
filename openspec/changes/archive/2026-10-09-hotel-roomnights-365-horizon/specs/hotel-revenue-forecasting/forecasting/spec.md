## MODIFIED Requirements

### Requirement: Future Forecast Generation
The system SHALL generate room-nights forecasts for a **365-day (12-month) horizon with daily detail** per property and for the portfolio, combining realized history with the current booking books, reported in room-night units and respecting inventory capacity.

#### Scenario: Forecast horizon configuration
- **WHEN** future forecasting is requested
- **THEN** the system generates a 365-day detailed forecast (configurable via `roomnights_real.horizon_days`) using the champion/selected model
- **THEN** the system generates point forecasts and conformal uncertainty bands in room-night units for every one of the 365 horizon steps

#### Scenario: Books-informed forecast inputs
- **WHEN** the future forecast is generated for dates present in the booking books
- **THEN** the system uses the current on-the-books values at lead buckets >= 365 days as the only forward-known whole-horizon inputs (buckets with lead < 365 are lookahead for dates beyond their lead window)
- **THEN** the as-of cutoff is reported so no snapshot after the forecast origin is used

#### Scenario: Books coverage by as-of-safe window segment
- **WHEN** the 365-day forecast is reported
- **THEN** the system reports booking-book coverage per segment using the finest as-of-safe bucket for each window: days 1-90 vs books at lead 90, days 91-180 vs books at lead 180, days 181-365 vs books at lead 365
- **THEN** each lead bucket is used only in the segment where it is forward-known at the forecast origin

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
- **THEN** the system forecasts it through the honest guard (recent-level anchoring) for the full horizon
- **THEN** the property is flagged as guard/cold-start in the report with an explicit confidence caveat that a 12-month extrapolation from a short history is low-confidence

#### Scenario: Forecast with exogenous variables
- **WHEN** best model requires exogenous variables (SARIMAX, Prophet with regressors)
- **THEN** the system generates or obtains future values for exogenous variables
- **THEN** the system documents assumptions for exogenous variable projections
