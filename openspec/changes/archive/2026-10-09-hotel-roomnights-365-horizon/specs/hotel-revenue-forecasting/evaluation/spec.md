## MODIFIED Requirements

### Requirement: Calibrated Uncertainty Bands (Split Conformal per Horizon)
The system SHALL accompany point forecasts with empirical uncertainty bands built from forecast residuals, with one calibrated width per horizon step for **every step of the full forecast horizon**.

#### Scenario: Conformal bands on the future forecast
- **WHEN** the final future forecast is generated for the configured horizon (365 days)
- **THEN** the system builds per-horizon-step bands from rolling-origin backtest residuals computed on windows as long as the full horizon (365-day windows) at the requested confidence level
- **THEN** the system reports the empirical coverage of the bands on held-out data alongside the nominal level
- **THEN** the final report displays total forecast with its lower/upper bounds

#### Scenario: Coverage validation when the hold-out is shorter than the horizon
- **WHEN** the evaluation hold-out window (e.g., 213 days) is shorter than the forecast horizon (365 days)
- **THEN** the system validates empirical coverage on the available overlap (the first N hold-out steps), slicing band calibration to that overlap for the validation
- **THEN** the report discloses that late-horizon steps are calibrated from fold residuals but not directly validated on hold-out data

#### Scenario: Bottom-up portfolio bands
- **WHEN** the operational portfolio forecast is the bottom-up sum of per-property forecasts
- **THEN** the portfolio bands are built from each property's own fold residuals (jointly sampled fold-by-fold, with short-history residual pools where applicable), composition-correct by construction
- **THEN** the empirical coverage of the bottom-up bands is reported alongside the top-down bands' coverage
