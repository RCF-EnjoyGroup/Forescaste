# Statistical Modeling Specification

## Purpose

Implement and evaluate classical statistical time series models for hotel revenue forecasting including Prophet, SARIMA/SARIMAX, and Exponential Smoothing (Holt-Winters).

## Requirements

### Requirement: Prophet Model Implementation
The system SHALL implement Facebook Prophet for room-nights forecasting with hotel-specific configurations.

#### Scenario: Prophet model training
- **WHEN** Prophet model is trained on training data
- **THEN** the system configures yearly, weekly, and daily seasonality appropriately for hotel data
- **THEN** the system adds country-specific holidays (Costa Rica) as regressors
- **THEN** the system uses appropriate changepoint prior scale for room-nights trends

#### Scenario: Prophet hyperparameter tuning
- **WHEN** hyperparameter tuning is enabled
- **THEN** the system tunes: changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale
- **THEN** the system uses cross-validation on temporal splits for tuning
- **THEN** the system selects best parameters by validation MAE

#### Scenario: Prophet prediction
- **WHEN** trained Prophet model predicts on validation/test data
- **THEN** the system generates point forecasts and prediction intervals (80%, 95%)
- **THEN** the system returns predictions aligned with test timestamps

### Requirement: SARIMA/SARIMAX Model Implementation
The system SHALL implement SARIMA and SARIMAX models for room-nights forecasting.

#### Scenario: SARIMA model selection
- **WHEN** SARIMA model is configured
- **THEN** the system uses auto_arima or grid search to find optimal (p,d,q)(P,D,Q,m) parameters
- **THEN** the system considers seasonal period m=7 (weekly) and m=365 (yearly) for daily data
- **THEN** the system validates model residuals are white noise (Ljung-Box test)

#### Scenario: SARIMAX with exogenous variables
- **WHEN** exogenous features are available (holidays, events, weather)
- **THEN** the system fits SARIMAX with exogenous regressors
- **THEN** the system forecasts exogenous variables for future prediction horizon

#### Scenario: SARIMA prediction
- **WHEN** trained SARIMA/SARIMAX predicts on validation/test data
- **THEN** the system generates point forecasts and prediction intervals
- **THEN** the system handles multi-step forecasting recursively with date-aligned predictions

### Requirement: Exponential Smoothing (Holt-Winters) Implementation
The system SHALL implement Exponential Smoothing models (Holt-Winters) for room-nights forecasting.

#### Scenario: ETS model configuration
- **WHEN** Exponential Smoothing is configured
- **THEN** the system supports additive and multiplicative trend/seasonal combinations
- **THEN** the system uses automatic model selection (AIC/BIC) or manual configuration
- **THEN** the system handles seasonality period appropriate for data frequency

#### Scenario: Holt-Winters prediction
- **WHEN** trained ETS model predicts on validation/test data
- **THEN** the system generates point forecasts and prediction intervals with date alignment
- **THEN** the system supports both additive and multiplicative seasonality

### Requirement: Statistical Model Evaluation
The system SHALL evaluate all statistical models using standard time series metrics.

#### Scenario: Metric computation
- **WHEN** statistical models are evaluated
- **THEN** the system computes: MAE, RMSE, MAPE, sMAPE, MASE, WAPE
- **THEN** the system computes metrics on validation and test sets separately
- **THEN** the system reports prediction interval coverage (for models providing intervals)

#### Scenario: Residual diagnostics
- **WHEN** statistical models are evaluated
- **THEN** the system performs residual analysis: ACF of residuals, normality test, heteroscedasticity test
- **THEN** the system flags models with problematic residuals
