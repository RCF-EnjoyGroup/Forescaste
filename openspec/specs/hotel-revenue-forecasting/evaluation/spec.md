# Evaluation Specification

## Purpose

Provide comprehensive model evaluation and comparison framework for hotel revenue forecasting with standardized metrics, statistical significance testing, and visualization.

## Requirements

### Requirement: Standard Time Series Metrics Computation
The system SHALL compute all required time series forecasting metrics for model comparison.

#### Scenario: Core metrics calculation
- **WHEN** evaluation is run on predictions vs actuals
- **THEN** the system computes MAE (Mean Absolute Error)
- **THEN** the system computes RMSE (Root Mean Squared Error)
- **THEN** the system computes MAPE (Mean Absolute Percentage Error)
- **THEN** the system computes sMAPE (Symmetric Mean Absolute Percentage Error)
- **THEN** the system computes MASE (Mean Absolute Scaled Error) using naive seasonal baseline
- **THEN** the system computes WAPE (Weighted Absolute Percentage Error)

#### Scenario: Metric robustness
- **WHEN** computing metrics with zero or near-zero actual values
- **THEN** the system handles division by zero gracefully (e.g., epsilon for MAPE)
- **THEN** the system reports which metrics are unreliable for given data

### Requirement: Multi-Horizon Evaluation
The system SHALL evaluate models at multiple forecast horizons.

#### Scenario: Horizon-specific metrics
- **WHEN** multi-horizon evaluation is configured
- **THEN** the system computes metrics for each horizon step (1-day, 7-day, 30-day ahead)
- **THEN** the system reports horizon-weighted average metrics
- **THEN** the system visualizes metric degradation over horizon

### Requirement: Model Comparison Table
The system SHALL produce a clear comparison table of all models.

#### Scenario: Comparison table generation
- **WHEN** all models are evaluated
- **THEN** the system generates table with columns: Model, MAE, RMSE, MAPE, sMAPE, MASE, WAPE, Training Time, Inference Time, Model Complexity
- **THEN** the system ranks models by primary metric (configurable, default MAE)
- **THEN** the system highlights best model per metric

### Requirement: Statistical Significance Testing
The system SHALL test for statistically significant differences between model performances.

#### Scenario: Diebold-Mariano test
- **WHEN** comparing two models' forecast accuracy
- **THEN** the system runs Diebold-Mariano test for predictive accuracy
- **THEN** the system reports p-value and significance level

#### Scenario: Multiple comparison correction
- **WHEN** comparing multiple models
- **THEN** the system applies Bonferroni or Holm correction for multiple comparisons

### Requirement: Residual Analysis and Diagnostics
The system SHALL perform residual diagnostics for the best models.

#### Scenario: Residual analysis
- **WHEN** residual analysis is requested for a model
- **THEN** the system plots residuals vs fitted values
- **THEN** the system plots residual ACF/PACF
- **THEN** the system runs normality test (Shapiro-Wilk or Jarque-Bera)
- **THEN** the system runs heteroscedasticity test (Breusch-Pagan)
- **THEN** the system runs Ljung-Box test for residual autocorrelation

### Requirement: Prediction vs Actual Visualizations
The system SHALL create clear visualizations comparing predictions to actuals.

#### Scenario: Time series forecast plots
- **WHEN** visualization is generated
- **THEN** the system plots actual vs predicted for train/validation/test periods
- **THEN** the system shows prediction intervals (if available) as shaded regions
- **THEN** the system highlights forecast horizon separately from historical fit

#### Scenario: Scatter and error distribution plots
- **WHEN** visualization is generated
- **THEN** the system creates predicted vs actual scatter plot with identity line
- **THEN** the system creates error distribution histogram
- **THEN** the system creates Q-Q plot for residual normality assessment
