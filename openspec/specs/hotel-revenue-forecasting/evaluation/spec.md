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
The system SHALL test for statistically significant differences between model performances, accounting for autocorrelation of multi-step forecast errors.

#### Scenario: Diebold-Mariano test
- **WHEN** comparing two models' forecast accuracy
- **THEN** the system runs Diebold-Mariano test for predictive accuracy
- **THEN** the system reports p-value and significance level

#### Scenario: Newey-West correction for multi-step errors
- **WHEN** forecast errors are multi-step and autocorrelated
- **THEN** the system estimates the loss-differential variance with a Newey-West HAC estimator (lag truncation based on the forecast horizon)
- **THEN** the reported significance is not inflated by ignoring error autocorrelation

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

### Requirement: Forecast Bias and Segment Diagnostics
The system SHALL report the direction of forecast errors alongside their magnitude, and break errors down by day-of-week for operational diagnosis.

#### Scenario: Signed bias reported with every evaluation
- **WHEN** metrics are computed for a model
- **THEN** the system reports a signed bias percentage (positive = over-forecasting, negative = under-forecasting) and the counts of over- and under-forecast days
- **THEN** the bias is displayed in the comparison table and in the conclusions report

#### Scenario: Per-day-of-week error breakdown for the best model
- **WHEN** a best model is selected
- **THEN** the system reports per-day-of-week error statistics (MAE, MAPE, signed bias)
- **THEN** systematically worse segments (e.g., weekends vs midweek) are visible for staffing and pricing follow-up

### Requirement: Rolling-Origin Backtesting for Rank Stability
The system SHALL evaluate model ranking stability across multiple historical origins (backtesting), not only on a single hold-out split.

#### Scenario: Multi-origin evaluation
- **WHEN** models are compared
- **THEN** the system runs a rolling-origin backtest with multiple folds, where each fold's forecast uses ONLY data observed strictly before the fold's window start
- **THEN** the system reports per-fold MAE per model, the winner of each fold, and mean rank across folds
- **THEN** the model-selection conclusion cites the backtest evidence, not only the single test split

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

### Requirement: Seasonal Naive Baseline in Every Comparison
The system SHALL include a Seasonal Naive baseline (last observed seasonal cycle repeated) in every model comparison for the room-nights target.

#### Scenario: Baseline included in comparison table
- **WHEN** models are compared on the evaluation window
- **THEN** the comparison table contains a SeasonalNaive entry built only from observed history before the evaluation window
- **THEN** MASE is computed against this baseline so every model is judged by its ability to beat the trivial forecast
- **THEN** models that fail to beat the baseline are reported as such (no value delivered)

### Requirement: Leakage-Free Recursive Multi-Step Evaluation
The system SHALL evaluate ML models on the hold-out window using recursive multi-step forecasting in which only the model's own predictions are fed back as lag inputs.

#### Scenario: Recursive evaluation of the hold-out window
- **WHEN** an ML model is evaluated on the validation/test window
- **THEN** the model receives only information observed before the evaluation window start
- **THEN** each step's prediction is fed back as the lag input for subsequent steps
- **THEN** actual values from the evaluation window are never used as model inputs
- **THEN** predictions for all models are aligned to the same requested dates and compared on identical series

#### Scenario: Final model retrained on all observed history
- **WHEN** the best model is selected for future forecasting
- **THEN** it is retrained on all observed data (train + validation + test) before generating the future forecast, using iteration counts fixed during validation
