## ADDED Requirements

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

## MODIFIED Requirements

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
