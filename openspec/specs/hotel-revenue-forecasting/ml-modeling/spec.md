# ML Modeling Specification

## Purpose

Implement and evaluate machine learning models (LightGBM, XGBoost, CatBoost) with temporal features for hotel revenue forecasting.

## Requirements

### Requirement: LightGBM Model Implementation
The system SHALL implement LightGBM for revenue forecasting with time series appropriate configuration.

#### Scenario: LightGBM training with temporal features
- **WHEN** LightGBM model is trained
- **THEN** the system uses engineered temporal features (lags, rolling stats, calendar features)
- **THEN** the system configures objective=regression, metric=mae (or rmse)
- **THEN** the system uses early stopping on validation set
- **THEN** the system handles categorical features (hotel_id) natively

#### Scenario: LightGBM hyperparameter optimization
- **WHEN** hyperparameter tuning is enabled
- **THEN** the system tunes: num_leaves, learning_rate, n_estimators, max_depth, min_child_samples, subsample, colsample_bytree
- **THEN** the system uses Optuna or similar for Bayesian optimization
- **THEN** the system uses temporal cross-validation (TimeSeriesSplit) for validation

#### Scenario: LightGBM prediction
- **WHEN** trained LightGBM predicts on validation/test data
- **THEN** the system generates point forecasts
- **THEN** the system optionally provides prediction intervals via quantile regression or conformal prediction

### Requirement: XGBoost Model Implementation
The system SHALL implement XGBoost for revenue forecasting with time series appropriate configuration.

#### Scenario: XGBoost training with temporal features
- **WHEN** XGBoost model is trained
- **THEN** the system uses engineered temporal features
- **THEN** the system configures objective=reg:squarederror, eval_metric=mae (or rmse)
- **THEN** the system uses early stopping on validation set
- **THEN** the system handles categorical features via one-hot or target encoding

#### Scenario: XGBoost hyperparameter optimization
- **WHEN** hyperparameter tuning is enabled
- **THEN** the system tunes: max_depth, learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda
- **THEN** the system uses temporal cross-validation for validation

#### Scenario: XGBoost prediction
- **WHEN** trained XGBoost predicts on validation/test data
- **THEN** the system generates point forecasts

### Requirement: CatBoost Model Implementation
The system SHALL implement CatBoost for revenue forecasting with native categorical feature support.

#### Scenario: CatBoost training with temporal features
- **WHEN** CatBoost model is trained
- **THEN** the system uses engineered temporal features
- **THEN** the system leverages native categorical feature handling for hotel_id
- **THEN** the system configures loss_function=MAE (or RMSE)
- **THEN** the system uses early stopping on validation set

#### Scenario: CatBoost hyperparameter optimization
- **WHEN** hyperparameter tuning is enabled
- **THEN** the system tunes: depth, learning_rate, iterations, l2_leaf_reg, bagging_temperature
- **THEN** the system uses temporal cross-validation for validation

#### Scenario: CatBoost prediction
- **WHEN** trained CatBoost predicts on validation/test data
- **THEN** the system generates point forecasts

### Requirement: ML Model Evaluation
The system SHALL evaluate all ML models using standard time series metrics.

#### Scenario: Metric computation
- **WHEN** ML models are evaluated
- **THEN** the system computes: MAE, RMSE, MAPE, sMAPE, MASE, WAPE
- **THEN** the system computes metrics on validation and test sets separately
- **THEN** the system reports feature importance for tree-based models

#### Scenario: Temporal cross-validation
- **WHEN** model evaluation is run
- **THEN** the system uses TimeSeriesSplit with expanding or sliding window
- **THEN** the system reports mean and std of metrics across folds
