## MODIFIED Requirements

### Requirement: LightGBM Model Implementation
The system SHALL implement LightGBM for room-nights forecasting with time series appropriate configuration.

#### Scenario: LightGBM training with temporal features
- **WHEN** LightGBM model is trained
- **THEN** the system uses engineered temporal features (lags, rolling stats, calendar features) built from rooms_sold
- **THEN** the system configures objective=regression, metric=mae (or rmse)
- **THEN** the system uses early stopping on validation set
- **THEN** the system handles categorical features (hotel_id) natively

#### Scenario: Feature set excludes future-unknown columns
- **WHEN** the ML feature matrix is assembled for training or recursive prediction
- **THEN** the system excludes all monetary-derived features (revenue, room_revenue families) because their future values are unknown
- **THEN** the training feature set and the recursive-prediction feature set contain exactly the same columns

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
The system SHALL implement XGBoost for room-nights forecasting with time series appropriate configuration.

#### Scenario: XGBoost training with temporal features
- **WHEN** XGBoost model is trained
- **THEN** the system uses engineered temporal features built from rooms_sold
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
The system SHALL implement CatBoost for room-nights forecasting with native categorical feature support.

#### Scenario: CatBoost training with temporal features
- **WHEN** CatBoost model is trained
- **THEN** the system uses engineered temporal features built from rooms_sold
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
