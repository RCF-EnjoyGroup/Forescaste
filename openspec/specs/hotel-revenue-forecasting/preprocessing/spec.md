# Preprocessing Specification

## Purpose

Prepare hotel revenue time series data for modeling through missing value treatment, outlier handling, temporal feature engineering, and proper temporal train/validation/test splits.

## Requirements

### Requirement: Missing Value Treatment
The system SHALL handle missing values in the time series data using appropriate imputation strategies.

#### Scenario: Forward/backward fill for time series
- **WHEN** missing values are detected in revenue columns
- **THEN** the system applies forward fill then backward fill for temporal consistency
- **THEN** the system logs the number of imputed values per column

#### Scenario: Interpolation for gaps
- **WHEN** missing values form gaps larger than 1 period
- **THEN** the system offers time-based interpolation as an option
- **THEN** the system validates interpolated values are reasonable

### Requirement: Outlier Treatment
The system SHALL provide configurable outlier treatment for revenue data.

#### Scenario: Winsorization
- **WHEN** outlier treatment is enabled
- **THEN** the system applies winsorization at configurable percentiles (default 1%/99%)
- **THEN** the system reports number of values capped

#### Scenario: Outlier flagging
- **WHEN** outlier treatment is set to flag-only mode
- **THEN** the system adds boolean column indicating outlier status
- **THEN** the system preserves original values

### Requirement: Temporal Feature Engineering
The system SHALL create relevant temporal features for time series forecasting.

#### Scenario: Calendar features
- **WHEN** feature engineering is run on datetime index
- **THEN** the system creates: year, quarter, month, week, day, dayofweek, dayofyear
- **THEN** the system creates: is_weekend, is_month_start, is_month_end, is_quarter_start, is_quarter_end

#### Scenario: Cyclical encoding
- **WHEN** cyclical features are requested
- **THEN** the system creates sin/cos transformations for: hour, dayofweek, dayofyear, month
- **THEN** the system uses 2π period for proper cyclical representation

#### Scenario: Lag features
- **WHEN** lag features are requested
- **THEN** the system creates lags at configurable periods (e.g., 1, 7, 14, 30, 365 days)
- **THEN** the system names lag columns consistently (e.g., revenue_lag_1, revenue_lag_7)

#### Scenario: Rolling window statistics
- **WHEN** rolling features are requested
- **THEN** the system creates rolling mean, std, min, max at configurable windows (7, 14, 30, 90 days)
- **THEN** the system uses only past data (shifted by 1) to avoid leakage

#### Scenario: Expanding window statistics
- **WHEN** expanding features are requested
- **THEN** the system creates expanding mean, std for long-term trend capture
- **THEN** the system uses only past data (shifted by 1) to avoid leakage

### Requirement: Holiday/Event Features
The system SHALL incorporate holiday and special event indicators if available.

#### Scenario: Holiday calendar integration
- **WHEN** holiday data is provided
- **THEN** the system creates binary holiday indicators
- **THEN** the system creates pre/post holiday windows (e.g., 1-3 days before/after)

### Requirement: Temporal Train/Validation/Test Split
The system SHALL split data chronologically respecting temporal order.

#### Scenario: Configurable split ratios
- **WHEN** temporal split is requested
- **THEN** the system splits by date: train (earliest), validation (middle), test (latest)
- **THEN** the system supports configurable ratios (default 70/15/15 or 80/10/10)
- **THEN** the system ensures no data leakage (no future data in train/val)

#### Scenario: Gap between splits
- **WHEN** temporal split is requested
- **THEN** the system optionally adds gap periods between splits to reduce leakage
- **THEN** the system reports split dates and observation counts per split

#### Scenario: Per-hotel splits
- **WHEN** data contains multiple hotels
- **THEN** the system supports splitting per hotel or globally
- **THEN** the system ensures each hotel has data in all splits (if sufficient data)

### Requirement: Feature Scaling
The system SHALL provide scaling options for ML models.

#### Scenario: Standard/MinMax scaling
- **WHEN** scaling is requested
- **THEN** the system fits scaler on training data only
- **THEN** the system applies same transformation to validation and test
- **THEN** the system supports StandardScaler, MinMaxScaler, RobustScaler
