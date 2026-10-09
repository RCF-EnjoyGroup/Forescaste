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
The system SHALL create relevant temporal features for room-nights forecasting, building lag/rolling/expanding features from the configured target column (`rooms_sold`) and excluding features whose future values are unknown.

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
- **THEN** the system creates lags of the target column at configurable periods (e.g., 1, 7, 14, 30, 365 days)
- **THEN** the system names lag columns consistently (e.g., rooms_sold_lag_1, rooms_sold_lag_7)

#### Scenario: Rolling window statistics
- **WHEN** rolling features are requested
- **THEN** the system creates rolling mean, std, min, max of the target at configurable windows (7, 14, 30, 90 days)
- **THEN** the system uses only past data (shifted by 1) to avoid leakage

#### Scenario: Expanding window statistics
- **WHEN** expanding features are requested
- **THEN** the system creates expanding mean, std of the target for long-term trend capture
- **THEN** the system uses only past data (shifted by 1) to avoid leakage

#### Scenario: Exclusion of future-unknown features
- **WHEN** the ML feature set is built for the room-nights target
- **THEN** the system excludes all features derived from monetary columns (revenue, room_revenue) whose future values are unknown, because recursive forecasting would otherwise require fabricated inputs
- **THEN** capacity columns (available_rooms) and calendar/holiday features remain eligible inputs

#### Scenario: Forward-known rule for exogenous candidates
- **WHEN** a candidate input variable is evaluated for inclusion in the feature set
- **THEN** the system includes it only if its future values are known in advance or it is itself reliably forecastable
- **THEN** realized outcome variables (observed adr, revpar, occupancy_rate) are excluded as inputs, while forward-decided variables (planned/published rates, if ever provided) would be eligible

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

### Requirement: On-the-Books (Pickup) Features with As-Of Discipline
The system SHALL build on-the-books features from booking snapshots (rooms with `snap_flag = 1`) at lead-time buckets (e.g., 7/14/30/60/90/180/365 days before stay), respecting the as-of rule: a feature for a forecast origin may only use snapshots taken at or before that origin.

#### Scenario: Lead-bucket features from snapshots
- **WHEN** pickup features are built for a stay date
- **THEN** the system derives on-the-books values at each configured lead bucket (rooms in books at stay_date − lead), per canonical property
- **THEN** missing snapshots at a bucket fall back to the nearest earlier snapshot, and the fallback is deterministic and reported

#### Scenario: As-of safety for the multi-step horizon
- **WHEN** the system builds ML inputs for a 90-day multi-step forecast window
- **THEN** only lead buckets >= the window length (e.g., d90 and longer) are forward-known across the entire window and are used as recursive inputs
- **THEN** shorter-lead buckets (d7/d14/d30/d60) are excluded from multi-step inputs (their snapshots would occur inside the forecast window = look-ahead) and are documented as the rolling-refresh feature set

#### Scenario: No leakage in training rows
- **WHEN** pickup features are attached to historical training rows
- **THEN** each row uses only snapshots taken at or before (stay_date − lead), never information from after that moment
- **THEN** a unit test proves a snapshot dated after (stay_date − lead) never influences the feature

### Requirement: Realized-Demand Feature Baseline on the Real Schema
The system SHALL keep the validated target-family feature set (lags, rolling, expanding of realized room nights) plus calendar/holiday and capacity features, unchanged in semantics, on the real-schema target series.

#### Scenario: Target-family features transfer
- **WHEN** features are engineered for the realized-demand series
- **THEN** lags/rolling/expanding of realized room nights, calendar, Costa Rica holidays and inventory capacity are produced with the same no-leakage shift semantics as before
- **THEN** monetary-derived and realized-outcome columns (revenue, ADR, occupancy) remain excluded from ML inputs by the forward-known rule
