## MODIFIED Requirements

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
