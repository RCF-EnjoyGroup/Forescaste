## ADDED Requirements

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
