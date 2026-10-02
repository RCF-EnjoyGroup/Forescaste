## MODIFIED Requirements

### Requirement: Data Loading from SQL Sources
The system SHALL load data from the specified SQL query combining test_enjoy_fac_hotel and enjoy_fac_hotel_financial tables, treating `rooms_sold` (room nights) as the primary forecast target column.

#### Scenario: Successful data load
- **WHEN** the data loading function is called with valid database credentials
- **THEN** the system returns a pandas DataFrame with all rows from both tables combined via UNION ALL

#### Scenario: Data validation on load
- **WHEN** data is loaded from the SQL sources
- **THEN** the system validates that required columns (date, hotel_id, rooms_sold) are present
- **THEN** the system validates that date column can be parsed as datetime
- **THEN** the system validates that the target column rooms_sold is numeric

## ADDED Requirements

### Requirement: NULL Target Handling Before Aggregation
The system SHALL exclude NULL `rooms_sold` values (which come exclusively from `enjoy_fac_hotel_financial` rows) from the daily room-nights aggregation BEFORE any missing-value imputation runs.

#### Scenario: Financial rows do not contaminate the target series
- **WHEN** the combined dataset contains rows with NULL rooms_sold (financial-table rows)
- **THEN** the daily aggregation skips NULL values rather than treating them as zeros
- **THEN** missing-value imputation runs only after aggregation, so forward-fill can never propagate financial NULLs into the room-nights series
- **THEN** the system reports how many rows were excluded and on which dates no operational observation existed

### Requirement: Synthetic Sample Data Generation
The system SHALL generate a synthetic (fictional) data file for offline development and testing of the room-nights forecasting model when SQL access or real data is unavailable.

#### Scenario: Fictional data file for offline testing
- **WHEN** the sample generator is invoked (e.g., no SQL configuration and no CSV present)
- **THEN** the system generates a CSV with daily rows for multiple hotels mirroring the production schema: date, hotel_id, rooms_sold, available_rooms, occupancy_rate, adr, revpar, room_revenue, revenue (plus NULL financial columns)
- **THEN** generated room nights are capacity-bounded (0 <= rooms_sold <= available_rooms) and integer-valued
- **THEN** the generated series exhibit realistic structure: weekly and annual seasonality, a gentle trend, Costa Rica holiday effects (including Monday-bridge long weekends), noise, and at least one simulated capacity change (renovation dip)
- **THEN** correlational variables are generated with exact physical identities: occupancy_rate = rooms_sold/available_rooms; room_revenue = rooms_sold × adr; revenue = room_revenue × (1 + ancillary share); revpar = room_revenue/available_rooms
- **THEN** the generated file can be loaded and processed end-to-end by the pipeline without code changes

#### Scenario: Reproducible generation
- **WHEN** the generator runs with the same seed
- **THEN** it produces an identical file (deterministic generation for testing)
