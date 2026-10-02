# Data Ingestion Specification

## Purpose

Load and validate hotel financial data from SQL sources (test_enjoy_fac_hotel UNION enjoy_fac_hotel_financial) into a structured format suitable for time series forecasting.

## Requirements

### Requirement: Data Loading from SQL Sources
The system SHALL load data from the specified SQL query combining test_enjoy_fac_hotel and enjoy_fac_hotel_financial tables.

#### Scenario: Successful data load
- **WHEN** the data loading function is called with valid database credentials
- **THEN** the system returns a pandas DataFrame with all rows from both tables combined via UNION ALL

#### Scenario: Data validation on load
- **WHEN** data is loaded from the SQL sources
- **THEN** the system validates that required columns (date, hotel_id, revenue, room_revenue) are present
- **THEN** the system validates that date column can be parsed as datetime
- **THEN** the system validates that revenue columns are numeric

### Requirement: Data Schema Documentation
The system SHALL document the expected schema of the input data including column names, types, and descriptions.

#### Scenario: Schema documentation available
- **WHEN** the data ingestion module is imported
- **THEN** a schema dictionary or documentation is accessible describing all expected columns

### Requirement: Handle Missing Financial Data
The system SHALL handle the NULL values from the UNION ALL (enjoy_fac_hotel_financial has 4 extra NULL columns) appropriately.

#### Scenario: NULL column handling
- **WHEN** loading data from enjoy_fac_hotel_financial table
- **THEN** the system identifies which columns are NULL and documents their intended purpose
- **THEN** the system provides options to drop, impute, or flag these columns
