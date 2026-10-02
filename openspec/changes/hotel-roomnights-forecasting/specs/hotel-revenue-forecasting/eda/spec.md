## MODIFIED Requirements

### Requirement: Descriptive Statistics and Distribution Analysis
The system SHALL compute and display descriptive statistics for the room-nights series (`rooms_sold`) as the primary target, alongside available contextual columns.

#### Scenario: Summary statistics generation
- **WHEN** EDA is run on loaded data
- **THEN** the system outputs mean, median, std, min, max, quartiles for rooms_sold
- **THEN** the system displays histogram and boxplot visualizations for the room-nights series

#### Scenario: Missing value analysis
- **WHEN** EDA is run on loaded data
- **THEN** the system reports count and percentage of missing values per column
- **THEN** the system visualizes missing value patterns (if any)

### Requirement: Stationarity Testing
The system SHALL test for stationarity using Augmented Dickey-Fuller (ADF) and KPSS tests on the room-nights series.

#### Scenario: ADF test execution
- **WHEN** stationarity testing is requested
- **THEN** the system runs ADF test on the rooms_sold series
- **THEN** the system reports test statistic, p-value, and critical values
- **THEN** the system provides clear interpretation (stationary vs non-stationary)

#### Scenario: KPSS test execution
- **WHEN** stationarity testing is requested
- **THEN** the system runs KPSS test on the rooms_sold series
- **THEN** the system reports test statistic, p-value, and critical values

### Requirement: Outlier Detection
The system SHALL identify outliers in the room-nights time series, including physically impossible values.

#### Scenario: Statistical outlier detection
- **WHEN** outlier analysis is run
- **THEN** the system applies IQR method and/or z-score method
- **THEN** the system flags outliers and provides count/percentage
- **THEN** the system visualizes outliers on time series plot

#### Scenario: Capacity violation detection
- **WHEN** any rooms_sold value exceeds available_rooms for the same date and hotel
- **THEN** the system flags it as a data-quality error (implied occupancy above 100%)
- **THEN** the system reports the offending dates and hotels

### Requirement: Correlation Analysis
The system SHALL analyze correlations between the room-nights target and contextual features (occupancy_rate, adr, revenue, room_revenue) for understanding, without implying they become ML inputs.

#### Scenario: Correlation heatmap
- **WHEN** correlation analysis is run
- **THEN** the system computes Pearson and Spearman correlations between rooms_sold and contextual columns
- **THEN** the system displays annotated heatmap visualization
