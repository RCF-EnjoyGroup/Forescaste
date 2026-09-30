## Purpose

Perform comprehensive exploratory data analysis on hotel revenue time series data to understand distributions, seasonality, trends, stationarity, and autocorrelation patterns.

## ADDED Requirements

### Requirement: Descriptive Statistics and Distribution Analysis
The system SHALL compute and display descriptive statistics for revenue and room_revenue columns.

#### Scenario: Summary statistics generation
- **WHEN** EDA is run on loaded data
- **THEN** the system outputs mean, median, std, min, max, quartiles for revenue and room_revenue
- **THEN** the system displays histogram and boxplot visualizations for both target variables

#### Scenario: Missing value analysis
- **WHEN** EDA is run on loaded data
- **THEN** the system reports count and percentage of missing values per column
- **THEN** the system visualizes missing value patterns (if any)

### Requirement: Time Series Decomposition
The system SHALL decompose the time series into trend, seasonal, and residual components.

#### Scenario: Seasonal decomposition
- **WHEN** the time series has sufficient observations (at least 2 full seasonal cycles)
- **THEN** the system performs additive and/or multiplicative decomposition
- **THEN** the system plots trend, seasonal, and residual components separately

### Requirement: Stationarity Testing
The system SHALL test for stationarity using Augmented Dickey-Fuller (ADF) and KPSS tests.

#### Scenario: ADF test execution
- **WHEN** stationarity testing is requested
- **THEN** the system runs ADF test on revenue and room_revenue series
- **THEN** the system reports test statistic, p-value, and critical values
- **THEN** the system provides clear interpretation (stationary vs non-stationary)

#### Scenario: KPSS test execution
- **WHEN** stationarity testing is requested
- **THEN** the system runs KPSS test on revenue and room_revenue series
- **THEN** the system reports test statistic, p-value, and critical values

### Requirement: Autocorrelation and Partial Autocorrelation Analysis
The system SHALL compute and visualize ACF and PACF for the time series.

#### Scenario: ACF/PACF visualization
- **WHEN** autocorrelation analysis is run
- **THEN** the system plots ACF and PACF with confidence intervals
- **THEN** the system identifies significant lags for model selection

### Requirement: Seasonality Detection
The system SHALL detect and characterize seasonal patterns at multiple granularities.

#### Scenario: Weekly seasonality detection
- **WHEN** data has daily granularity
- **THEN** the system tests for day-of-week effects using boxplots and statistical tests

#### Scenario: Monthly/Yearly seasonality detection
- **WHEN** data spans multiple months/years
- **THEN** the system visualizes monthly and yearly patterns
- **THEN** the system creates seasonal heatmaps (month vs year)

### Requirement: Outlier Detection
The system SHALL identify outliers in the revenue time series.

#### Scenario: Statistical outlier detection
- **WHEN** outlier analysis is run
- **THEN** the system applies IQR method and/or z-score method
- **THEN** the system flags outliers and provides count/percentage
- **THEN** the system visualizes outliers on time series plot

### Requirement: Correlation Analysis
The system SHALL analyze correlations between revenue, room_revenue, and any additional features.

#### Scenario: Correlation heatmap
- **WHEN** correlation analysis is run
- **THEN** the system computes Pearson and Spearman correlations
- **THEN** the system displays annotated heatmap visualization