## MODIFIED Requirements

### Requirement: Data Loading from SQL Sources
The system SHALL load hotel data from the real production sources: `staging.enjoy_fac_hotel` (booking snapshots) and `public.enjoy_inventory` (official capacity), treating realized room nights (`snap_flag = 0`, column `rooms`, by `stay_date`) as the primary forecast target.

#### Scenario: Successful data load
- **WHEN** the data loading function is called with valid source access (SQL or exported aggregate CSVs)
- **THEN** the system returns compact pandas DataFrames (realized demand, pickup by lead, capacity) — never the raw snapshot rows

#### Scenario: Data validation on load
- **WHEN** aggregates are loaded from the sources
- **THEN** the system validates that required columns (property, stay_date, rooms, snap_flag for snapshots; property, business_date, oficial_inventory for inventory) are present and typed correctly
- **THEN** the system validates that rooms is numeric and stay_date/business_date parse as datetime

#### Scenario: SQL-side aggregation pushdown
- **WHEN** the loader runs against the snapshot source
- **THEN** aggregation happens in SQL (`GROUP BY property, stay_date` with `SUM(rooms)` / `SUM(room_revenue)`), returning compact daily series to the pipeline — never loading the ~14M raw rows into pandas
- **THEN** the same applies to pickup rows (`snap_flag = 1`, grouped additionally by lead bucket) and to capacity (inventory grouped by property and business_date)

#### Scenario: Realized-demand series
- **WHEN** the target series is built
- **THEN** the system aggregates rooms with `snap_flag = 0` by canonical property and stay_date, producing one realized room-nights value per property per day
- **THEN** service-charge / system-adjust rows (rooms = 0, e.g., ratecode SYSTEM_ADJUST) do not distort the demand series

#### Scenario: Capacity series from inventory
- **WHEN** the capacity series is built
- **THEN** the system sums `oficial_inventory` across room types per canonical property and business_date
- **THEN** dates with NULL inventory fall back to the property's last known capacity and the fallback is reported

#### Scenario: Canonical property mapping
- **WHEN** demand, pickup and capacity series are joined
- **THEN** all sources resolve property identifiers through a config-driven canonical dictionary (codes and names, e.g., CORIN/Hotel Royal Corin, LAPAS, MARINA, FIESTA)
- **THEN** an unknown property identifier fails loudly rather than being silently dropped

#### Scenario: Locale-safe revenue parsing
- **WHEN** `room_revenue` arrives as text with European decimal commas (e.g., "329,6")
- **THEN** the system parses it defensively (comma-to-dot conversion, invalid values to NULL with a report)
- **THEN** parsed revenue is used for contextual/ADR analysis only, never as an ML input (future unknown)

## ADDED Requirements

### Requirement: Synthetic Generator Mirroring the Real Schemas
The system SHALL generate fictional test data that mirrors the real production schemas — a snapshot table with realized and future-book rows (`snap_flag` 0/1, `snapshotdate`/`stay_date`, lead-dependent booking growth) and an inventory table with per-room-type capacity — so the pipeline is developed and tested offline exactly as it will run in production.

#### Scenario: Snapshot-faithful fictional data
- **WHEN** the generator runs with a seed
- **THEN** it produces realized rows (one flag-0 record per property per past stay_date) and booking snapshots (flag-1 rows across many snapshotdates per stay_date, growing as the stay approaches, depth >= 365 days)
- **THEN** generated rooms are integer and capacity-bounded by the generated inventory
- **THEN** two runs with the same seed produce identical data

#### Scenario: Exchange-format parity
- **WHEN** real aggregates are exported (demand/pickup/capacity CSVs) and consumed by the pipeline
- **THEN** the same code paths run on synthetic aggregates without modification
