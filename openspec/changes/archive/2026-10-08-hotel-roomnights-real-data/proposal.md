## Why

The real production schemas are now confirmed and they obsolete the assumed schema the pipeline was built on. `staging.enjoy_fac_hotel` is a booking-snapshot table (13.9M+ future-stay rows, `snap_flag` 0=realized/1=future books, pickup depth 1,065 days, 8 properties with stay dates out to 2028) and `public.enjoy_inventory` provides official capacity by property/date/room-type. This is the industry's #1 forward-looking data (reservations on the books) already captured with history — the "excellent precision" unlock documented in the research. The validated honest-protocol machinery (recursive evaluation, backtesting, conformal bands, capacity ceiling) transfers unchanged; this change re-targets the data layer to production reality.

## What Changes

- **Real-schema ingestion**: loader consumes `staging.enjoy_fac_hotel` (snapshots) and `public.enjoy_inventory` (capacity) with **SQL-side aggregation** — 14M rows never enter pandas raw; three compact aggregate queries (realized demand, pickup by lead, capacity)
- **Realized-demand target series**: rooms with `snap_flag=0` aggregated by (canonical property, stay_date); service-charge/system-adjust rows (rooms=0) excluded from demand
- **Pickup (on-the-books) features**: rooms with `snap_flag=1` pivoted by lead time (stay_date − snapshotdate) into on-the-books features at lead buckets (7/14/30/60/90/180/365), with the **as-of discipline**: for a 90-day forecast window only leads >= 90 are forward-known across the whole window; shorter leads are documented for a rolling-refresh cadence
- **Canonical property mapping**: codes (LAPAS, MARINA, FIESTA, ...) and names ("Hotel Royal Corin") joined through a config-driven dictionary; demand, pickup and capacity must join on the same key
- **Capacity ceiling from inventory**: `oficial_inventory` summed across room types per (property, business_date) — the ceiling and implied occupancy; NULL inventory dates fall back to the property's last known capacity (documented)
- **Locale-safe revenue parsing**: `room_revenue` arrives as text with European comma decimals — parsed defensively for contextual/ADR analysis (not an ML feature)
- **New-hotel regime (SJOSL)**: properties with no realized history are forecast through the global model using pickup, capacity and calendar signals; reported honestly as a cold-start case
- **Synthetic generator v2 mirroring the REAL schemas**: snapshot table (flag 0/1 with pickup process) + inventory table, so offline development and tests keep running without SQL access
- **Training window policy**: recent window (default last 3-5 years) for model fitting despite 18-year history; regime changes in old data are documented, not silently mixed in

## Capabilities

### New Capabilities
None - this re-targets and extends the existing forecasting capabilities.

### Modified Capabilities
- `hotel-revenue-forecasting/data-ingestion`: real snapshot/capacity sources, SQL-side aggregation, snap_flag semantics, canonical property mapping, locale-safe parsing, synthetic generator mirroring real schemas
- `hotel-revenue-forecasting/preprocessing`: on-the-books (pickup) features by lead bucket with as-of discipline, alongside the existing target-family features
- `hotel-revenue-forecasting/forecasting`: per-property future forecasts built from realized history + current books, capped by inventory capacity, with cold-start (new-hotel) handling

## Impact

- `config.yaml`: `roomnights` section re-targeted to real tables (source queries, property mapping, lead buckets, training window)
- `src/forecasting/data/`: new aggregation loader (3 pushdown queries), property mapping, revenue parsing; `schema.py` documents the real schemas
- `src/forecasting/preprocessing/`: pickup feature builder (as-of safe)
- Synthetic sample files regenerated mirroring real schemas (`data/` , offline dev)
- `tests/`: snap_flag semantics, as-of no-leakage for pickup features, mapping integrity, parsing, cold-start handling
- Dependencies: openpyxl (Excel exports) added; no model-stack changes
- The honest-protocol evaluation stack (baseline, recursive, backtesting, conformal) is reused unchanged
