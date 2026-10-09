## Context

The honest-protocol room-nights pipeline (archived `hotel-roomnights-forecasting`) runs on an assumed schema that the confirmed production sources now obsolete. Confirmed reality (from systems' exports):

- `staging.enjoy_fac_hotel`: booking-snapshot table. `snap_flag` 0 = realized, 1 = future books. One stay date carries many snapshot rows (e.g., 209 snapshots for one hotel-night; books grow from 2 rooms at d-223 to ~51 realized). ~14M rows with stay_date > snapshotdate; pickup depth 1,065 days; 8 properties; stay dates out to 2028; `room_revenue` is text with European comma decimals; service-charge rows (SYSTEM_ADJUST) carry rooms=0.
- `public.enjoy_inventory`: capacity by property × business_date × room_type (`oficial_inventory`); property codes (LAPAS, MARINA, CORIN→Hotel Royal Corin mapping); ooo/os/size columns observed all-NULL; sparsity in preview requires a coverage check.
- Property identifiers mix codes and names in the same table; SJOSL opened 2026-01-05 (no realized history).

See proposal.md - Why: the #1 forward-looking variable (pickup) is already captured with history.

## Goals / Non-Goals

**Goals:**
- Re-target ingestion to the real sources with SQL-side aggregation (3 compact queries; raw 14M rows never leave the database)
- Build as-of-safe pickup features by lead bucket
- Capacity ceiling + occupancy from official inventory with canonical property mapping
- Cold-start handling for SJOSL (honest, explicit)
- Offline development parity: synthetic generator mirrors the real schemas (snapshots + inventory)
- Re-run the full honest protocol (baseline, recursive multi-step, backtesting, conformal bands, per-property) on real aggregates

**Non-Goals:**
- Room-type-level forecasting (roadmap; hotel-level first)
- Rolling/weekly re-forecast cadence with short-lead pickup (documented as the natural next step)
- Hierarchical reconciliation (portfolio vs sum-of-hotels gap reported, not corrected)
- Direct production deployment/scheduler

## Decisions

### 1. Three aggregate queries instead of raw loads (pushdown)
**Decision:** SQL layer produces: (a) realized demand — `SELECT property, stay_date, SUM(rooms), SUM(revenue-parsed) WHERE snap_flag=0 GROUP BY 1,2`; (b) pickup — `snap_flag=1 GROUP BY property, stay_date, lead bucket` (lead = stay_date − snapshotdate); (c) capacity — inventory `GROUP BY property, business_date` summing oficial_inventory. CSV-export workflow is the default interface (systems runs the queries; pipeline consumes exports); a live-SQL mode exists but exports keep development offline and auditable.
**Rationale:** 14M raw rows in pandas is a memory and reproducibility hazard; the aggregates are the actual modeling objects.
**Alternatives considered:** load raw with filters (rejected: fragile, slow, still large); full warehouse views (out of our control).

### 2. Target series = flag-0 realized rooms; segments filtered
**Decision:** realized demand per (canonical property, stay_date) = SUM(rooms) over `snap_flag=0`, excluding ratecode SYSTEM_ADJUST / service-charge rows (they carry rooms=0; they are accounting noise for demand).
**Rationale:** matches the confirmed structure (one realized record per stay night); service charges never carry room nights.
**Alternatives considered:** including all rows (rejected: dilutes nothing but complicates; rooms=0 rows are no-ops yet the filter documents intent).

### 3. Pickup features by lead bucket with strict as-of rule
**Decision:** features `rokb_d{7,14,30,60,90,180,365}` = rooms in books at stay_date − lead (nearest earlier snapshot if the exact lead is missing). For multi-step forecasts of length H (default 90 days), only buckets with lead >= H enter the ML input (rokb_d90+); d7–d60 are the rolling-refresh set, documented. Training rows attach snapshots dated at or before (stay_date − lead) only — unit-tested.
**Rationale:** using a d30 book for a day 40 days into a 90-day window would be look-ahead (the d30 snapshot happens after the origin). Depth 1,065 days makes d90/d180/d365 fully trainable.
**Alternatives considered:** only latest-snapshot books (rejected: throws away the curve); per-step lead alignment (complexity deferred).

### 4. Capacity from oficial_inventory with last-known fallback
**Decision:** capacity per (property, business_date) = SUM(oficial_inventory across room types). NULL dates → last known capacity for the property, counted and reported. Dead columns (ooo_rooms/os_rooms/size) ignored.
**Rationale:** oficial_inventory matched physical_room where present and is the official sellable number; silent NULLs would break the ceiling.
**Alternatives considered:** max-observed-rooms proxy (kept as emergency fallback only if a property has no inventory at all).

### 5. Canonical property dictionary in config
**Decision:** a config section maps every identifier variant (CORIN, "Hotel Royal Corin", LAPAS, MARINA, VILLAS, SJOSL, LIRAK, LIREL, FIESTA) to one canonical key; joins happen only on canonical keys; unknown identifiers raise.
**Rationale:** the same table mixes codes and names; a silent mismatch would drop a hotel's capacity or demand silently — the classic integration bug.
**Alternatives considered:** CASE in SQL per query (rejected: duplicates the dictionary in N queries).

### 6. Cold start via the global model, honestly labeled
**Decision:** SJOSL (and any property with < ~90 realized days) is forecast only by the global GBM family (property as categorical) using pickup/capacity/calendar; statistical per-series models and per-hotel backtests are skipped for it; reports flag it as cold-start.
**Rationale:** no own-history lags exist; the global model transfers cross-property structure — exactly what the feature design enables; pretending a per-hotel SARIMAX is meaningful there would be fake rigor.
**Alternatives considered:** excluding SJOSL (rejected: it is a real business asset needing a forecast, with honest caveats).

### 7. Training window: recent years only
**Decision:** default training window = last 4 years of realized data per property (config), despite 18-year history for Corin.
**Rationale:** pre-2015 demand regimes (different systems/markets; SKILL4 legacy) add noise; recent window keeps the model representative. Documented in ASSUMPTIONS; revisit if backtesting shows older data helps.
**Alternatives considered:** full history (rejected: regime change risk); fixed global dates (rejected: per-property history differs — window is relative).

### 8. Synthetic generator v2 mirrors the real schemas
**Decision:** generator emits (a) a snapshot table replica (flag-0 realized rows + flag-1 booking rows with lead-dependent growth and depth ≥ 365d) and (b) an inventory replica (property × date × room type with oficial_inventory), with a booking-arrival distribution per hotel (leisure vs group mix); outputs the same three aggregate CSVs the real queries produce.
**Rationale:** offline parity — every pipeline code path (as-of logic, joins, cold-start) is exercised identically on synthetic and real data; the previous generator mirrored the obsolete schema, which is exactly the failure mode this decision prevents repeating.
**Alternatives considered:** patching the old generator (rejected: snapshot semantics are a different data model, not extra columns).

## Risks / Trade-offs

- [Property mapping missing a variant] → unknown identifiers fail loudly; a mapping test iterates all observed IDs.
- [Inventory sparsity / NULL capacity dates] → last-known fallback + coverage report; if a property shows < 50% coverage, its occupancy reporting is flagged unreliable.
- [snap_flag semantics assumption (single realized record per stay_date)] → loader asserts uniqueness of (property, stay_date) for flag=0; violation surfaces immediately.
- [Comma-decimal revenue parsing edge cases] → invalid parses become NULL with a count; revenue is contextual-only, so a parse miss cannot corrupt demand.
- [Cold-start forecasts over-trusted] → explicit cold-start flag + no per-hotel backtest claims for SJOSL.
- [Old-history regime mixing] → training-window policy (Decision 7) + backtesting only within the window.

## Migration Plan

Additive re-target: the aggregation layer, mapping, and pickup features are new modules/paths; the synthetic v1 file and notebook remain as historical deliverables. Rollback = revert this change; nothing in the archived pipeline's modules is deleted (loader gains new methods; schema constants gain the real schemas alongside).

## Open Questions

1. **Inventory coverage**: exact % of NULL `oficial_inventory` per property in the full table (preview suggested sparsity; the coverage report will quantify on first real run).
2. **Access mode**: will the pipeline run the aggregate queries directly against the DWH, or will systems export CSVs on a schedule? (Both are supported; the default is exports.)
3. **Business training window**: is 4 years the right default for Enjoy's market memory, or does management want a different window?
