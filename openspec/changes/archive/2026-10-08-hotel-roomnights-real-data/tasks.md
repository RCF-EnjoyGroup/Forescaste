## 1. Configuration & Property Mapping

- [x] 1.1 Add `roomnights_real` section to `config.yaml` (source queries, canonical property dictionary covering all 8 observed identifiers, lead buckets, training window, inventory fallback policy); verify `yaml.safe_load` returns it and existing sections are unchanged
- [x] 1.2 Implement the canonical property resolver (`data/property_map.py`): every known variant maps to one key; unknown identifiers raise with the offending value; verify a unit test passes for all 8 observed IDs and fails loudly for an unknown one

## 2. Ingestion & Aggregation (SQL pushdown)

- [x] 2.1 Implement the three aggregate SQL queries in the loader (realized demand by snap_flag=0 excluding SYSTEM_ADJUST; pickup by snap_flag=1 grouped by lead bucket; capacity summing oficial_inventory by property/date); verify each query's SQL string is documented and the loader returns compact DataFrames from the exported CSVs
- [x] 2.2 Implement locale-safe `room_revenue` parsing (comma-to-dot, invalid -> NULL + count); verify a unit test parses "329,6" -> 329.6 and reports invalid values
- [x] 2.3 Assert flag-0 uniqueness (one realized record per property+stay_date) at load time; verify a synthetic violation raises a clear error
- [x] 2.4 Implement capacity series build with last-known fallback for NULL inventory dates; verify a unit test exercises the fallback and counts it

## 3. Pickup Features (as-of safe)

- [x] 3.1 Implement the pickup feature builder (lead buckets 7/14/30/60/90/180/365; nearest-earlier-snapshot fallback, deterministic + reported); verify a unit test builds rotb_d90 from snapshot rows at the exact lead and from an earlier snapshot when the exact one is missing
- [x] 3.2 Implement the as-of guard for multi-step inputs: only lead >= horizon buckets enter recursive inputs; verify a unit test proves a snapshot dated after (stay_date - lead) never influences a training row (no-leakage)
- [x] 3.3 Verify the recursive forecast machinery accepts per-date forward-known covariates (rokb_d90 varies by future date, unlike constant capacity); extend `recursive_forecast` with a future-covariates DataFrame and a unit test asserting incremental == reference paths

## 4. Synthetic Generator v2 (real-schema mirror)

- [x] 4.1 Implement snapshot-table generation (flag-0 realized rows; flag-1 booking rows with lead-dependent growth, depth >= 365 days, per-hotel arrival mix including a group-heavy hotel) + inventory generation (property x date x room type); verify determinism by seed, integer capacity-bounded rooms, and books <= realized on average
- [x] 4.2 Generate and persist `data/roomnights_real_sample/` (3 aggregate CSVs: demand, pickup, capacity) with a cold-start property (SJOSL-like, no realized history); verify the pipeline consumes them end-to-end without code changes vs the real export format

## 5. Modeling & Notebook on the Real Schema

- [x] 5.1 Re-target the room-nights notebook to the real aggregates (target = realized demand; capacity ceiling from inventory; canonical property names); verify every prior honest-protocol element is present (baseline, recursive multi-step, DM Newey-West, backtest, conformal bands, bias/weekday diagnostics)
- [x] 5.2 Add pickup features (rotb_d90+) to the ML inputs and re-run tuning + comparison + backtesting; verify the as-of rule holds in every model input and report the pickup feature importance
- [x] 5.3 Implement cold-start handling for SJOSL (global-model-only forecast, explicit flag, no per-hotel backtest claims); verify the report labels it
- [x] 5.4 Implement per-property forecast with books coverage summary (share of the 90-day forecast already on the books) and implied occupancy from inventory; verify the summary shows all properties with their occupancy and books coverage
- [x] 5.5 Execute the notebook end-to-end on the synthetic real-schema aggregates; verify 0 errors and the honest conclusions (then repeat on real exports when systems delivers them)
- [x] 5.6 Export the notebook to a self-contained HTML report; verify the file is generated

## 6. Tests & Validation

- [x] 6.1 Add tests: snap_flag semantics (demand from flag-0 only; pickup growth monotonic on average), as-of no-leakage, property mapping integrity, revenue parsing, capacity fallback, cold-start flagging; verify all pass
- [x] 6.2 Run the complete suite (existing 54 + new); verify pytest exits 0
- [x] 6.3 Update README, ASSUMPTIONS.md (training window, inventory fallback, cold-start, books coverage) and CHANGELOG; verify the three files document the real-data pivot
