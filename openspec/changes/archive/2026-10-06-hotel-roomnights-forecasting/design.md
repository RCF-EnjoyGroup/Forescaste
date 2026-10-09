## Context

The honest-protocol forecasting pipeline delivered by the archived `hotel-revenue-forecasting-pipeline` change is implemented and verified (28 tests, notebook executed end-to-end): leakage-safe preprocessing, date-aligned statistical models, recursive multi-step ML evaluation with a verified incremental fast path, SeasonalNaive baseline, and Diebold-Mariano with Newey-West correction. See proposal.md - Why for the business pivot.

New constraints for this change:
- Primary target is room nights (`rooms_sold`), an integer count bounded by `available_rooms`.
- `rooms_sold` is NULL on every row originating from `enjoy_fac_hotel_financial` — only `test_enjoy_fac_hotel` contributes operational values.
- Development must be testable offline: a fictional data file is a first-class requirement, not a convenience.
- Revenue forecasting must remain intact (the pivot is "first", not "instead").

## Goals / Non-Goals

**Goals:**
- Re-target the existing pipeline to `rooms_sold` via configuration, not code duplication
- Deliver a dedicated synthetic data file with realistic, capacity-bounded room nights for offline testing
- Guarantee the NULL rows from the financial table never contaminate the room-nights series
- Report forecasts in room-night units with capacity ceiling and implied occupancy
- Preserve the honest evaluation protocol unchanged (baseline, recursive evaluation, DM + Newey-West)

**Non-Goals:**
- Count-specific model families (Poisson/NB GLM) — point forecasts from existing models suffice at this scale; documented as future work
- Changes to the revenue notebook or its results
- Production deployment, schedulers, dashboards
- Hierarchical reconciliation (hotel → portfolio)

## Decisions

### 1. Configurable target column, not hard-coded pivot
**Decision:** `config.yaml` gains `target_col: rooms_sold` (with `available_rooms_col`); the pipeline, feature engineer and recursive forecaster already receive `target_col` parameters — the change parameterizes defaults rather than forking code.
**Rationale:** the same modules must keep serving the revenue notebook unchanged; a hard-coded pivot would fork the codebase.
**Alternatives considered:** duplicating the package under a roomnights namespace (rejected: double maintenance); making the notebook the only place to set the target (rejected: config is the versioned source of truth).

### 2. Synthetic data file: dedicated generator output, realistic correlational structure
**Decision:** extend `DataLoader.load_sample` with room-nights semantics and write a dedicated file `data/hotel_roomnights_sample.csv` mirroring the REAL production schema exactly (date, hotel_id, rooms_sold, available_rooms, occupancy_rate, adr, revpar, room_revenue, revenue, cost_center/account_code/debit/credit as NULLs — plus optional NULL financial rows for the financial table, exercising the NULL handling). N hotels × ~3 years of daily rows.

Demand process: `rooms_sold = round(available_rooms × occupancy(t))`, clipped to [0, available_rooms], where occupancy(t) = base + weekly seasonality + annual seasonality (dry/green seasons) + CR holiday uplift (including Monday-bridge effect from Law 8605) + noise. One hotel gets a simulated multi-week renovation dip in `available_rooms` so models learn a moving ceiling.

Correlational variables are generated with EXACT physical identities so EDA shows realistic correlation structure and no internal contradiction:
- `occupancy_rate = rooms_sold / available_rooms`
- `adr = seasonal base + weekend/holiday premium + noise`
- `room_revenue = rooms_sold × adr`
- `revenue = room_revenue × (1 + ancillary share)` (gastronomy/spa/events)
- `revpar = room_revenue / available_rooms`

Deterministic under a seed.

**Rationale:** the user explicitly requires a fictional-data file for testing; deriving monetary columns FROM room nights preserves the physical identity (revenue ≈ room nights × ADR) so EDA and models see coherent, hotel-realistic correlations. Mirroring the real schema guarantees the same code paths run against SQL later without changes.
**Alternatives considered:** reusing the existing `hotel_data.csv` (rejected: its room nights were a side effect, not capacity-bounded by design); generating on the fly without saving (rejected: the requirement is a persisted file for reproducible offline runs); adding PMS-style columns such as reservations-on-the-books to the synthetic file (rejected: the real tables do not carry them — synthetic columns with no production counterpart create false expectations; captured in the forward-looking roadmap instead).

### 3. NULL handling: aggregate first, impute second
**Decision:** the daily room-nights series is built by grouping operational values by date and summing with NULLs skipped (`groupby.sum()` semantics); imputation (ffill/bfill) runs only afterwards on the aggregated series. Dates with no operational observation remain missing until imputation and are counted in a data-quality report.
**Rationale:** imputing before aggregation would let forward-fill propagate financial NULLs (or, worse, zeros) into the target series — the exact "phantom number" failure this pipeline exists to avoid.
**Alternatives considered:** filtering to table-1 rows only (rejected: the UNION output may not always carry provenance in every environment; NULL-skipping aggregation is robust either way); treating NULLs as 0 (rejected: silently biases demand downward).

### 4. ML feature set: forward-known inputs only (target family + calendar + capacity)
**Decision:** for ML models, features = lags/rollings/expanding of `rooms_sold` + calendar/cyclical + CR holiday features + `available_rooms` (capacity is known/planned in advance by the hotel). All realized monetary columns (`revenue`, `room_revenue`, and by extension `adr`/`revpar`/`occupancy_rate` as observed outcomes) are excluded as inputs.

**The forward-known rule (drives every feature decision):** a variable improves forecast precision only if its future value is known in advance or can itself be forecast with confidence. `occupancy_rate` is a deterministic function of the target and capacity (redundant, future-unknown); realized `adr` is an outcome, not a decision — but a *planned/published rate* would be a legitimate forward-known input if the hotel ever provides a rate calendar (future work).

**Rationale:** recursive forecasting needs every input computable from the past; monetary outcomes are future-unknown. `available_rooms` is operationally plannable and gives trees the ceiling explicitly.
**Alternatives considered:** including realized occupancy_rate/adr as features (rejected: future-unknown — the phantom-input failure this pipeline exists to avoid); imputing monetary features with predictions (rejected: multi-target recursion doubles complexity and error compounding for marginal gain).

### 5. Capacity ceiling: cap predictions, report transparently
**Decision:** after any model produces future forecasts, cap daily values at the available capacity for that date (portfolio: sum of `available_rooms`); report capped dates and magnitudes, and report implied occupancy before capping as a sanity signal.
**Rationale:** room nights are physically bounded; an uncapped forecast above 100% occupancy is not "extra accuracy" — it is a constraint violation that would be caught (and ridiculed) by operations staff. Capping post-prediction keeps models simple and the correction auditable.
**Alternatives considered:** log-transform + capacity-aware models (rejected: complexity without need); no capping (rejected: spec requires the ceiling).

### 6. Point forecasts remain continuous; reporting may round
**Decision:** models predict continuous room nights; the forecast report may round to integers for presentation, but metrics are computed on continuous predictions.
**Rationale:** rounding before metric computation injects quantization noise without improving decisions; integer-ness matters for display, not for accuracy measurement.
**Alternatives considered:** Poisson/Negative-Binomial objectives for GBMs (rejected for now: added complexity; documented as future work if count-specific calibration is ever required).

### 7. New notebook, same structure
**Decision:** create `notebooks/hotel_roomnights_forecasting.ipynb` mirroring the documented structure of the revenue notebook (portfolio aggregate, honest protocol, baseline, DM table, refit-on-full-history forecast) with room-nights units, capacity ceiling and occupancy reporting. The revenue notebook is untouched.
**Rationale:** management said "para iniciar" — room nights first, revenue later; keeping both deliverables isolates the two targets and lets both be demonstrated.
**Alternatives considered:** one parameterized notebook run twice (rejected: outputs would overwrite each other; two notebooks double as two business deliverables).

### 8. TimesFM: same honest exclusion
**Decision:** carry over the platform-availability check: on Windows, TimesFM is excluded from the comparison with a visible note (never run its fallback under its name).
**Rationale:** identical to the revenue change's verified behavior; the spec now captures it explicitly (dl-modeling delta).

### 9. Forward-looking data roadmap (documented, explicitly out of scope)
**Decision:** research into hotel demand drivers identifies a hierarchy of correlational variables. With the current two SQL tables, the achievable precision ceiling is set by the target's own structure + calendar/holidays + capacity. The variables with the highest additional predictive power are forward-looking internal data that requires a PMS/reservations source and is therefore documented as future work, NOT built here:
- **Reservations on the books / pickup curves** (rooms already booked for date t, measured at t-30/t-60/t-90) — the single most powerful forward-looking predictor in hotel revenue management
- **Group/convention blocks** (committed 1-2 years ahead — literally known demand)
- **Planned/published rates** (rate strategy decided months ahead — unlike realized ADR)
- **Cancellation rates by lead window** (nets down gross reservations)
- **Segment mix** (transient vs group vs corporate — different seasonality and lead times)
- **External forward-known signals**: published flight schedules into SJO/LIR, source-market school calendars (US winter = Dec-Feb inbound peak), exchange rates, search interest

**Rationale:** being explicit about this protects the project's credibility: with current data we forecast the best possible honest forecast; the roadmap tells management exactly which data acquisition unlocks the next precision tier instead of leaving an unspoken "why isn't it more accurate?" hanging.
**Alternatives considered:** approximating pickup from the available tables (rejected: no reservation-level data exists in either table — any approximation would be fabricated signal).

## Risks / Trade-offs

- [Real data may have rooms_sold gaps beyond financial NULLs (missing operational days)] → the aggregation reports gap dates; imputation strategy stays configurable; document in ASSUMPTIONS.
- [Capacity changes over time (renovations, closures) break a constant-capacity assumption] → `available_rooms` is read per date when present; future capacity beyond the observed calendar falls back to the last known value and the assumption is flagged in the report.
- [Occupancy >100% in REAL data indicates upstream data errors] → EDA flags them as data-quality issues (spec), never silently "fixed".
- [Synthetic file is mistaken for business truth] → filename, schema docstring and notebook disclaimers state it is fictional; conclusions template carries the same disclaimer as the revenue notebook.
- [Integer rounding hides model bias near capacity] → report implied occupancy before capping so systematic ceiling-bumping is visible.

## Migration Plan

Additive: config keys, generator extension, notebook and tests are added; nothing existing is deleted. Rollback = revert this change's commits; the revenue pipeline keeps working because nothing it depends on was modified in a breaking way.

## Open Questions

1. **SQL availability for real data**: when will credentials be available to run against the real tables? (Affects only when real-data validation happens, not this change.)
2. **Future capacity values**: will hotels provide planned capacity (closures/renovations) for the forecast horizon, or is last-known-capacity the accepted assumption?
3. **Per-hotel vs portfolio as the first business deliverable**: design compares models on the portfolio aggregate (same as revenue change); if management wants per-hotel first, it is a config/notebook adjustment, not an architectural one.
