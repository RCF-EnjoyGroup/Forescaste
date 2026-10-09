## Why

The business objective — clarified by operations — is a **reliable 12-month detailed
room-nights forecast**. The current pipeline delivers a 90-day horizon. Extending to
365 days is not a constant change: the as-of booking rule, conformal band calibration,
coverage validation, and books-coverage reporting all depend on the horizon and must
be re-derived honestly for a full year.

## What Changes

- **Deliverable horizon 90 → 365 days**: the production pipeline (and the
  re-validation pipeline) must forecast and report a full 12-month detail per
  property and for the portfolio (daily granularity, occupancy, bands).
- **As-of booking rule tightened to the full horizon** (`multi_step_min_lead: 90 → 365`):
  only buckets with lead ≥ 365 days (`rotb_d365`) are forward-known for every date of a
  365-day forecast; using `rotb_d90/d180` as whole-horizon covariates would be lookahead.
- **Conformal bands calibrated for 365 steps**: rolling-origin windows widen from
  90 to 365 days (12 windows, step 30d — data sufficiency verified for all active
  properties); empirical coverage is validated on the test overlap (213 days) and
  disclosed, since the hold-out is shorter than the horizon.
- **Books coverage reported per as-of-safe window segment**: days 1-90 vs `rotb_d90`,
  91-180 vs `rotb_d180`, 181-365 vs `rotb_d365` — each bucket is only used in the
  segment where it is forward-known (revenue-management-friendly booking-curve view).
- **Slicing guards** in both notebooks: every `[:HORIZON]` slice over the test window
  must use `min(HORIZON, len(test))` — with a 365-day horizon the 213-day test would
  otherwise break band construction.
- **Re-validation artifact coherence**: the re-validation notebook is re-executed at
  the new horizon (its GBM covariate set narrows to `rotb_d365` by the as-of rule).

## Capabilities

### New Capabilities

(none — the change extends existing capabilities)

### Modified Capabilities

- `hotel-revenue-forecasting/forecasting`: the deliverable horizon requirement changes
  (90-day forecast → 365-day detailed forecast), and the booking-books covariate rule
  becomes horizon-complete (only lead ≥ 365 buckets are forward-known across the whole
  horizon; segment-wise finer buckets are reported for context, never used as
  whole-horizon covariates).
- `hotel-revenue-forecasting/evaluation`: uncertainty bands must be calibrated for
  every horizon step up to 365 (365-day rolling windows), and empirical coverage is
  reported on the available test overlap with the limitation disclosed when the
  hold-out is shorter than the horizon.

## Impact

- `config.yaml` (`roomnights_real.horizon_days`, `roomnights_real.multi_step_min_lead`)
- `notebooks/hotel_roomnights_production.ipynb` (weekly production: 365-day forecast,
  segment books coverage, slicing guards) + its HTML report
- `notebooks/hotel_roomnights_realdata.ipynb` (re-validation: slicing guards, 365-day
  covariate rule flows from config; re-executed) + its HTML report
- `CHANGELOG.md` (0.9.0)
- No `src/` code changes — the horizon flows through configuration; helpers
  (`sn365`, guards) are horizon-agnostic and verified for 365-day use
- Risk: the 12-month forecast is less certain than 90 days; bands widen per step and
  this is reported, not hidden. Guards (new hotels) tile a full year from the last
  week — flagged explicitly.
