## Context

The room-nights pipeline (production + re-validation) currently delivers a 90-day
forecast (CHANGELOG 0.8.0): champion `Prophet(0.01,10)` per property, honest floors
(SeasonalNaive, SN365), conformal bands from 12 rolling windows × 90 steps, books
coverage via `rotb_d90`. See proposal.md — Why. The horizon flows through
`config.yaml` (`roomnights_real.horizon_days`), so most mechanics are
horizon-agnostic; the horizon-dependent pieces are: the as-of booking rule
(`multi_step_min_lead`), conformal window length, coverage validation slicing, and
books-coverage reporting.

Data-sufficiency check (verified): 12 rolling windows × 365 days stepping back 30
from 2026-03-07 reach back to a window starting 2024-08-09 — every active property
reports from ≤ 2024-01, so all folds are usable for all active properties.

## Goals / Non-Goals

**Goals**
- Deliver a 365-day detailed (daily) forecast per property and portfolio with
  calibrated bands and per-segment books coverage.
- Keep the honest protocol intact: floors every run, as-of-safe books only,
  empirical coverage reported, guards flagged.
- No `src/` code changes — the horizon flows through configuration.

**Non-Goals**
- Changing the train/val/test split (the 213-day hold-out stays: it measures model
  skill; re-splitting would invalidate all accumulated comparisons).
- Hierarchical reconciliation of bottom-up vs top-down (the gap is reported).
- New model families for the 12-month horizon (the champion and the SN365 floor are
  exactly the yearly-seasonal family; the re-validation trigger policy governs
  future model changes).

## Decisions

**D1 — As-of rule for the full horizon: `multi_step_min_lead: 365`.**
Only `rotb_d365` is forward-known for every date of a 365-day forecast; using
`rotb_d90/d180` as whole-horizon covariates is lookahead beyond their lead windows.
Consequence: re-validation GBMs lose the d90/d180 covariates (they already lose the
comparison — acceptable and honest). The production champion (Prophet) uses no
books covariates, so production forecasts are unaffected; books enter production only
through reporting and the stale/cold safety nets.

**D2 — Books coverage reported per as-of-safe segment.**
Days 1-90 vs `rotb_d90`, 91-180 vs `rotb_d180`, 181-365 vs `rotb_d365`: each bucket
only in the segment where it is forward-known. This is the booking-curve view
revenue management actually uses, and it stays honest under the full-year horizon.

**D3 — Bands: representative folds only; coverage validated on the test overlap.**
*(Amended during implementation — 2026-10-08, evidence below.)* The original plan
(12 windows × 365d) is structurally confounded: every top-down fold model trains
through the 2024-01 composition shift and over-predicts by −100 to −513 rn/day
(measured across all 12 folds), a bias the deployed model (2.75 years of stable
composition) does not share — conformal exchangeability fails and bands landed at
15% (top-down) / 55% (bottom-up) coverage. Amendment:
- **Bottom-up bands** are calibrated ONLY from folds whose training contains ≥1
  full year of the current portfolio composition (window_start ≥ composition
  anchor + 1 year; anchor = latest first-report date among fitted properties →
  2025-01-01 for this data), jointly MC-sampled across properties with
  short-history pools. Measured: **71% empirical coverage** on the 213d overlap
  (consistent with the 90d-era calibration of 70%), width 141 rn/day.
- **Top-down 365d bands are withdrawn from the report** (even recent folds give 9%
  — the portfolio series always contains the shift): the cross-check remains a
  point forecast + consistency gap.
- **Disclosure**: coverage is validated only on the hold-out overlap (213d);
  steps beyond it are fold-calibrated but not hold-out-validated.

**D4 — Slicing guards in both notebooks.**
Every `[:HORIZON]` slice over test-window arrays uses `min(HORIZON, len(actuals))`
and residual matrices are sliced `[:, :n_cov]` to keep shapes aligned
(`per_horizon_conformal_bands` requires residuals steps == forecast length).

**D5 — Guards and floors stretch naturally.**
SN365 (last year's same-calendar shape × damped YoY growth) is the natural 12-month
floor. Short-history guards (SJO) tile the last week for a full year — flagged as
low-confidence 12-month extrapolation (the honest statement; no better information
exists for a hotel with <120 days of history).

**D6 — Re-validation re-executed for artifact coherence.**
With the config change, the re-validation notebook's source implies 365-day behavior;
its executed outputs (90-day era) would be inconsistent. Re-run it once (~45 min):
its GBM books covariates narrow to `rotb_d365` per D1, bands go to 365 steps, and
the comparison/test protocol is otherwise unchanged.

**D7 — Selection folds stay at 90 days; band folds are the 365d representative set.**
*(Added during implementation — 2026-10-08.)* Letting the horizon change flow into
the per-property selection (365-day fold windows) was an unintentional protocol
change with degraded outcomes: at 365-day folds the composite score drifts to
BooksAnchor (books × a ~27× year-ahead completion — a non-stationary multiplier,
the same ramp pathology documented for SJO in 0.8.0), winning FIESTA/Marina/Villas
and degrading the bottom-up test MAE from 55.7 to 100.05 (Villas alone: +63.3% bias).
The decoupling is principled: **selection answers "which method"** — evaluated on
the validated 12 × 90-day fold protocol (0.7.0), where fold models are representative
of the deployment task; **bands answer "how uncertain at 365d"** — the representative
365-day folds of D3, with the winner's residuals recomputed on them (as production
already does). The re-validation notebook is the quarterly runbook: shipping the
degraded selection protocol would repeat the error every cycle.

## Risks / Trade-offs

- **Late-horizon uncertainty is wide** — that is the honest physics of 12-month
  forecasting; bands widen per step and coverage limitations are disclosed (D3).
- **Guard properties extrapolate 52 weeks from one week** — flagged explicitly
  (D5); monitored every run against actuals.
- **GBMs weaken further in re-validation** (d365-only covariates) — they already
  lose the comparison; if the champion's edge vs SN365 ever drops below 5% the
  re-validation trigger fires (existing policy).
