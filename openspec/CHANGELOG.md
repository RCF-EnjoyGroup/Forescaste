# Changelog

All notable changes to the Hotel Revenue Forecasting Pipeline.

## [0.9.0] - 2026-10-09

### Changed — 365-day horizon: 12-month detailed room-nights forecast (OpenSpec `hotel-roomnights-365-horizon`)

- **Deliverable extended 90 → 365 days**: daily detail per property + portfolio for a full year. The same 3 SQL aggregate queries feed it unchanged (verified: pickup leads 0-1065, stay_dates through 2028-12, year-ahead books exist for every horizon date; `rotb_d365` snapshots are all in the past — as-of safe)
- **As-of rule tightened to the full horizon** (`multi_step_min_lead: 365`): only `rotb_d365` is forward-known across all 365 steps; re-validation GBMs now run with the single d365 covariate and degrade accordingly (LightGBM 90.6→96.3, CatBoost 78.1→95.8 — they already lost the comparison); books coverage is reported per as-of-safe segment — **the real booking curve: days 1-90: 66.3% on the books, 91-180: 31.2%, 181-365: 6.5%**
- **D3 amended (implementation evidence)**: 365-day whole-horizon conformal is structurally confounded — every portfolio-fold model trains through the 2024 composition shift and over-predicts by −100..−513 rn/day (measured across all 12 folds; 15% top-down / 55% bottom-up coverage). Fix: bottom-up bands calibrate ONLY on representative folds (training ≥1 full year of the current composition → 3 folds), jointly MC-sampled → **71% coverage** (production); top-down 365d bands withdrawn (cross-check = point + gap); coverage validated on the 213d hold-out overlap with the limitation disclosed
- **D7 added (implementation evidence)**: letting the 365d horizon flow into per-property selection drifted it to BooksAnchor ×27 (a non-stationary year-ahead completion multiplier) winning 3 properties and degrading the bottom-up test MAE 55.7 → 100.05 (Villas +63.3% bias). Fix: selection keeps its validated 12×90-day fold protocol; the winner's residuals are recomputed on the representative 365d folds for the bands
- **Production results (365d)**: **225,602 room nights, implicit occupancy 63.9%**, band 80% 180,677–234,295 (71% empirical coverage on the overlap); champion edge vs floors unchanged (+10.3% vs SN365 → re-validation trigger: NO)
- **Re-validation results (365d)**: Prophet remains champion (63.43 test MAE; SN365 2nd at 68.76 with best bias); per-property winners on the 90d-fold protocol — Corin/FIESTA Prophet, LIREL/Villas Blend(P+SN365), Marina BooksAnchor (fold-legitimate, test-disclosed monitoring point); bottom-up test 56.45 (+7.4% bias); **future 365d 226,737 (+0.5% vs production — both pipelines agree)**; bands 196,853–243,380 (62% coverage)
- Both notebooks executed end-to-end at 365d (29 cells each, 0 errors); HTML reports regenerated; 73 tests green

## [0.8.0] - 2026-10-08

### Added — Champion/challenger production architecture (slim weekly pipeline)

- **New production notebook** (`hotel_roomnights_production.ipynb`, executed end-to-end, ~3 min vs ~45): fixed champion `Prophet(0.01,10)` per property (no selection logic in production), SeasonalNaive guard <120 fit days, frozen/stale/cold safety nets, bottom-up operational forecast, conformal bands (12 post-2024 windows, joint fold-aligned MC), top-down cross-check, and **honest floors reported every run** (SeasonalNaive weekly + SN365) with the champion's edge and the re-validation trigger status (edge <5% vs SN365 → run the full bake-off; plus quarterly cadence and composition-event triggers)
- **Rationale (validated)**: the 0.7.0 selection machinery (5 candidate families, composite scores, margins, ~370 fits) bought only 3.3% of test MAE (53.96 vs 55.73) and measurement showed selection designs could underperform a fixed champion (selection overfit) — the fixed model is more robust, explainable in one page, and reproduces the diagnostic exactly (55.73 ✓)
- **Production results (this run)**: 54,561 room nights 90d, implicit occupancy 62.6%, band 80% 47,780–61,438 with **70% empirical test coverage** (best of the session; trajectory 46%→64%→66%→68%→70%), top-down cross-check +1.8%, books 66.3% already reserved, **edge +10.3% vs SN365 → re-validation trigger: NO**
- **Fixed in production**: capacity lookup used the CSV's last row (year-2099 partial inventory = 222 rooms) instead of the last observed date (968) — produced absurd occupancy/cross-check numbers; now anchored at the last observed day with fallback
- **Label bug found & fixed (re-validation notebook, source)**: the 0.7.0 executed table showed `Prophet(...)` where the actual per-property method was `Blend(P+SN365)` (LIREL's 17.90 was the blend; pure Prophet = 21.46) — numbers were always correct, the method name was not; takes effect on the next re-validation run
- The full re-validation notebook remains unchanged as the challenger experiment (9-model comparison, DM, rolling backtest, per-property composite selection); README documents both artifacts and the trigger policy
- 73 tests green

## [0.7.0] - 2026-10-08

### Changed — Specialist round 2: bias-aware selection, seasonal & books candidates

- **Diagnosed the remaining +8.93% bottom-up bias** (per-property decomposition + level analysis): the pre-test window ends at the seasonal PEAK (Dec-Feb) while the test window is Mar-Oct — every recent-level anchor over-forecasts the high→low transition; per-prop Prophet grids estimate yearly seasonality on <3 cycles; **books fix LIREL** (test bias +12.6% → −2.2%); portfolio is GROWING YoY (+4-14%) — the issue was seasonality, not demand softening
- **Composite-score per-property selection**: fold MAE + |fold MBE| (rooms/day) — for staffing, systematic drift is operationally costly; fold-MAE-only selection ignored it. Candidates (fold evidence ONLY, test stays pure evaluation): Prophet grid, **SN365** (last year's same-calendar shape × damped YoY growth — the missing seasonal baseline), **BooksAnchor** (rotb_d90 × recent-120d completion), **Blend50** (0.5·Prophet + 0.5·SN365), SN7 floor (≥10% fold-MAE margin to displace)
- **SeasonalNaive365 added to the global comparison** (9 models): debuts 2nd (MAE 68.76 vs Prophet 63.43) with the best calibration of the table (−2.51% bias); Prophet's DM edge vs SN365 is not significant ("supera a 7/8") — honest
- **Guard hardened with documented evidence**: BooksAnchor is invalid for short-history ramps — SJO's realized/books_d90 ratio is ~8× (portfolio 2.68×) because new-hotel booking curves are extremely back-loaded; the ratio is non-stationary and produced implausible ~4×-level futures (occ 0.64-0.73 vs settled ~0.17). Guard reverted to recent-level weekly tiling with the a-priori rationale in-code
- **Results (active portfolio)**: **bottom-up test MAE 53.96, WAPE 9.11%, bias +6.69%** (was 62.21/+8.93%; top-down Prophet 63.43/10.71%/+3.02% → bottom-up wins by 14.9% MAE). Per-property winners: Corin Prophet(0.01,1.0) 7.74/+1.25%; FIESTA Prophet(0.01,10) 37.21/−1.31%; LIREL Prophet(0.01,10) **17.90/+5.59%** (was 21.50/+12.58%); Marina Prophet(0.1,1.0) **5.99/+14.3%** (was SN7 10.58/+37.2%); SJO SN-guard 18.24 (ramp — irreducible from 60 pre-test days; future anchored at its settled ~26/day, occ 16.5%); Villas Prophet(0.05,10) 25.28/+25.5% — **monitoring point**: the 2026 softening was invisible pre-test AND in books
- **Operational 90d: 54,299 room nights, implicit occupancy 62.3%, +2.5% vs top-down cross-check** (52,950) — consistency restored (was +9-10% during the guard bug); band 80%: 47,709–61,242 with **68% empirical coverage** (trajectory: 46% → 64% → 68%; nominal 80%, documented path: more windows with more history); books coverage 66.6%
- 73 tests green; notebook executed end-to-end (29 cells, 0 errors)

## [0.6.0] - 2026-10-08

### Changed — Specialist validation: bottom-up operational forecast, per-property fold-selected models, regime-aware conformal

- **Diagnosed root cause** (validated on test, exact protocol replication): the "portfolio" series is a moving composition — hotels joined at 2023-01 (+105), 2024-01 (**+517/day, the portfolio quadrupled with FIESTA+VILLAS**) and 2026-01 (SJO). The 4-year window spans 3 different portfolios: GBMs win val (45-52) and folds (CatBoost rank 1.5, 4/6) but collapse on test (78-93, recursive drift +13.9% bias); Prophet is regime-robust (folds ≈ test at 63.4); the 6-fold conformal pooled a calmer regime (46% coverage, and no scale inflation can fix it — coarse quantiles + shape, max 51%)
- **Bottom-up is now the OPERATIONAL portfolio forecast** (sum of per-property forecasts): composition-immune by construction, beats top-down on test, and is what operations needs anyway; top-down stays as an honest cross-check
- **Per-property model selection via 12 pre-test folds** (Prophet grid cps∈{0.01,0.05,0.1}×sps∈{1,10} vs SeasonalNaive floor with a ≥10% margin to displace the structural model — margin set a priori, never tuned on test). Test stays a pure evaluation
- **Short-history guard** (<120 fit days → SeasonalNaive, flagged): SJO had only 60 pre-test days — Prophet exploded at +896% bias / MAE 233; the guard gives MAE 18.24
- **Regime-aware conformal**: winner-only calibration on 12 windows (step 30d, all post-2024 composition) + for the bottom-up, joint fold-aligned Monte Carlo of each property's own fold residuals plus short-history pools — composition-correct by construction
- **Results**: per-property winners — Corin/LIREL Prophet(0.01,1.0), FIESTA/Villas Prophet(0.01,10), **Marina SeasonalNaive** (folds prefer it ≥10%; test disagrees — 10.6 vs 4.9 — disclosed, monitoring point for the next export), SJO SeasonalNaive via guard. Bottom-up test MAE 62.21 vs top-down 63.43 (Prophet, +52.9% vs naive, DM-significant 7/7). **Operational 90d: 55,127 room nights, implicit occupancy 63.3%, band 80% 48,259–62,190 with 64% empirical coverage** (was 46%); top-down cross-check 52,950 (+4.1% gap) with a conservative 94%-coverage band; books coverage 65.6%; frozen LAPAS/LIRAK excluded (ops-confirmed)
- Honest disclosures kept: per-property table with methods/bias, empirical coverages, stability (Prophet rank 3.7/8, 1/6 fold wins). Improvement path documented: more history → more windows → finer conformal quantiles; per-prop books regressors (Villas bias +26.6%) as next step
- 73 tests green; notebook executed end-to-end (29 cells, 0 errors)

## [0.5.1] - 2026-10-08

### Changed — Frozen properties excluded (LAPAS, LIRAK) from the active portfolio

- **Ops-confirmed exclusion**: `roomnights_real.frozen_properties: [lapas, lirak]` in `config.yaml` — closed/paused properties are shown in the quality report (with a `frozen` column) but dropped from demand/pickup/capacity frames right after it; unknown frozen keys fail loudly. The stale/cold-start books-estimate machinery stays as a safety net for future exports where an *active* property goes stale
- **Rationale**: frozen hotels' stale history contaminated portfolio patterns (the 0.5.0 run included books-based estimates for their non-operable future); the active portfolio is now 6 properties / 8,851 property-days (2016-02 → 2026-10)
- **Results (active portfolio)**: winner flips to **Prophet** — MAE 63.4 rooms/day, WAPE 10.7%, +52.9% vs SeasonalNaive, DM Newey-West significant vs all 7 alternatives (LightGBM, the previous winner on the contaminated series, falls to MAE 90.6 with +13.9% bias); backtest mean rank 3.7/8 with 1/6 window wins (disclosed)
- 90-day forecast: 52,950 room nights, implicit occupancy 60.8%; books coverage 68.3% already reserved; per-property (6 evaluated, mean MAE 54.8) sums to +5.2% vs the portfolio model
- **Honest disclosure**: conformal 80% bands' empirical test coverage drops to 46% with the Prophet-winner residuals (6 backtest windows give coarse quantiles — 0.3.2's improvement path stands: more windows or moving-block bootstrap)

## [0.5.0] - 2026-10-08

### Added — Real-data production run + honest stale-property handling

- **Executed on the REAL systems exports** (`data/roomnights_real/`: realized demand 2013-09→2026-10 across 13,874 property-days / 8 properties, pickup 1.57M rows, capacity): `hotel_roomnights_realdata.ipynb` runs end-to-end unchanged (28 cells, 0 errors) — the 0.4.0 schema work validated against production data
- **Honest stale-property handling**: realized-demand freshness audit per property; **LAPAS stopped reporting 2026-07-04 and LIRAK 2025-06-30** (their books continue) — per-property test evaluation only on each property's OWN available dates (LIRAK: 0 test days → NaN row, no fake metrics; LAPAS: 119/213 days); 90d forecast: fresh properties use the winner model, stale/cold-start use a transparent books-based estimate (current books × learned pickup-completion factor 2.68×, capacity-capped) with explicit per-property status flags; stale warning surfaced in conclusions for systems follow-up (closure/sale?)
- **Results (real data)**: **LightGBM wins** (MAE 54.4 rooms/day, WAPE 9.2%, +62.4% vs SeasonalNaive; DM Newey-West significant vs all 7 alternatives); 90-day portfolio forecast 57,790 room nights (implicit occupancy 59.0%, capacity 1,089); conformal 80% band 50,896–69,756 with **test coverage 86% vs 80% nominal**; books coverage 62.6% of the forecast already reserved; per-property sum −6.7% vs portfolio direct (7 properties evaluated, mean MAE 19.9)
- Honest stability disclosure: rolling-origin backtest shows LightGBM mean rank 4.2/8 with 1/6 window wins — the test-split winner is not uniformly dominant across historical regimes
- Fixes: `evaluation/evaluator.py` empty-observation metrics now include Bias%/Over_days/Under_days (consistent keys, fixes `KeyError: 'Bias%'` on zero-test-day properties); notebook per-property cell guards zero-test-day properties and its narrative now reflects the real data (no synthetic-generator claims)
- Data quality: rotb_d90 vs realized correlation 0.370 on real data (booking behavior is noisier than the synthetic 0.93 — expected); SJOSL now has 273 days of history (no cold-starts remain); 92 historical gaps all pre-2022 (4-year training window contiguous)

## [0.4.0] - 2026-10-07

### Added — Real production sources (re-target with pickup/capacity)

- **Real-schema ingestion** (`data/real_sources.py`): three SQL pushdown aggregate queries (realized demand by `snap_flag=0` excluding SYSTEM_ADJUST; pickup by `snap_flag=1` grouped by lead days; capacity summing `oficial_inventory`) — the ~14M raw snapshot rows never leave the database; CSV-export workflow with locale-safe revenue parsing ("329,6" -> 329.6), flag-0 uniqueness assertion and capacity last-known fallback
- **Canonical property mapping** (`data/property_map.py`): all observed variants (codes + "Hotel Royal Corin") resolve through one config dictionary; unknown identifiers fail loudly
- **On-the-books (pickup) features** (`data/pickup.py`): `rotb_d{7..365}` by lead bucket with the as-of rule — for a 90-day multi-step window only leads >= 90 are forward-known; no-lookahead unit-tested; portfolio aggregation supported
- **Per-date future covariates in recursive forecasting**: `recursive_forecast` accepts `future_covariates` (books vary by future date, unlike constant capacity); incremental path verified against reference; `tune_gbm` optimizes the recursive val MAE with covariates
- **Synthetic generator v2** (`data/synthetic_real.py`): mirrors the REAL schemas (snapshot semantics flag 0/1 with lead-dependent arrival curves, per-room-type capacity table, cold-start property, sparse-inventory stretch) — 8 properties, raw identifiers, deterministic by seed; exports the exact aggregate CSV format systems will produce
- **Real-data notebook** (`hotel_roomnights_realdata.ipynb`, executed end-to-end + HTML): EDA with pickup curves (rotb_d90 vs realized corr **0.93**), honest protocol (baseline/recursive/backtest 6x90d/DM Newey-West/bias/weekday), 90-day forecast with current books as covariates, conformal bands with coverage report, per-property forecasts with capacity ceilings, cold-start SJOSL via books × learned completion (flagged), books-coverage summary (43.3% of the forecast already on the books)
- Results (real-schema synthetic): **XGBoost wins** (MAE 33.7 rooms/day, WAPE 3.8%, +55.3% vs naive; backtest mean rank 1.8/8); bias −2.24%; per-property sum vs portfolio +3.2%
- Tests: 73 total (19 new — property mapping, locale parsing, flag semantics, as-of no-lookahead, capacity fallback, covariate recursion equivalence, generator determinism)

## [0.3.2] - 2026-10-06

### Added — Review hardening: backtesting, honest tuning, conformal bands, per-hotel

- **Rolling-origin backtesting** (`evaluation/backtesting.py`): 6 historical windows × 90 days; each fold sees only strictly pre-origin data (no-leakage unit-tested); per-fold MAE, winner-per-fold, mean-rank stability. Evidence: LightGBM mean rank 1.8/8, 3/6 wins — ranking no longer rests on a single split
- **Honest hyperparameter tuning** (`models/tuning.py`): Optuna (25 trials/family) whose objective is the **recursive multi-step MAE on the validation window** — never a one-step shortcut; Prophet grid search likewise. **Tuning flipped the winner from CatBoost to LightGBM** (MAE 15.2→14.3, bias −1.97%→−1.13%): the ranking was parameter-sensitive, proving the tuning investment necessary
- **Conformal uncertainty bands** (`evaluation/conformal.py`): per-horizon-step P10/P90 built from the winner's pooled backtest residuals, physically clamped (≥0, ≤capacity). 90-day total: 26,435–29,450 room nights around 27,169. **Empirical coverage on test: 60% vs 80% nominal** — reported honestly; 6 windows give coarse quantiles (improvement path: more windows or moving-block bootstrap)
- **Per-hotel evaluation & forecast** (notebook Section 8.5): winner re-fitted per hotel (frozen hyperparams); per-hotel test MAE avg 5.3 room nights; per-hotel 90-day forecasts with implied occupancy; portfolio-vs-sum-of-hotels gap **+0.9%** (documented, no hierarchical reconciliation)
- Tests: 54 total (backtesting no-leakage spy, rank discrimination, conformal coverage/width-monotonicity, tuning smoke, bias)
- Notebook re-executed end-to-end (57 cells, 0 errors) + HTML report

## [0.3.1] - 2026-10-02

### Added — Forecast direction & segment diagnostics (Lighthouse MAPE guidance)

- **Signed bias metrics** in every evaluation: `Bias%` (positive = over-forecasting, negative = under-forecasting) plus over/under day counts — magnitude metrics alone don't reveal systematic pessimism/optimism
- **Per-day-of-week error breakdown** for the best model in the room-nights notebook (MAE, MAPE, signed bias per weekday) — operational signal for staffing/pricing (weekend vs midweek weakness localization)
- **Industry MAPE benchmarks documented** (ASSUMPTIONS.md) with the honest caveat: published benchmarks measure short-horizon forecasts; this pipeline's multi-step protocol is a stricter regime — the meaningful benchmark is the pipeline's own trend over time
- Tests: 44 total (bias metrics: sign symmetry, over/under counts, perfect-calibration zero)

## [0.3.0] - 2026-10-02

### Added — Room-Nights Forecasting (primary target per management decision)

- **Target pivot**: room nights (`rooms_sold`) as the primary forecasting target; the revenue pipeline remains fully intact
- **Synthetic test-data file** (`data/hotel_roomnights_sample.csv`): 5 hotels × 3 years, capacity-bounded integer room nights with weekly/annual seasonality, CR holiday uplift (incl. Monday bridges), one simulated renovation, 785 financial rows with NULL target (exercising NULL handling), and exact physical identities (occupancy/ADR/room_revenue/revenue/revpar) — deterministic by seed
- **NULL-safe daily aggregation** (`data/aggregation.py`): financial-table rows with NULL `rooms_sold` are excluded BEFORE imputation; gap dates reindexed; data-quality report
- **Forward-known rule in ML inputs**: monetary-family features excluded (future unknown); `available_rooms` carried through recursive forecasting with its last observed (planned) value
- **Capacity ceiling**: future forecasts capped at available capacity with capped-dates reporting; implied occupancy (position-based) reported
- **Schema validation by target**: `rooms_sold` required and validated when it is the active target
- **Notebook** `hotel_roomnights_forecasting.ipynb` (documented, executed end-to-end) + HTML report
- Tests: 43 total (15 new — NULL handling, generator determinism/identities/correlations, capacity ceiling, forward-known passthrough equivalence)
- Results (synthetic data, honest multi-step protocol): **CatBoost wins** (MAE 14.3 room nights/day, WAPE 3.96%, 71.3% better than SeasonalNaive); DM-significant over Prophet/ETS/SARIMAX/PatchTST/naive

## [0.2.1] - 2026-10-01

### Performance — Incremental Recursive Forecasting

- **`recursive_forecast` fast path**: history features computed once; each future row built with O(1)/O(window) lookups (lags, rolling, expanding, calendar, CR holidays) instead of re-transforming the full context per step
  - **27.8x speedup** at notebook scale (937-day history, 158 forecast steps, 46 features): 14.5s → 0.5s per model, with **identical predictions** (max diff = 0)
  - **Self-verification**: the incremental path is validated feature-by-feature against the reference transform on the first step and falls back automatically on any mismatch — speed never compromises correctness
  - Reference path preserved (`use_incremental=False`) as auditable ground truth
  - Per-step INFO log spam from feature/holiday builders silenced during recursion (restored afterwards)
- Tests (28 total): step-by-step feature equivalence, end-to-end prediction agreement, and silent-fallback for unsupported configs

## [0.2.0] - 2026-09-29

### Fixed — Honest Evaluation Protocol (anti "números falsos")

- **No target scaling**: metrics now reported in real dollars (previously z-scores)
- **Date-aligned prediction** for SARIMAX/ETS: forecasts start at the last observed day and are indexed by requested date (previously misaligned ~194 days)
- **Recursive multi-step ML evaluation** (`src/forecasting/forecasting/recursive.py`): GBM test predictions feed back as lags — test actuals are never model inputs (previously one-step-ahead with true lags = optimistic fake metrics)
- **GBM 3-step protocol**: early stopping on val → `final_refit` on all pre-test history with fixed best iteration → recursive forecast
- **PatchTST**: stored real training context (previously predicted from zeros), multi-window recursive forecasting for horizons > 30, fixed 4-D tensor bug in the transformer forward pass
- **TimesFM**: honest seasonal-naive fallback (previously explosive compounding); excluded from comparison on Windows (requires Linux/JAX)
- **XGBoost**: early stopping now actually activates (constructor param for xgboost>=2.0)
- **Diebold-Mariano**: Newey-West HAC variance (lag h-1) for autocorrelated multi-step errors
- **plot_outliers**: boolean-mask alignment (crashed with duplicated date index)
- **Prophet/Windows**: pinned `cmdstanpy<1.3` (bundled cmdstan validation)

### Added

- Seasonal Naive baseline in the comparison table (a model must beat it)
- Portfolio aggregation: all models evaluated on the same daily series
- Leakage tests: SARIMAX/ETS date alignment, recursive no-leakage, PatchTST fallback (25 tests total)
- `holidays` and `pmdarima` dependencies (real CR holidays, working SARIMAX)
- Notebook executed end-to-end; honest conclusions with synthetic-data disclaimer

## [0.1.0] - 2026-08-26

### Added

- **Project setup**: `pyproject.toml`, `requirements.txt`, `.gitignore`, `config.yaml`
- **Data Ingestion Module** (`src/forecasting/data/`):
  - `DataLoader` class with SQL connection and CSV fallback
  - Schema validation and NULL column detection
  - Synthetic sample data generator for offline development
- **EDA Module** (`src/forecasting/eda/`):
  - Descriptive statistics with skewness and kurtosis
  - Missing value analysis
  - Time series decomposition (additive/multiplicative STL)
  - Stationarity tests (ADF, KPSS)
  - Autocorrelation analysis (ACF/PACF)
  - Seasonality detection (weekly, monthly, yearly)
  - Outlier detection (IQR, z-score)
  - Correlation analysis with heatmaps
  - `EDAVisualizer` for all plot types
- **Preprocessing Module** (`src/forecasting/preprocessing/`):
  - `MissingValueHandler` (ffill/bfill, interpolation)
  - `OutlierHandler` (winsorization, flagging)
  - `TemporalFeatureEngineer` (calendar, cyclical, lag, rolling, expanding)
  - `HolidayFeatureEngineer` (Costa Rica holidays + custom events)
  - `TemporalSplitter` (expanding window with configurable ratios and gap)
  - `FeatureScaler` (Standard, MinMax, Robust)
  - `PreprocessingPipeline` composing all steps
- **Statistical Models** (`src/forecasting/models/statistical/`):
  - `ProphetForecaster` with Costa Rica holidays
  - `SARIMAXForecaster` with auto_arima parameter selection
  - `ETSForecaster` with automatic model selection (AIC)
- **ML Models** (`src/forecasting/models/ml/`):
  - `LightGBMForecaster` with categorical support and early stopping
  - `XGBoostForecaster` with early stopping
  - `CatBoostForecaster` with native categorical features
- **DL Models** (`src/forecasting/models/dl/`):
  - `TimesFMForecaster` wrapper for Google TimesFM (zero-shot/fallback)
  - `PatchTSTForecaster` with Transformer encoder and patching
- **Evaluation Module** (`src/forecasting/evaluation/`):
  - Core metrics: MAE, RMSE, MAPE, sMAPE, MASE, WAPE
  - Diebold-Mariano test for statistical significance
  - Residual diagnostics (Jarque-Bera, Ljung-Box, Breusch-Pagan)
  - Bootstrap confidence intervals
  - `EvaluationVisualizer` (forecast plots, scatter, error distribution, Q-Q)
- **Forecasting Module** (`src/forecasting/forecasting/`):
  - `FutureForecaster` with multi-hotel support
  - Aggregated portfolio forecasting
  - Assumptions, limitations, and recommendations documentation
- **Jupyter Notebook** (`notebooks/hotel_revenue_forecasting.ipynb`):
  - Complete 10-section workflow
  - All models trained and compared
  - Publication-ready visualizations
- **Documentation**:
  - `README.md` with setup and usage instructions
  - `ASSUMPTIONS.md` documenting all forecasting assumptions
  - `CHANGELOG.md` for version tracking
