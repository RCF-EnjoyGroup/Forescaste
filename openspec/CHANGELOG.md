# Changelog

All notable changes to the Hotel Revenue Forecasting Pipeline.

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
