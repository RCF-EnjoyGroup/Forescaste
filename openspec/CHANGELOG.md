# Changelog

All notable changes to the Hotel Revenue Forecasting Pipeline.

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
