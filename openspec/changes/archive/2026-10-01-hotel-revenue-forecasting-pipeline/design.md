## Context

The project requires building a comprehensive time series forecasting pipeline for hotel revenue prediction. Data comes from a SQL query combining `test_enjoy_fac_hotel` and `enjoy_fac_hotel_financial` tables via UNION ALL. The pipeline must be delivered as a well-structured Jupyter notebook following time series best practices.

Key constraints:
- Single data source (the SQL UNION query)
- Must predict both Revenue and Room Revenue
- Must include statistical (Prophet, SARIMA, ETS), ML (LightGBM, XGBoost, CatBoost), and DL (TimesFM + one more) models
- Temporal splits only - no random splits
- Reproducibility with seeds and version pinning
- Spanish language for documentation/comments

## Goals / Non-Goals

**Goals:**
- Single Jupyter notebook (`hotel_revenue_forecasting.ipynb`) containing complete workflow
- Modular Python package structure for reusable components (`src/forecasting/`)
- Comprehensive EDA with statistical tests and visualizations
- Proper temporal train/val/test splits (e.g., 70/15/15 or 80/10/10)
- All required model families implemented and compared
- Rigorous evaluation with MAE, RMSE, MAPE, sMAPE, MASE, WAPE
- Future forecast generation with uncertainty quantification
- Clear documentation of assumptions, limitations, and recommendations

**Non-Goals:**
- Production deployment / MLOps pipeline
- Real-time inference service
- Automated model retraining scheduler
- External data integration (weather, events, competitor data) - documented as future work
- Hierarchical/multi-level forecasting (hotel → region → portfolio) - documented as future work
- Web dashboard or UI

## Decisions

### 1. Notebook-First with Modular Package Structure
**Decision:** Deliver as a single comprehensive Jupyter notebook backed by a modular `src/forecasting/` package.

**Rationale:** The deliverable is explicitly a notebook, but modular code enables testing, reuse, and cleaner notebook cells. The notebook imports from `src/forecasting/` keeping cells focused on workflow orchestration.

**Alternatives considered:**
- Pure notebook (no package): Rejected - hard to test, reuse, leads to copy-paste
- Pure Python scripts: Rejected - doesn't meet "notebook deliverable" requirement

### 2. Data Loading Strategy
**Decision:** Abstract SQL loading behind a `DataLoader` class with configurable connection. Provide CSV fallback for reproducibility.

**Rationale:** SQL credentials may not be available in all environments. CSV export allows offline development and CI/CD.

**Alternatives considered:**
- Direct SQL in notebook: Rejected - security risk, not reproducible
- Only CSV: Rejected - doesn't match "connect/load from query" requirement

### 3. Temporal Split Strategy
**Decision:** Use expanding window with configurable ratios. Default: train=70%, val=15%, test=15% by date. Add 7-day gap between splits to reduce leakage.

**Rationale:** Hotel revenue has strong seasonality; expanding window mimics real-world forecasting where history grows. Gap reduces leakage from autocorrelation.

**Alternatives considered:**
- Sliding window: Rejected - wastes data, less realistic for growing history
- Fixed date splits: Rejected - not adaptive to data length

### 4. Multi-Hotel vs Aggregated Forecasting
**Decision:** Support both per-hotel and aggregated modeling. Default to per-hotel with global model (hotel_id as feature) for ML/DL, separate models for statistical.

**Rationale:** Different hotels may have different patterns. Per-hotel statistical models capture individual seasonality; global ML models share statistical strength.

**Alternatives considered:**
- Only aggregated: Rejected - loses hotel-level insights
- Only per-hotel: Rejected - insufficient data for some hotels, no sharing

### 5. Model Implementation Choices

| Model Family | Library | Key Configuration |
|--------------|---------|-------------------|
| Prophet | `prophet` | Yearly/weekly seasonality, Costa Rica holidays, changepoint_prior_scale=0.05 |
| SARIMA/SARIMAX | `statsmodels` | Auto ARIMA with seasonal periods [7, 365], stepwise search |
| ETS | `statsmodels` | Auto model selection via AIC, damped trend |
| LightGBM | `lightgbm` | Objective=regression, metric=mae, early_stopping=50, categorical=hotel_id |
| XGBoost | `xgboost` | Objective=reg:squarederror, early_stopping=50 |
| CatBoost | `catboost` | Loss=MAE, cat_features=[hotel_id], early_stopping=50 |
| TimesFM | `timesfm` | Pre-trained 200M/500M, context_len=512, horizon=configurable |
| PatchTST | Custom/`transformers` | Patch len=16, stride=8, Transformer encoder, 3 layers |

**Rationale:** Covers all required families. TimesFM is mandatory per requirements. PatchTST selected as second DL model for strong benchmark performance on time series.

**Alternatives considered:**
- Chronos: Good but requires more dependencies (T5 tokenizer)
- LSTM: Weaker baseline than PatchTST for this data scale

### 6. Feature Engineering Pipeline
**Decision:** Use scikit-learn compatible `FeatureEngineer` class with fit/transform. Features: calendar (sin/cos), lags [1,7,14,30,365], rolling [7,14,30,90], expanding. All shifted by 1 to prevent leakage.

**Rationale:** sklearn compatibility enables pipeline composition and cross-validation. Comprehensive feature set covers known hotel revenue drivers.

### 7. Evaluation Framework
**Decision:** Implement `Evaluator` class computing all metrics with bootstrap confidence intervals. Use `TimeSeriesSplit` (n_splits=5, expanding) for CV.

**Rationale:** Single split is unreliable for time series. Bootstrap CIs quantify uncertainty in metric estimates. TimeSeriesSplit respects temporal order.

### 8. Reproducibility
**Decision:** Pin all versions in `requirements.txt` and `pyproject.toml`. Set global seeds (numpy, torch, python, lightgbm, xgboost, catboost). Log environment info in notebook.

**Rationale:** Time series results are sensitive to random initialization and library versions. Full pinning ensures exact reproduction.

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient data per hotel for statistical models | Medium | High | Fall back to global models; use hierarchical pooling |
| TimesFM/PatchTST import/installation failures | Medium | High | Provide fallback to local model files; document install steps |
| Strong seasonality not captured by default model configs | High | Medium | Explicit seasonality configuration; validate with ACF/PACF |
| Data leakage from feature engineering | Low | Critical | Shift all features by 1; unit test for leakage |
| Non-stationary revenue trends breaking SARIMA | Medium | High | Use differencing; consider Prophet/ETS as more robust |
| Holiday calendar mismatch (Costa Rica vs data) | Low | Medium | Validate holidays against data; allow custom calendar |
| Computational cost of DL models on CPU | High | Medium | Default to smaller models; allow GPU flag; cache predictions |
| MAPE/sMAPE instability with low revenue values | Medium | Medium | Report WAPE/MASE as primary; flag MAPE when mean < threshold |

## Migration Plan

Not applicable - this is a new analysis project, not a migration.

## Open Questions

1. **Data granularity**: Is the data daily, weekly, or monthly? (Affects seasonality periods, lag choices)
2. **Number of hotels**: How many unique hotel_ids? (Affects per-hotel vs global strategy)
3. **Date range**: What is the start/end date of the data? (Affects split ratios, horizon)
4. **SQL access**: Will the notebook run with live SQL connection or only CSV export?
5. **Forecast horizon**: What is the business requirement? (30 days? 90 days? 365 days?)
6. **Compute environment**: GPU available for TimesFM/PatchTST? (Affects model size selection)
7. **Holiday calendar**: Should we use Costa Rica national holidays or hotel-specific events?
8. **Missing financial columns**: What are the 4 NULL columns from `enjoy_fac_hotel_financial`? Do they have business meaning?