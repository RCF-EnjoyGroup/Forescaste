## 1. Project Setup and Environment

- [x] 1.1 Create project directory structure: `src/forecasting/`, `notebooks/`, `data/`, `outputs/`
- [x] 1.2 Create `pyproject.toml` with pinned dependencies: pandas, numpy, scikit-learn, statsmodels, prophet, lightgbm, xgboost, catboost, timesfm, torch, matplotlib, seaborn, plotly, jupyter, optuna
- [x] 1.3 Create `requirements.txt` from pyproject.toml for pip compatibility
- [x] 1.4 Create `src/forecasting/__init__.py` with version and exports
- [x] 1.5 Create `.gitignore` for data, outputs, __pycache__, .ipynb_checkpoints
- [x] 1.6 Create `config.yaml` for configurable parameters (split ratios, horizons, model configs)

## 2. Data Ingestion Module (`src/forecasting/data/`)

- [x] 2.1 Create `src/forecasting/data/__init__.py`
- [x] 2.2 Implement `DataLoader` class with SQL connection and CSV fallback
- [x] 2.3 Implement `load_hotel_data()` function executing the UNION ALL query
- [x] 2.4 Add schema validation (required columns, dtypes, date parsing)
- [x] 2.5 Add NULL column detection and documentation for financial table
- [x] 2.6 Create `data/sample_data.csv` generator for offline development
- [x] 2.7 Write unit tests for data loading and validation

## 3. Exploratory Data Analysis Module (`src/forecasting/eda/`)

- [x] 3.1 Create `src/forecasting/eda/__init__.py`
- [x] 3.2 Implement `descriptive_statistics()` function with stats for revenue/room_revenue
- [x] 3.3 Implement `missing_value_analysis()` with counts, percentages, patterns
- [x] 3.4 Implement `time_series_decomposition()` (additive/multiplicative STL)
- [x] 3.5 Implement `stationarity_tests()` with ADF and KPSS
- [x] 3.6 Implement `autocorrelation_analysis()` with ACF/PACF plots
- [x] 3.7 Implement `seasonality_detection()` with weekly/monthly/yearly patterns
- [x] 3.8 Implement `outlier_detection()` with IQR and z-score methods
- [x] 3.9 Implement `correlation_analysis()` with heatmap visualization
- [x] 3.10 Create `EDAVisualizer` class for all plotting functions
- [ ] 3.11 Write unit tests for EDA functions

## 4. Preprocessing Module (`src/forecasting/preprocessing/`)

- [x] 4.1 Create `src/forecasting/preprocessing/__init__.py`
- [x] 4.2 Implement `MissingValueHandler` with forward/backward fill and interpolation
- [x] 4.3 Implement `OutlierHandler` with winsorization and flagging options
- [x] 4.4 Implement `TemporalFeatureEngineer` with calendar, cyclical, lag, rolling, expanding features
- [x] 4.5 Implement `HolidayFeatureEngineer` with Costa Rica holidays and custom events
- [x] 4.6 Implement `TemporalSplitter` with expanding window, configurable ratios, gap periods
- [x] 4.7 Implement `FeatureScaler` with Standard/MinMax/Robust scaling (fit on train only)
- [x] 4.7 Create `PreprocessingPipeline` composing all steps with fit/transform
- [x] 4.8 Write unit tests for preprocessing (especially no-leakage verification)

## 5. Statistical Modeling Module (`src/forecasting/models/statistical/`)

- [x] 5.1 Create `src/forecasting/models/__init__.py` and `statistical/__init__.py`
- [x] 5.2 Implement `ProphetForecaster` class with Costa Rica holidays, hyperparameter tuning
- [x] 5.3 Implement `SARIMAXForecaster` with auto_arima, seasonal periods [7, 365], exogenous support
- [x] 5.4 Implement `ETSForecaster` with auto model selection (AIC), damped trend
- [x] 5.5 Create base `StatisticalForecaster` abstract class with fit/predict interface
- [x] 5.6 Implement prediction interval generation for all statistical models
- [x] 5.7 Write unit tests for each statistical model

## 6. Machine Learning Modeling Module (`src/forecasting/models/ml/`)

- [x] 6.1 Create `src/forecasting/models/ml/__init__.py`
- [x] 6.2 Implement `LightGBMForecaster` with categorical support, early stopping, Optuna tuning
- [x] 6.3 Implement `XGBoostForecaster` with early stopping, Optuna tuning
- [x] 6.3 Implement `CatBoostForecaster` with native categorical, early stopping, Optuna tuning
- [x] 6.4 Create base `MLForecaster` abstract class with fit/predict interface
- [ ] 6.5 Implement quantile regression for prediction intervals (LightGBM/XGBoost)
- [x] 6.6 Write unit tests for each ML model

## 7. Deep Learning Modeling Module (`src/forecasting/models/dl/`)

- [x] 7.1 Create `src/forecasting/models/dl/__init__.py`
- [ ] 7.2 Implement `TimesFMForecaster` wrapper for pre-trained TimesFM (200M/500M)
- [ ] 7.3 Implement zero-shot and fine-tuning modes for TimesFM
- [x] 7.4 Implement `PatchTSTForecaster` with patching, Transformer encoder
- [x] 7.5 Create base `DLForecaster` abstract class with fit/predict interface
- [x] 7.6 Implement training loop with early stopping, validation monitoring
- [x] 7.7 Add GPU/CPU detection and batch inference optimization
- [x] 7.8 Write unit tests for DL models (smoke tests with small data)

## 8. Evaluation Module (`src/forecasting/evaluation/`)

- [x] 8.1 Create `src/forecasting/evaluation/__init__.py`
- [x] 8.2 Implement `compute_all_metrics()`: MAE, RMSE, MAPE, sMAPE, MASE, WAPE
- [x] 8.3 Implement `bootstrap_confidence_intervals()` for metric uncertainty
- [ ] 8.4 Implement `TimeSeriesCV` with expanding window splits
- [x] 8.5 Implement `ModelComparator` for multi-model comparison table
- [x] 8.6 Implement `DieboldMarianoTest` for statistical significance
- [x] 8.7 Implement `ResidualDiagnostics` with ACF, normality, heteroscedasticity, Ljung-Box
- [x] 8.8 Create `EvaluationVisualizer` for forecast plots, scatter, error distributions, Q-Q
- [x] 8.9 Write unit tests for evaluation functions

## 9. Forecasting Module (`src/forecasting/forecasting/`)

- [x] 9.1 Create `src/forecasting/forecasting/__init__.py`
- [x] 9.2 Implement `FutureForecaster` for generating forecasts with best model
- [ ] 9.3 Implement exogenous variable projection for SARIMAX/Prophet
- [x] 9.4 Implement multi-hotel and aggregated forecasting
- [ ] 9.5 Create `ForecastVisualizer` for publication-ready plots
- [x] 9.6 Implement `ForecastReporter` for summary reports with assumptions/limitations
- [ ] 9.7 Write unit tests for forecasting functions

## 10. Main Notebook (`notebooks/hotel_revenue_forecasting.ipynb`)

- [x] 10.1 Create notebook with markdown sections matching required structure
- [x] 10.2 Section 0: Setup & Imports - environment info, seeds, config loading
- [x] 10.3 Section 1: Data Loading - load data, show schema, basic info
- [x] 10.4 Section 2: EDA - call all EDA functions, show visualizations, document findings
- [x] 10.5 Section 3: Preprocessing - feature engineering, temporal splits, show feature list
- [x] 10.6 Section 4: Baseline & Statistical Models - train Prophet, SARIMA, ETS; evaluate
- [x] 10.7 Section 5: ML Models - train LightGBM, XGBoost, CatBoost; evaluate with CV
- [x] 10.8 Section 6: DL Models - train TimesFM (zero-shot + fine-tune), PatchTST; evaluate
- [x] 10.9 Section 7: Final Comparison - metrics table, significance tests, residual analysis
- [x] 10.10 Section 8: Future Forecast - generate forecasts, visualize, document assumptions
- [x] 10.11 Section 9: Conclusions - best model recommendation, limitations, improvements
- [x] 10.12 Execute notebook end-to-end and verify all cells run without errors
- [x] 10.13 Export notebook to HTML for sharing

## 11. Documentation and Reproducibility

- [x] 11.1 Create `README.md` with project overview, setup instructions, usage
- [x] 11.2 Document all assumptions in `ASSUMPTIONS.md`
- [x] 11.3 Create `CHANGELOG.md` for version tracking
- [ ] 11.4 Verify reproducibility: run notebook in clean environment
- [ ] 11.5 Pin exact versions in `requirements-lock.txt` with `pip freeze`