## 1. Project Setup and Environment

- [ ] 1.1 Create project directory structure: `src/forecasting/`, `notebooks/`, `data/`, `outputs/`
- [ ] 1.2 Create `pyproject.toml` with pinned dependencies: pandas, numpy, scikit-learn, statsmodels, prophet, lightgbm, xgboost, catboost, timesfm, torch, matplotlib, seaborn, plotly, jupyter, optuna
- [ ] 1.3 Create `requirements.txt` from pyproject.toml for pip compatibility
- [ ] 1.4 Create `src/forecasting/__init__.py` with version and exports
- [ ] 1.5 Create `.gitignore` for data, outputs, __pycache__, .ipynb_checkpoints
- [ ] 1.6 Create `config.yaml` for configurable parameters (split ratios, horizons, model configs)

## 2. Data Ingestion Module (`src/forecasting/data/`)

- [ ] 2.1 Create `src/forecasting/data/__init__.py`
- [ ] 2.2 Implement `DataLoader` class with SQL connection and CSV fallback
- [ ] 2.3 Implement `load_hotel_data()` function executing the UNION ALL query
- [ ] 2.4 Add schema validation (required columns, dtypes, date parsing)
- [ ] 2.5 Add NULL column detection and documentation for financial table
- [ ] 2.6 Create `data/sample_data.csv` generator for offline development
- [ ] 2.7 Write unit tests for data loading and validation

## 3. Exploratory Data Analysis Module (`src/forecasting/eda/`)

- [ ] 3.1 Create `src/forecasting/eda/__init__.py`
- [ ] 3.2 Implement `descriptive_statistics()` function with stats for revenue/room_revenue
- [ ] 3.3 Implement `missing_value_analysis()` with counts, percentages, patterns
- [ ] 3.4 Implement `time_series_decomposition()` (additive/multiplicative STL)
- [ ] 3.5 Implement `stationarity_tests()` with ADF and KPSS
- [ ] 3.6 Implement `autocorrelation_analysis()` with ACF/PACF plots
- [ ] 3.7 Implement `seasonality_detection()` with weekly/monthly/yearly patterns
- [ ] 3.8 Implement `outlier_detection()` with IQR and z-score methods
- [ ] 3.9 Implement `correlation_analysis()` with heatmap visualization
- [ ] 3.10 Create `EDAVisualizer` class for all plotting functions
- [ ] 3.11 Write unit tests for EDA functions

## 4. Preprocessing Module (`src/forecasting/preprocessing/`)

- [ ] 4.1 Create `src/forecasting/preprocessing/__init__.py`
- [ ] 4.2 Implement `MissingValueHandler` with forward/backward fill and interpolation
- [ ] 4.3 Implement `OutlierHandler` with winsorization and flagging options
- [ ] 4.4 Implement `TemporalFeatureEngineer` with calendar, cyclical, lag, rolling, expanding features
- [ ] 4.5 Implement `HolidayFeatureEngineer` with Costa Rica holidays and custom events
- [ ] 4.6 Implement `TemporalSplitter` with expanding window, configurable ratios, gap periods
- [ ] 4.7 Implement `FeatureScaler` with Standard/MinMax/Robust scaling (fit on train only)
- [ ] 4.7 Create `PreprocessingPipeline` composing all steps with fit/transform
- [ ] 4.8 Write unit tests for preprocessing (especially no-leakage verification)

## 5. Statistical Modeling Module (`src/forecasting/models/statistical/`)

- [ ] 5.1 Create `src/forecasting/models/__init__.py` and `statistical/__init__.py`
- [ ] 5.2 Implement `ProphetForecaster` class with Costa Rica holidays, hyperparameter tuning
- [ ] 5.3 Implement `SARIMAXForecaster` with auto_arima, seasonal periods [7, 365], exogenous support
- [ ] 5.4 Implement `ETSForecaster` with auto model selection (AIC), damped trend
- [ ] 5.5 Create base `StatisticalForecaster` abstract class with fit/predict interface
- [ ] 5.6 Implement prediction interval generation for all statistical models
- [ ] 5.7 Write unit tests for each statistical model

## 6. Machine Learning Modeling Module (`src/forecasting/models/ml/`)

- [ ] 6.1 Create `src/forecasting/models/ml/__init__.py`
- [ ] 6.2 Implement `LightGBMForecaster` with categorical support, early stopping, Optuna tuning
- [ ] 6.3 Implement `XGBoostForecaster` with early stopping, Optuna tuning
- [ ] 6.3 Implement `CatBoostForecaster` with native categorical, early stopping, Optuna tuning
- [ ] 6.4 Create base `MLForecaster` abstract class with fit/predict interface
- [ ] 6.5 Implement quantile regression for prediction intervals (LightGBM/XGBoost)
- [ ] 6.6 Write unit tests for each ML model

## 7. Deep Learning Modeling Module (`src/forecasting/models/dl/`)

- [ ] 7.1 Create `src/forecasting/models/dl/__init__.py`
- [ ] 7.2 Implement `TimesFMForecaster` wrapper for pre-trained TimesFM (200M/500M)
- [ ] 7.3 Implement zero-shot and fine-tuning modes for TimesFM
- [ ] 7.4 Implement `PatchTSTForecaster` with patching, Transformer encoder
- [ ] 7.5 Create base `DLForecaster` abstract class with fit/predict interface
- [ ] 7.6 Implement training loop with early stopping, validation monitoring
- [ ] 7.7 Add GPU/CPU detection and batch inference optimization
- [ ] 7.8 Write unit tests for DL models (smoke tests with small data)

## 8. Evaluation Module (`src/forecasting/evaluation/`)

- [ ] 8.1 Create `src/forecasting/evaluation/__init__.py`
- [ ] 8.2 Implement `compute_all_metrics()`: MAE, RMSE, MAPE, sMAPE, MASE, WAPE
- [ ] 8.3 Implement `bootstrap_confidence_intervals()` for metric uncertainty
- [ ] 8.4 Implement `TimeSeriesCV` with expanding window splits
- [ ] 8.5 Implement `ModelComparator` for multi-model comparison table
- [ ] 8.6 Implement `DieboldMarianoTest` for statistical significance
- [ ] 8.7 Implement `ResidualDiagnostics` with ACF, normality, heteroscedasticity, Ljung-Box
- [ ] 8.8 Create `EvaluationVisualizer` for forecast plots, scatter, error distributions, Q-Q
- [ ] 8.9 Write unit tests for evaluation functions

## 9. Forecasting Module (`src/forecasting/forecasting/`)

- [ ] 9.1 Create `src/forecasting/forecasting/__init__.py`
- [ ] 9.2 Implement `FutureForecaster` for generating forecasts with best model
- [ ] 9.3 Implement exogenous variable projection for SARIMAX/Prophet
- [ ] 9.4 Implement multi-hotel and aggregated forecasting
- [ ] 9.5 Create `ForecastVisualizer` for publication-ready plots
- [ ] 9.6 Implement `ForecastReporter` for summary reports with assumptions/limitations
- [ ] 9.7 Write unit tests for forecasting functions

## 10. Main Notebook (`notebooks/hotel_revenue_forecasting.ipynb`)

- [ ] 10.1 Create notebook with markdown sections matching required structure
- [ ] 10.2 Section 0: Setup & Imports - environment info, seeds, config loading
- [ ] 10.3 Section 1: Data Loading - load data, show schema, basic info
- [ ] 10.4 Section 2: EDA - call all EDA functions, show visualizations, document findings
- [ ] 10.5 Section 3: Preprocessing - feature engineering, temporal splits, show feature list
- [ ] 10.6 Section 4: Baseline & Statistical Models - train Prophet, SARIMA, ETS; evaluate
- [ ] 10.7 Section 5: ML Models - train LightGBM, XGBoost, CatBoost; evaluate with CV
- [ ] 10.8 Section 6: DL Models - train TimesFM (zero-shot + fine-tune), PatchTST; evaluate
- [ ] 10.9 Section 7: Final Comparison - metrics table, significance tests, residual analysis
- [ ] 10.10 Section 8: Future Forecast - generate forecasts, visualize, document assumptions
- [ ] 10.11 Section 9: Conclusions - best model recommendation, limitations, improvements
- [ ] 10.12 Execute notebook end-to-end and verify all cells run without errors
- [ ] 10.13 Export notebook to HTML for sharing

## 11. Documentation and Reproducibility

- [ ] 11.1 Create `README.md` with project overview, setup instructions, usage
- [ ] 11.2 Document all assumptions in `ASSUMPTIONS.md`
- [ ] 11.3 Create `CHANGELOG.md` for version tracking
- [ ] 11.4 Verify reproducibility: run notebook in clean environment
- [ ] 11.5 Pin exact versions in `requirements-lock.txt` with `pip freeze`