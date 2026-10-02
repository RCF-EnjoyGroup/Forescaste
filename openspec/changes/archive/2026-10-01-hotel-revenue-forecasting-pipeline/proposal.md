## Why

Enjoy Costa Rica needs a robust, reproducible time series forecasting pipeline to predict hotel Revenue and Room Revenue from financial data. Accurate revenue forecasting is critical for revenue management, budgeting, and strategic decision-making in the hospitality industry. Currently there is no systematic forecasting approach, leading to reactive rather than proactive revenue management.

## What Changes

- **New capability**: End-to-end time series forecasting pipeline for hotel revenue prediction
- **New capability**: Jupyter notebook with complete forecasting workflow (EDA, preprocessing, modeling, evaluation, future forecasting)
- **New capability**: Multiple model comparison framework (statistical, ML, and deep learning/foundation models)
- **New capability**: Rigorous evaluation with proper time series metrics (MAE, RMSE, MAPE, sMAPE, MASE, WAPE)
- **New capability**: Reproducible, modular code with documentation of assumptions and limitations

## Capabilities

### New Capabilities
- `hotel-revenue-forecasting/data-ingestion`: Data loading and validation from SQL sources (test_enjoy_fac_hotel UNION enjoy_fac_hotel_financial)
- `hotel-revenue-forecasting/eda`: Exploratory data analysis for time series (distribution, seasonality, trend, stationarity, autocorrelation)
- `hotel-revenue-forecasting/preprocessing`: Time series preprocessing (missing values, outliers, temporal features, train/val/test split)
- `hotel-revenue-forecasting/statistical-modeling`: Classical statistical models (Prophet, SARIMA/SARIMAX, Exponential Smoothing/Holt-Winters)
- `hotel-revenue-forecasting/ml-modeling`: Machine learning models with temporal features (LightGBM, XGBoost, CatBoost)
- `hotel-revenue-forecasting/dl-modeling`: Deep learning/foundation models (TimesFM, PatchTST/Chronos/LSTM)
- `hotel-revenue-forecasting/evaluation`: Model evaluation and comparison framework with time series metrics
- `hotel-revenue-forecasting/forecasting`: Future forecast generation and visualization

### Modified Capabilities
None - this is a new forecasting capability

## Impact

- New Jupyter notebook in project root (e.g., `hotel_revenue_forecasting.ipynb`)
- New Python modules for reusable components (data loading, preprocessing, modeling, evaluation)
- Dependencies: prophet, statsmodels, lightgbm, xgboost, catboost, timesfm, torch, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- Requires access to SQL data sources (test_enjoy_fac_hotel, enjoy_fac_hotel_financial tables)