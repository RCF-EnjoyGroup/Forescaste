# Assumptions

This document records all assumptions made in the hotel revenue forecasting pipeline.

## Data Assumptions

1. **Data granularity**: Daily data (one row per hotel per day).
2. **Data completeness**: At least 2 years of historical data for yearly seasonality detection.
3. **Data quality**: Revenue values are non-negative and represent actual financial transactions.
4. **Hotel stability**: The set of hotels remains relatively constant over the forecast period.
5. **NULL columns**: The 4 extra NULL columns from `enjoy_fac_hotel_financial` (cost_center, account_code, debit_amount, credit_amount) are not used for forecasting.

## Modeling Assumptions

1. **Stationarity**: Revenue series may be non-stationary; differencing and transformations are applied as needed.
2. **Seasonality**: Weekly (7-day) and yearly (365-day) seasonality patterns exist and are stable.
3. **Trend**: Revenue trends are smooth and can be captured by model components.
4. **Independence**: Residuals from different hotels are independent (for per-hotel models).
5. **Feature stability**: Temporal features (lags, rolling stats) maintain their predictive power.
6. **No structural breaks**: No major policy changes, renovations, or market shifts during the forecast horizon.

## Temporal Split Assumptions

1. **Chronological ordering**: Data is sorted by date before splitting.
2. **Expanding window**: Training data is the earliest period; test data is the latest.
3. **Gap between splits**: 7-day gap between train/val and val/test reduces autocorrelation leakage.
4. **Representative test set**: The test period is representative of future conditions.

## Forecast Assumptions

1. **Continuity**: Historical patterns (seasonality, trends) continue into the forecast horizon.
2. **No external shocks**: No pandemics, natural disasters, or major events during the forecast period.
3. **Constant operations**: Hotel operations remain consistent (no major renovations, closures).
4. **Exogenous variables**: If SARIMAX/Prophet use exogenous regressors, their future values are projected forward (e.g., holidays are known).
5. **Model stability**: The best model during evaluation remains the best for future forecasting.

## Metric Assumptions

1. **MAPE stability**: MAPE may be unreliable when actual revenue values are near zero; WAPE and MASE are preferred as primary metrics.
2. **sMAPE range**: sMAPE values are bounded between 0% and 200%.
3. **MASE baseline**: MASE uses a seasonal naive baseline (shift by 1 period) for scaling.
4. **Bootstrap CIs**: Bootstrap confidence intervals assume exchangeability of forecast errors.

## Limitations (Not Assumptions)

- No external data (weather, events, competitor pricing).
- No hierarchical forecasting (hotel → region → portfolio).
- No real-time inference or automated retraining.
- Deep learning models (TimesFM, PatchTST) may not converge with limited data.
- Single-model forecast; ensemble methods may improve robustness.
