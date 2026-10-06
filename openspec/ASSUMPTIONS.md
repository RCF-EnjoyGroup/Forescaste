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
5. **Industry MAPE benchmarks (Lighthouse, 2025)** — context, not targets: below 10% is generally excellent; 10–20% acceptable for most properties; typical ranges by property type: city-center corporate 8–12%, seasonal independent 15–25%, leisure/event resort 18–28%, small (<50 rooms) 15–30%. **Caveat**: these benchmarks usually measure short-horizon forecasts (days–weeks ahead); this pipeline evaluates multi-step windows (up to 158 days ahead honestly), a stricter regime — do not compare our multi-step MAPE against short-horizon benchmarks. The meaningful benchmark is the pipeline's own MAPE trend over time (tracked with each retraining).
6. **Forecast bias**: magnitude metrics (MAE/MAPE/WAPE) do not carry direction. A signed bias % and over/under day counts are reported with every evaluation; |Bias%| > 5% warrants operational adjustment (systematic under-forecasting = lost revenue; systematic over-forecasting = overstaffing/discounting).
7. **Segment diagnostics**: per-day-of-week error breakdown localizes systematic weakness (e.g., weekends vs midweek) for staffing and pricing follow-up.
8. **Conformal bands interpretation**: the P10/P90 bands are empirical, built from the winner's backtest residuals per horizon step. With few backtest windows (6), the per-step quantiles are coarse and the empirical coverage can undershoot the nominal level (observed: 60% vs 80% nominal on the test window). Read the bands as "honest but wide-uncertainty" estimates; more windows (longer history, real data) or a moving-block bootstrap will tighten the calibration claim. A band's total is NOT additive certainty about the sum — it is the sum of per-day bounds.
9. **Backtesting evidence**: model ranking is reported with rolling-origin stability (mean rank across 6 windows) in addition to the single test split; a winner that does not hold its rank across windows would be flagged as unstable regardless of its test metrics.
10. **Tuning objective**: hyperparameters are selected by the recursive multi-step MAE on the validation window — the same regime the business will experience — not by one-step shortcuts; tuned hyperparameters are frozen before backtesting so stability is measured, not re-selected.

## Limitations (Not Assumptions)

- No external data (weather, events, competitor pricing).
- No hierarchical forecasting (hotel → region → portfolio).
- No real-time inference or automated retraining.
- Deep learning models (TimesFM, PatchTST) may not converge with limited data.
- Single-model forecast; ensemble methods may improve robustness.

## Room-Nights Pipeline Assumptions

1. **Primary target**: Room nights (`rooms_sold`) — management decision to forecast demand first; monetary forecasting (revenue ≈ room nights × ADR) builds on this foundation.
2. **NULL target handling**: Rows from `enjoy_fac_hotel_financial` carry NULL `rooms_sold`; they are excluded from the daily aggregation BEFORE any imputation (imputing first would contaminate the target series with financial NULLs).
3. **Forward-known rule**: A variable is an ML input only if its future value is known or forecastable. `revenue`/`room_revenue`/`adr`/`occupancy_rate` are realized outcomes (future unknown) and excluded; `available_rooms` is planned capacity and included with its last observed value carried forward.
4. **Future capacity**: Future `available_rooms` = last observed value (no announced renovations/closures); the assumption is flagged in the forecast report.
5. **Capacity ceiling**: Forecasts are capped at available capacity; implied occupancy above 100% before capping is reported as a sanity warning.
6. **Point forecasts continuous**: Metrics are computed on continuous predictions; rounding to integers is presentation only.
7. **Synthetic data**: `data/hotel_roomnights_sample.csv` is a fictional test file generated with exact physical identities (occupancy_rate = rooms_sold/available_rooms; room_revenue = rooms_sold × adr; revenue = room_revenue × (1 + ancillary share); revpar = room_revenue/available_rooms). Results on it validate the methodology, not business numbers.
8. **Precision ceiling**: With the current two SQL tables, the achievable precision is bounded by the target's own structure + calendar/holidays + capacity. The forward-looking data that unlocks "excellent" precision (reservations/pickup curves, group blocks, planned rates, cancellation rates, segment mix) requires a PMS/reservations source and is documented as future work (see the room-nights change's design.md).
