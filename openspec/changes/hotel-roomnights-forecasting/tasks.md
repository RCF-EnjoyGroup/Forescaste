## 1. Configuration & Data

- [x] 1.1 Add room-nights keys to `config.yaml` (`target_col: rooms_sold`, `available_rooms_col`, sample file path); verify `yaml.safe_load` returns them and the revenue config keys remain unchanged
- [x] 1.2 Extend `DataLoader.load_sample` to generate capacity-bounded integer room nights (base occupancy + weekly/annual seasonality + Costa Rica holiday uplift with Monday-bridge effect + noise, clipped to [0, capacity]) plus a simulated renovation dip in one hotel, with contextual variables derived under exact identities (occupancy_rate = rooms_sold/available_rooms; room_revenue = rooms_sold × adr; revenue = room_revenue × (1 + ancillary share); revpar = room_revenue/available_rooms); verify two runs with the same seed produce identical files
- [x] 1.3 Generate and persist `data/hotel_roomnights_sample.csv` (5 hotels × ~3 years); verify it loads end-to-end through the pipeline, satisfies `0 <= rooms_sold <= available_rooms` on every row, and its identities hold exactly (occupancy_rate, room_revenue, revpar within floating tolerance; correlations between rooms_sold and adr/occupancy are positive and realistic)
- [x] 1.4 Implement NULL target handling: daily aggregation sums operational values with NULLs skipped BEFORE any imputation, and reports excluded-row/gap dates; verify a unit test proves financial-NULL rows never appear in the aggregated room-nights series
- [x] 1.5 Update schema validation to require `rooms_sold` for the room-nights target; verify loading a CSV without `rooms_sold` raises a clear validation error

## 2. Preprocessing

- [x] 2.1 Parameterize the feature engineering by target column so lags/rollings/expanding are built from `rooms_sold` (e.g., `rooms_sold_lag_7`, `rooms_sold_rmean_30`); verify engineered column names and the no-leakage shift semantics via the existing test suite
- [x] 2.2 Build the ML feature set excluding all `revenue`/`room_revenue`-derived features while keeping `available_rooms`, calendar and holiday features; verify the training feature list and the recursive-prediction feature list are identical
- [x] 2.3 Re-run the full existing test suite with the target switched; verify all prior tests still pass (no regression in the honest protocol)

## 3. Models

- [x] 3.1 Train Prophet, SARIMAX and ETS on the pre-test room-nights history in the notebook; verify each produces date-aligned multi-step predictions for the full test window
- [x] 3.2 Run LightGBM, XGBoost and CatBoost with the 3-step honest protocol (early stopping on val → refit pre-test history → recursive test forecast); verify predictions cover the same dates as the statistical models
- [x] 3.3 Train PatchTST with the internal temporal validation split and predict the test window recursively from the observed context; verify the multi-window forecast length matches the test window
- [x] 3.4 Keep the TimesFM honest platform check; verify the notebook reports its exclusion (never fallback numbers under its name) when unavailable on Windows

## 4. Evaluation & Forecast

- [x] 4.1 Build the comparison table including the SeasonalNaive baseline and all models on the same series and dates, with MASE computed against the baseline; verify every model is judged against the naive
- [x] 4.2 Run Diebold-Mariano tests (best vs each competitor) with Newey-West HAC variance; verify p-values and significance flags are reported
- [x] 4.3 Apply the capacity ceiling to future forecasts (cap at available capacity, portfolio = sum across hotels) and report capped dates/magnitudes; verify no forecasted value exceeds capacity
- [x] 4.4 Report implied occupancy (forecast / available rooms) alongside the forecast and flag implausible values; verify occupancy appears in the summary and final report
- [x] 4.6 Add forecast-bias and per-day-of-week diagnostics to the evaluation (signed bias %, over/under day counts, weekday/weekend error table per Lighthouse guidance); verify they appear in the comparison table and notebook report
- [x] 4.5 Refit the best model on all observed history and generate the 90-day room-nights forecast; verify no NaNs and units are room nights (not dollars)

## 5. Notebook & Documentation

- [x] 5.1 Create `notebooks/hotel_roomnights_forecasting.ipynb` mirroring the revenue notebook's documented structure (protocol notes, portfolio aggregate, honest evaluation, conclusions with synthetic-data disclaimer)
- [x] 5.2 Execute the notebook end-to-end; verify every cell runs without errors
- [x] 5.3 Export the notebook to a self-contained HTML report; verify the HTML file is generated
- [x] 5.4 Update README, ASSUMPTIONS.md and CHANGELOG.md with the room-nights capability and the fictional-data disclaimer; verify the three files document the new target
- [x] 5.5 Confirm the revenue deliverable is untouched; verify the revenue notebook, its outputs and config sections show no modifications

## 6. Tests & Validation

- [x] 6.1 Add unit test for NULL target handling (financial rows excluded from the aggregation; imputation never sees them); verify the test passes
- [x] 6.2 Add unit test for the synthetic generator (determinism by seed, capacity bounds, integer room nights); verify the test passes
- [x] 6.3 Add unit test for the capacity ceiling (forecasts capped, capping reported); verify the test passes
- [x] 6.4 Run the complete test suite (existing + new); verify pytest exits 0 with all tests passing
- [x] 6.5 Add unit test for the bias metrics (signed bias %, over/under day counts); verify the test passes
