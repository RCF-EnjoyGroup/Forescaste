## Why

Management decided to start the forecasting program with **room nights** (noches de habitación vendidas, column `rooms_sold`) instead of monetary revenue. Room nights is the structural demand signal that drives staffing, inventory and pricing decisions, and it is the base of the revenue identity (`revenue ≈ room nights × ADR`), so a validated demand forecast becomes the foundation for later monetary forecasting. The honest-protocol pipeline delivered by the archived `hotel-revenue-forecasting-pipeline` change (leakage-safe splits, recursive multi-step evaluation, SeasonalNaive baseline, Diebold-Mariano with Newey-West) transfers directly; this change points it at the new target and adds room-nights-specific behavior (NULL handling, capacity ceiling, synthetic test data).

## What Changes

- **Target pivot**: the primary forecast target becomes room nights (`rooms_sold`, integer room-nights sold per day); evaluation metrics are reported in room-night units.
- **Synthetic sample data file**: generate a dedicated fictional-data file (with realistic capacity-bounded room nights, weekly/annual seasonality and Costa Rica holiday effects) so the model can be developed and tested offline before SQL access is available.
- **NULL-safe ingestion**: `rooms_sold` is NULL on rows coming from `enjoy_fac_hotel_financial`; the daily aggregation SHALL exclude those NULLs before any imputation, preventing phantom-zero contamination of the target series.
- **Feature discipline for ML**: features derived from monetary columns (`revenue`, `room_revenue`) are excluded from ML inputs — their future values are unknown, and using them would require fabricated inputs during recursive forecasting.
- **Capacity ceiling**: generated forecasts SHALL not exceed available capacity (`available_rooms`), and the implied occupancy SHALL be reported as a sanity check.
- **Same honest protocol**: SeasonalNaive baseline in every comparison, recursive multi-step evaluation for ML models, Diebold-Mariano significance with Newey-West correction, final model retrained on all observed history.
- **New deliverable notebook** for room nights; the existing revenue notebook and results are preserved untouched.

## Capabilities

### New Capabilities
None - this change re-targets and extends the existing forecasting capabilities.

### Modified Capabilities
- `hotel-revenue-forecasting/data-ingestion`: target column validation (`rooms_sold`), NULL-operational-column handling before aggregation, synthetic sample-data generation for offline testing
- `hotel-revenue-forecasting/eda`: exploratory analyses re-targeted to the room-nights series (stats, stationarity, outliers, correlation)
- `hotel-revenue-forecasting/preprocessing`: lag/rolling/expanding features built from `rooms_sold`; exclusion of future-unknown monetary features from ML inputs
- `hotel-revenue-forecasting/statistical-modeling`: Prophet, SARIMAX and ETS forecast room nights instead of revenue
- `hotel-revenue-forecasting/ml-modeling`: LightGBM/XGBoost/CatBoost forecast room nights; monetary-derived features excluded
- `hotel-revenue-forecasting/dl-modeling`: PatchTST (and TimesFM where installable) forecast room nights
- `hotel-revenue-forecasting/evaluation`: adds the honest-protocol requirements (SeasonalNaive baseline, leakage-free recursive multi-step evaluation, Newey-West-corrected significance)
- `hotel-revenue-forecasting/forecasting`: future forecast reported in room-night units, capped by available capacity, with implied occupancy reporting

## Impact

- `config.yaml`: configurable `target_col: rooms_sold` + room-nights section
- `src/forecasting/data/`: sample-data generator emits capacity-bounded room nights; loader validation updated
- `src/forecasting/preprocessing/`: feature engineering parameterized by target column
- `notebooks/`: new `hotel_roomnights_forecasting.ipynb` (revenue notebook unchanged)
- `tests/`: target-agnostic suite re-run + room-nights-specific tests (NULL handling, capacity cap)
- Dependencies: unchanged (no new libraries)
