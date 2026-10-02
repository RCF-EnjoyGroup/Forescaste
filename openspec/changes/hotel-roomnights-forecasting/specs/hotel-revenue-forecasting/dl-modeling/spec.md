## MODIFIED Requirements

### Requirement: TimesFM Model Implementation
The system SHALL implement Google's TimesFM foundation model for zero-shot/few-shot room-nights forecasting, and SHALL report its absence honestly when the platform cannot run it.

#### Scenario: TimesFM model loading
- **WHEN** TimesFM is initialized on a supported platform
- **THEN** the system loads the pre-trained TimesFM checkpoint (e.g., timesfm-1.0-200m or timesfm-1.0-500m)
- **THEN** the system configures context length and prediction horizon appropriately for hotel data

#### Scenario: TimesFM zero-shot forecasting
- **WHEN** TimesFM is used for zero-shot forecasting
- **THEN** the system provides historical context window to the model
- **THEN** the system generates forecasts for the specified horizon
- **THEN** the system handles multiple time series (per hotel) if applicable

#### Scenario: TimesFM fine-tuning (optional)
- **WHEN** fine-tuning is enabled
- **THEN** the system fine-tunes TimesFM on room-nights data
- **THEN** the system uses appropriate learning rate and epochs to avoid catastrophic forgetting
- **THEN** the system validates on temporal holdout set

#### Scenario: Honest platform exclusion
- **WHEN** TimesFM cannot be installed or executed on the current platform (e.g., Windows without JAX support)
- **THEN** the system excludes TimesFM from the model comparison instead of reporting fallback predictions under the TimesFM name
- **THEN** the exclusion and its reason are reported alongside the results

#### Scenario: TimesFM prediction
- **WHEN** TimesFM predicts on validation/test data
- **THEN** the system generates point forecasts and prediction intervals (if supported)
- **THEN** the system handles batch inference for efficiency
