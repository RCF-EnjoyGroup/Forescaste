# DL Modeling Specification

## Purpose

Implement and evaluate deep learning and foundation models for hotel revenue forecasting including TimesFM (mandatory) and at least one additional model (PatchTST, Chronos, or LSTM/Transformer).

## Requirements

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

### Requirement: Additional Foundation/DL Model Implementation
The system SHALL implement at least one additional deep learning model: PatchTST, Chronos, or custom LSTM/Transformer.

#### Scenario: PatchTST implementation (if selected)
- **WHEN** PatchTST is configured
- **THEN** the system implements patching mechanism for time series
- **THEN** the system uses Transformer encoder architecture
- **THEN** the system trains on hotel revenue data with temporal splits

#### Scenario: Chronos implementation (if selected)
- **WHEN** Chronos is configured
- **THEN** the system loads pre-trained Chronos model (e.g., chronos-t5-small/base/large)
- **THEN** the system uses zero-shot or fine-tuned forecasting
- **THEN** the system handles probabilistic forecasting with quantile outputs

#### Scenario: LSTM/Transformer implementation (if selected)
- **WHEN** custom LSTM/Transformer is configured
- **THEN** the system implements sequence-to-sequence architecture
- **THEN** the system uses teacher forcing during training
- **THEN** the system supports multi-step forecasting (recursive or direct)

#### Scenario: DL model training
- **WHEN** DL model is trained
- **THEN** the system uses temporal train/validation split
- **THEN** the system implements early stopping on validation loss
- **THEN** the system uses appropriate loss (MSE, MAE, or quantile loss for probabilistic)
- **THEN** the system logs training curves (loss, validation metrics)

#### Scenario: DL model prediction
- **WHEN** trained DL model predicts on validation/test data
- **THEN** the system generates point forecasts
- **THEN** the system generates prediction intervals (if probabilistic model)

### Requirement: DL Model Evaluation
The system SHALL evaluate all DL models using standard time series metrics.

#### Scenario: Metric computation
- **WHEN** DL models are evaluated
- **THEN** the system computes: MAE, RMSE, MAPE, sMAPE, MASE, WAPE
- **THEN** the system computes metrics on validation and test sets separately
- **THEN** the system reports inference time and model size

#### Scenario: Computational efficiency reporting
- **WHEN** DL models are evaluated
- **THEN** the system reports: training time, inference time per sample, GPU memory usage
- **THEN** the system compares efficiency against statistical and ML baselines
