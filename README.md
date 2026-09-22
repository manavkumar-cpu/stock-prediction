# Stock Price Prediction using LSTM

Forecast the next trading session’s closing price using an LSTM,
with baseline comparisons and chronological evaluation.

## Project Plan

See [Project Plan](docs/project_plan.md) for the scope,
features, models, and evaluation strategy.

## Environment

Initial verified environment:
- macOS, Apple Silicon
- Python 3.12.14
- TensorFlow 2.21.0

## Setup

Run all commands from the project root: `stock-price-lstm`.

### 1. Install Python 3.12

On macOS with Homebrew:

```bash
brew install python@3.12
```

On Windows or Linux, install Python 3.12 using the appropriate
installer or package manager.

### 2. Create a Virtual Environment

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

The current lock file was generated on an Apple Silicon Mac.
Other platforms may require dependency adjustments.

`requirements.txt` lists direct dependencies.
`requirements-lock.txt` records the installed versions.

### 4. Verify Installation

```bash
python --version
python -m pip check
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.constant([1.0, 2.0, 3.0])).numpy())"
```

Expected:
- Python 3.12.x
- No broken requirements found
- TensorFlow calculation returns 6.0

### 5. Configure VS Code

Select the Python interpreter inside `.venv`.
For notebooks, select the same environment as the kernel.

## Returning to the Project

Activate the existing environment whenever opening a new terminal.

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You do not need to recreate the environment or reinstall packages
every time.

To leave the environment:

```bash
deactivate
```

## Download Historical Data

From the project root, with the virtual environment active:

```bash
python -m src.data.download
```

Settings are read from `configs/base.yaml`.
The raw CSV is saved in `data/raw/`.
Repeating the command with the same settings overwrites that CSV.

## Validate the Raw Dataset

```bash
python -m src.data.validate
```

Checks dates, exchange-calendar coverage, numeric values,
missing data, volume, and OHLC consistency.

Report: `reports/metrics/AAPL_validation.json`

The validator does not modify the raw CSV.

## Exploratory Data Analysis

Open `notebooks/01_eda.ipynb` in VS Code.
Select the project's `.venv` kernel and run all cells in order.

EDA uses only the 2010–2019 training period.
Plots are saved in `reports/figures/`.
Summary tables are saved in `reports/metrics/`.
The raw CSV is unchanged.

## Build Candidate Features

```bash
python -m src.features.build
```

Creates returns, moving averages, MACD, RSI, rolling volatility,
relative volume, and daily range ratio.

Output:
`data/processed/AAPL_2010-01-01_2026-01-01_features.csv`

Features use current and earlier observations only.
Initial warm-up values remain missing.
Raw data and dates are preserved.
No scaling or target construction is performed in this step.

The first LSTM experiment remains Close-only.

## Add Market Context

```bash
python -m src.features.market
```

Downloads and caches S&P 500, NASDAQ Composite, and VIX data
under `data/raw/market/`.

Adds previous-session market returns, VIX level, and VIX change.
The lag is measured on AAPL's trading-session index.

Output:
`data/processed/AAPL_2010-01-01_2026-01-01_features_market.csv`

Required market closing values are checked against AAPL sessions.
Missing observations are not silently filled.
Original AAPL dates and existing features are preserved.

The initial LSTM remains Close-only.

## Create Targets and Assign Chronological Splits

```bash
python -m src.data.prepare
```

Pairs each forecast date with the next trading session's Close.

Splits are assigned by target date:
- Training: through 2019.
- Validation: 2020–2022.
- Final test: 2023–2025.

Output:
`data/processed/AAPL_2010-01-01_2026-01-01_targets_h1.csv`

Split summary:
`reports/metrics/AAPL_target_split_summary.csv`

Targets are stored separately from input features.
Counts are before sequence windowing.
No scaling or model training is performed in this step.

## Scale Inputs and Targets

```bash
python -m src.data.scale
```

Uses separate MinMaxScalers for selected inputs and target prices.

Input fitting uses history through the last training forecast.
Target fitting uses training target prices only.
Validation and test data use the fitted training scalers.

Scaled tables:
`data/processed/close_only_h1/`

Scalers and fitting metadata:
`artifacts/close_only_h1/`

Values outside the training range are not clipped.
Original target prices and naive predictions are preserved.

## Build LSTM Sequences

```bash
python -m src.data.sequences
```

Creates chronological 60-session inputs with one next-session target.

Arrays have shape:
- X: (samples, sessions, features)
- y: (samples, 1)

Output:
`data/processed/close_only_h1/sequences_w60/`

Splits follow target dates.
Input windows may include earlier observed history across split boundaries.
Windows containing missing selected features are excluded.
Dates, original target prices, and naive predictions are saved as metadata.

## Evaluate the Naive Baseline

```bash
python -m src.evaluation.naive
```

Predicts the next session's Close using the forecast session's Close.

Evaluates only the validation examples retained by sequence preparation.
Reports MAE, RMSE, MSE, MAPE (%), and R² in the original price scale.

Metrics and predictions:
`reports/metrics/close_only_h1/w60/`

Plot:
`reports/figures/close_only_h1/w60/naive_validation.png`

No model fitting is required. Final-test performance is not evaluated.

## Train the First LSTM

```bash
python -m src.models.train
```

Architecture:
- 60-session Close-only input
- LSTM with 32 units
- Linear Dense output

Uses Adam and scaled-target MSE.
Early stopping and checkpoint selection monitor validation loss.
Final-test data is not loaded.

Every run is saved under:
`artifacts/close_only_h1/runs/lstm_<timestamp>/`

The run folder includes the best model, matching scalers,
configuration, epoch history, and learning curves.

## LSTM Experiments: Direct Price and Residual Forecasting

### Experimental Setup

- Stock: AAPL
- Training: 2010–2019
- Validation: 2020–2022
- Final test: 2023–2025, reserved for the frozen experimental design
- Input: 60 observed trading sessions of Close
- Target: Next trading session’s Close
- Training sequences: 2,456
- Validation sequences: 756
- Architecture: 32-unit LSTM with a single linear output
- Trainable parameters: 4,385
- Scalers fitted on training data only
- Checkpoint selection and early stopping based on validation MSE

All reported results below are validation results, not final-test results.

### Experiment 1: Direct Price LSTM

The model predicts tomorrow’s scaled closing price directly from the
previous 60 closing prices.

Best checkpoint: epoch 39.

Run:
`artifacts/close_only_h1/runs/lstm_20260918T060419_711053Z`

The model showed persistent underprediction, especially at higher prices.
For forecast-date Close above $150, it underpredicted on 198 of 199
validation sessions.

Difficulty extrapolating beyond training price levels is a possible
explanation, not a proven cause.

### Experiment 2: Residual LSTM

The model predicts a correction to the current closing price:

Predicted next Close = Current Close + Predicted correction

The input window is centered on its final observed Close inside the model.
This removes the absolute price level from the LSTM’s input.

A fixed connection passes the current Close to the output. The correction
layer starts at zero, so the untrained model reproduces the naive baseline.

The input and target scalers have different fitted ranges. A fixed affine
conversion puts the current Close into target-scaler units before adding
the correction.

Both input representation and output structure changed in this experiment.
Their individual contributions have not yet been isolated.

Best checkpoint: epoch 1.
Training stopped after epoch 11.

Run:
`artifacts/close_only_delta_h1/runs/residual_lstm_20260918T090838_562896Z`

### Validation Results

Price errors are calculated after inverse-transforming predictions.

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| Naive | 2.076200 | 2.791289 | 1.667125 | 0.991650 |
| Direct price LSTM | 5.788713 | 7.368096 | 4.107835 | 0.941817 |
| Residual LSTM | 2.075140 | 2.790226 | 1.666567 | 0.991656 |

Mean prediction error, defined as prediction minus actual:
- Direct price LSTM: −$4.9721
- Residual LSTM: +$0.0312

The residual model substantially reduced the original model’s overall bias.
However, its improvement over naive was only approximately 0.05% in MAE
and 0.04% in RMSE.

These small differences do not establish reliable predictive skill.
The displayed examples suggest nearly constant positive corrections;
the full prediction table requires inspection.

### Commands

Run from the project root with the virtual environment active.

Train the residual model:

```bash
python -m src.models.train_residual --source-run artifacts/close_only_h1/runs/lstm_20260918T060419_711053Z
```

Each training execution creates a new timestamped run directory.

Evaluate the recorded residual run:

```bash
python -m src.evaluation.lstm --run-dir artifacts/close_only_delta_h1/runs/residual_lstm_20260918T090838_562896Z
```

Inspect errors by year and forecast-date price:

```bash
python -m src.evaluation.diagnostics --run-dir artifacts/close_only_delta_h1/runs/residual_lstm_20260918T090838_562896Z
```

### Saved Outputs

Each run contains its checkpoint, configuration, scaler copies,
training history, model summary, and run summary.

Its `evaluation/` folder contains:
- `validation_metrics.json`
- `validation_comparison.csv`
- `validation_predictions.csv`
- `validation_predictions.png`

Running diagnostics also creates:
- `errors_by_year.csv`
- `errors_by_price_band.csv`

### Current Status

The direct price and residual LSTM experiments have been evaluated
on matching validation dates.

Next: inspect the residual corrections and compare against a constant
price-change baseline estimated using training samples only.

### Development Walk-Forward Findings

Three expanding training folds scored 2020, 2021, and 2022.
Each fold used a preceding year for checkpoint selection and fresh
training-fitted scalers.

Pooled RMSE:
- Naive: $2.791289
- Training mean change: $2.791546
- Direct LSTM: $12.344359
- Residual LSTM: $2.806713

The residual LSTM slightly improved on naive in 2020 and 2021,
but performed worse in 2022 and overall. Its corrections were
positive on every scored session.

No consistent advantage over naive has been demonstrated.
These are development results; final-test evaluation remains pending.

### Six-Feature LSTM Comparison

Compared Close-only and six-feature residual LSTMs on matching
development walk-forward samples.

Added features: Return_1, RSI_14, Volatility_20,
Relative_Volume_20, and Range_Ratio.

Pooled results:
- Naive: MAE 2.076200, RMSE 2.791289
- Close-only LSTM: MAE 2.084446, RMSE 2.801225
- Six-feature LSTM: MAE 2.084211, RMSE 2.794764

Six features slightly improved pooled LSTM errors but did not beat naive.
Benefits varied by year. Near-zero pooled bias did not imply lower MAE.

Run: artifacts/multivariate_walk_forward/20260918T153239_161878Z

Final-test evaluation remains pending.

## Market-Context Feature Experiment

Compared six-feature and ten-feature residual LSTMs using
expanding-window development folds scoring 2020, 2021, and 2022.

The ten-feature model added:
- SP500_Return_1_Lag1
- NASDAQ_Return_1_Lag1
- VIX_Level_Lag1
- VIX_Change_1_Lag1

Both models used matching forecast dates and training-only
preprocessing. Checkpoints were selected using the preceding year.

| Model | Pooled MAE | Pooled RMSE |
|---|---:|---:|
| Naive | 2.076200 | 2.791289 |
| Training mean change | 2.076335 | 2.791556 |
| Six-feature LSTM | 2.084211 | 2.794764 |
| Ten-feature LSTM | 2.101119 | 2.815636 |

The market features increased pooled MAE and RMSE relative to
the six-feature control. The ten-feature model also had worse
RMSE in each scoring year.

Decision: retain the six-feature LSTM as the multivariate
reference. Naive remains the strongest pooled benchmark.
These findings are from one seed and the tested configuration.

Run: artifacts/market_walk_forward/20260919T151350_677802Z

Final-test data from 2023–2025 has not been evaluated.


## Phase 15 — Window Comparison

Compared 20, 30, 60, and 120-session windows using the
six-feature residual LSTM and development folds scoring
2020, 2021, and 2022.

All variants used matching training, selection, and scoring
target dates, with shared training-only preprocessing per fold.

| Window | MAE | RMSE |
|---|---:|---:|
| 20 | 2.091610 | 2.798840 |
| 30 | 2.091676 | 2.798443 |
| 60 | 2.092042 | 2.798426 |
| 120 | 2.091985 | 2.798373 |
| Naive | 2.076200 | 2.791289 |

Selected 120 sessions by the predefined minimum pooled
development RMSE rule. Differences between windows were
very small; 20 sessions achieved the lowest LSTM MAE.

The selected LSTM did not outperform naive.
Results reflect one seed and the tested configuration.
Final-test data remains untouched.

### Market-regime analysis

Regimes were assigned using forecast-date information.
Volatility thresholds were estimated from each fold's training history.

- One-session LSTM and Attention-LSTM did not outperform naive RMSE
  in any of the four reported pooled trend/volatility groups.
- Five-session LSTM improved lower-volatility RMSE by 0.77%
  across 339 forecasts, but worsened high-volatility RMSE by 0.78%.
- Naive remained stronger on pooled overall MAE and RMSE.
- Subgroup differences are descriptive development findings;
  their statistical uncertainty has not yet been assessed.
- Final-test data (2023–2025) remains unused for model selection.

### Bootstrap uncertainty analysis

Used 5,000 paired circular block-bootstrap repetitions per block
length, with lengths 5, 10, and 20. Resampling stayed within each
scoring year and preserved model/naive forecast pairing.

All 18 approximate 95% percentile intervals for MAE and RMSE
differences included zero.

Naive achieved slightly lower observed errors, but this analysis
did not clearly distinguish performance. No reliable LSTM
advantage was established, and equivalence was not demonstrated.

Results are conditional on saved development predictions.
They do not account for retraining variability or previous model
selection. The lower-volatility subgroup advantage remains
descriptive and was not assessed by this overall bootstrap.

### Controlled feature ablation

Compared four LSTM feature variants using identical 120-session
windows, target dates, preprocessing, and training settings.

The full six-feature control reproduced the previous pooled
MAE and RMSE to the reported precision.

Removing Return_1 and RSI_14 worsened RMSE in all three scoring
years. Removing volatility, relative volume and range produced
a small pooled deterioration, with mixed effects across years.

Close-only achieved lower pooled MAE than the full six-feature
model, but higher RMSE. Naive remained best on both pooled metrics.

These findings are conditional on this setup and one random seed.
Run: artifacts/feature_ablation/w120/20260920T103211_767261Z

### OHLCV-only comparison

Evaluated an LSTM using Close, Open, High, Low and Volume on the
same 120-session windows and development target dates as the
feature-ablation experiments.

OHLCV slightly improved on naive in 2020 and 2021, but performed
worse in 2022. Pooled MAE was 2.092349 and RMSE was 2.813766,
both worse than naive and the six-feature LSTM.

Best epochs: 10, 34 and 2 for the three folds.
The run completed successfully with checkpoint reload checks.

OHLCV was not selected as an improvement under this setup.
Results reflect one random seed.

Run: artifacts/ohlcv_walk_forward/w120/20260920T105352_629673Z

# Final Evaluation Protocol — AAPL

## Purpose

Evaluate the completed forecasting designs on 2023–2025 target
dates after development using data through 2022.

LSTM is the primary project model. Naive is the reference baseline.
Development results have not established a reliable LSTM advantage.

## Data and timing

- Use the saved AAPL dataset; do not refresh it for this experiment.
- Forecast after the current session's required data is available.
- Primary target: next trading session's Close.
- Secondary target: Close five trading sessions ahead.
- Predictions and evaluation use Yahoo's split-adjusted Close basis.
- Record that downloaded historical prices are retrospectively adjusted;
  this is not a point-in-time reconstruction of historical data vintages.

## Final chronological partitions

Assign examples by target dates:

- Training: 2010–2019.
- Validation: 2020–2022.
- Final test: 2023–2025.

For five-session forecasts, all future sessions t+1 through t+5
must belong to the same partition. Input history may precede a
partition boundary.

Partition boundaries refer to these complete periods, not individual
calendar years within validation or test.

No random observation split is permitted.

This fixed-split evaluation is separate from the earlier expanding-window
development experiments.

## Features and preprocessing

Neural models and XGBoost use:

1. Close
2. Return_1
3. RSI_14
4. Volatility_20
5. Relative_Volume_20
6. Range_Ratio

For each window:
- Center Close on its final observed Close.
- Divide relative Close by the training input Close range.
- Standardize the five other features using unique training input rows.
- Predict future Close minus current Close.

For neural models, divide the target change by the training target
Close range. Reconstruct predictions in dollars before evaluation.

Fit all learned preprocessing on training data only.
Do not clip validation or test values to training ranges.
Preserve session spacing and exclude incomplete input windows.

## Models

One-session comparison:
- Naive.
- Training-mean-change baseline.
- Six-feature LSTM, 120 sessions.
- Six-feature Attention-LSTM, 120 sessions.
- Six-feature GRU, 60 sessions.
- XGBoost, flattened 120-session six-feature windows.
- ARIMA using historical Close.

Five-session comparison:
- Naive.
- Training-mean-change baseline.
- Six-feature LSTM, 120 sessions.

The mean-change baseline is estimated separately for each horizon,
using only eligible training examples.

Retain the implemented architectures and development training settings.
Record all settings explicitly in the final run manifest.

## Model selection

Neural models:
- Fit weights on training examples.
- Select the checkpoint with minimum validation MSE.
- Preserve the existing seed, optimizer, batch size and stopping rules.

XGBoost:
- Retain the existing candidate depths and training settings.
- Select depth and boosting rounds using validation performance only.

ARIMA:
- Retain the existing candidate orders and no-trend specification.
- Fit candidate coefficients on training history.
- Select order using sequential validation RMSE.
- Update filtering state with observed values after forecasting;
  do not refit coefficients during validation or test.

Do not refit models or preprocessing on validation observations.

## Test execution

- Save selected checkpoints, preprocessing and configuration before scoring.
- Freeze model weights, fitted parameters and preprocessing during test.
- Use earlier observed test sessions as later forecast inputs.
- Never use an unrealized target as an input.
- Evaluate every listed model, including unsuccessful results.
- Do not choose architectures, features or checkpoints using test metrics.

Generate five-session predictions directly, without supplying intervening
future actual prices to the model.

## Evaluation

Within each horizon, compare models on exactly matching forecast and
target dates and verify actual-price alignment.

Report:
- MAE and RMSE as primary metrics.
- MSE, MAPE, R² and mean error.
- Pooled test metrics and descriptive yearly metrics.
- Actual-versus-predicted plots and error plots.

Use dollars for price-error metrics.
Do not directly rank one-session and five-session models by raw RMSE.

Report bootstrap uncertainty as conditional on saved predictions.
Five-session errors overlap and are not independent.

## Interpretation

Report observed rankings without assuming complex models beat naive.
Do not interpret a confidence interval containing zero as proof of equivalence.

The existing bootstrap analysis measures uncertainty in aggregate metric
differences. It does not provide individual forecast prediction intervals.

Any changes motivated by final-test results are subsequent exploratory
work and cannot reuse this period as an untouched final test.

## Scope of this evaluation

This protocol covers AAPL.
Additional tickers, optional ten-session forecasts and optional trading
backtests are separate extensions.

The feature experiments compared alternative feature sets; they did not
evaluate a complete nested OHLCV-plus-all-indicators sequence.

### Final neural checkpoint selection

Training targets: 2010–2019.
Validation targets: 2020–2022.
No refitting on validation data.

Selected epochs:
- LSTM h1/w120: 35.
- Attention-LSTM h1/w120: 10.
- GRU h1/w60: 9.
- LSTM h5/w120: 34.

All checkpoint reload checks passed.
Plain LSTM achieved the lowest validation MAE and RMSE among
the evaluated neural models at each horizon.

Validation metrics were used for checkpoint selection and must
not be presented as independent final-test performance.

Run: artifacts/final_neural/20260920T112518_629227Z
Final-test evaluation has not started.

### Final XGBoost selection

Training: 2010–2019; validation: 2020–2022.
Inputs: 120 sessions × six features, flattened to 720 columns.
Output: next-session price change in dollars.

Selected depth: 4.
Selected boosting rounds: 98.
Validation MAE: 2.073923.
Validation RMSE: 2.781877.

All candidate reload checks passed.
Use iteration_range=(0, 98) for the selected model's predictions.
No refitting on validation data; final test remains unevaluated.

Run: artifacts/final_xgboost/20260920T115749_481182Z

### Final ARIMA selection

Selected ARIMA(1,1,1), without trend, using validation RMSE.
Validation MAE: 2.077471.
Validation RMSE: 2.790839.

All five candidate checks passed.
ARIMA(0,1,0) reproduced naive predictions.

Coefficients were fitted through 2019.
Filtering state was updated sequentially through 2022.
No coefficient refitting occurred during validation.

Run: artifacts/final_arima/20260920T120716_481620Z
Final-test evaluation has not started.

### Final-test error analysis

Attention-LSTM achieved slightly lower one-session MAE and RMSE
than naive in each of 2023, 2024 and 2025. Its pooled advantage
was marginal, and statistical reliability has not been established.

Plain LSTM improved on naive in 2023 and 2024 at both horizons,
but its deterioration in 2025 outweighed those gains.

All evaluated models had their largest yearly dollar errors in
2025. Forecast plots show substantial shared error spikes around
spring 2025, with learned-model errors closely tracking naive errors.

The results demonstrate limited improvement over persistence,
rather than reliable anticipation of large price movements.
Five-session errors overlap and should not be treated as independent.

These analyses describe the frozen final-test results.
No model changes were made in response to them.

## Reproducibility

See [Reproducibility Guide](docs/reproducibility.md) for environment
restoration, saved model locations, evaluation commands, and backup
instructions.

## Project Status

Completed — AAPL forecasting and evaluation scope.

Implemented data preparation, LSTM forecasting, baseline comparisons,
development experiments, frozen final-test evaluation, and bootstrap
uncertainty analysis.

Final-test period: 2023–2025.

Standard LSTM did not outperform naive forecasting in pooled test
results. Attention-LSTM achieved small observed one-session gains,
but bootstrap intervals did not establish superiority.

Documentation:
- [Final Results](docs/final_results.md)
- [Reproducibility Guide](docs/reproducibility.md)
- [Final Evaluation Protocol](docs/final_evaluation_protocol.md)

Additional tickers, individual prediction intervals, ten-session
forecasting, and trading backtests are future work outside the
completed scope.