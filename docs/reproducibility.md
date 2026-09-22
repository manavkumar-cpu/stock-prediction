# Reproducibility Guide

## 1. Restore the Environment

Run commands from the project root.

Create a virtual environment if this is a new device:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

For an existing environment:

```bash
source .venv/bin/activate
```

Verify the installation:

```bash
python --version
python -m pip check
python -c "import sys; print(sys.executable)"
```

The Python executable should belong to this project's `.venv`.

The dependency snapshot was generated on an Apple Silicon Mac.
Other operating systems or processor architectures may require
dependency adjustments.

Do not transfer `.venv` between devices. Recreate it using the
recorded dependencies.

## 2. Preserve the Experiment Inputs

Keep the original downloaded data, generated feature tables,
configuration snapshots, preprocessing objects, model checkpoints,
and evaluation outputs.

Do not download replacement data over the frozen source files.
A later Yahoo Finance download may differ from the original
historical snapshot.

The base configuration describes defaults. Individual experiment
snapshots record the settings actually used for those experiments,
including overrides for window length and forecast horizon.

## 3. Final Development Data

Prepared datasets:

```text
data/processed/final_development/20260920T112216_338230Z/
```

Dataset variants:

| Dataset | Horizon | Window |
|---|---:|---:|
| h1_w120 | 1 session | 120 sessions |
| h1_w60 | 1 session | 60 sessions |
| h5_w120 | 5 sessions | 120 sessions |

These folders contain training and validation arrays, date metadata,
and training-fitted preprocessing.

The final-test evaluator reconstructs test inputs using the frozen
preprocessing.

## 4. Selected Model Artifacts

Neural models:

```text
artifacts/final_neural/20260920T112518_629227Z/
```

This run contains:

- LSTM_h1_w120
- Attention_LSTM_h1_w120
- GRU_h1_w60
- LSTM_h5_w120

XGBoost:

```text
artifacts/final_xgboost/20260920T115749_481182Z/
```

Selected configuration: depth 4, 98 boosting rounds.

ARIMA:

```text
artifacts/final_arima/20260920T120716_481620Z/
```

Selected order: ARIMA(1, 1, 1).

Use the frozen manifest as the authoritative mapping from models
to checkpoints and preprocessing files.

## 5. Frozen Manifest

Manifest:

```text
artifacts/final_evaluation/20260920T121603_146385Z/frozen_manifest.json
```

Recorded SHA-256:

```text
65f4e588c479dacbf8be41d1ae83b8d21e46745d79fc4169d025739fd2d39803
```

The manifest records fingerprints for 72 files.

The evaluator verified the manifest and tracked files before
calculating final-test predictions.

Fingerprint checks detect changed files. They do not independently
prove that the implementation is correct.

## 6. Original Final-Test Results

Results directory:

```text
artifacts/final_evaluation/20260920T121603_146385Z/test_results_20260920T142042_817122Z/
```

Important files:

| File | Contents |
|---|---|
| pooled_metrics.csv | Overall results for each horizon |
| all_metrics.csv | Overall and yearly results |
| h1_predictions.csv | One-session forecasts and actual prices |
| h5_predictions.csv | Five-session forecasts and actual prices |
| h1_forecasts.png | One-session forecast and error plots |
| h5_forecasts.png | Five-session forecast and error plots |
| frozen_manifest.json | Manifest copied into the result directory |
| execution.json | Evaluation execution metadata |
| completion.json | Completion status and sample counts |

The original evaluation used:

```bash
python -m src.evaluation.evaluate_final \
  --manifest artifacts/final_evaluation/20260920T121603_146385Z/frozen_manifest.json
```

Running this again is a reproducibility check on an already observed
test period. It does not create a new independent test.

Preserve the original result directory when comparing repeated runs.

## 7. Bootstrap Analysis

Original bootstrap output:

```text
artifacts/final_evaluation/20260920T121603_146385Z/test_results_20260920T142042_817122Z/bootstrap_analysis/20260920T143356_519943Z/
```

The analysis used:

```bash
python -m src.evaluation.bootstrap_final \
  --results-dir artifacts/final_evaluation/20260920T121603_146385Z/test_results_20260920T142042_817122Z
```

Settings:

- 5,000 paired repetitions.
- Circular blocks of 5, 10, and 20 sessions.
- Sampling within target year.
- Seed 42, with block-specific offsets.
- Individual 95% percentile intervals.
- Metric differences calculated as model minus naive.

The saved bootstrap summary reports uncertainty in metric
differences. It is not a set of individual price prediction intervals.

## 8. Retraining Versus Repeating Evaluation

Repeating evaluation loads saved checkpoints and preprocessing.

Retraining fits new model parameters and creates a new experiment.
Even with the same seed, results may vary across software versions,
hardware, or nondeterministic operations.

Record new run identifiers and preserve the original checkpoints.

Do not tune models using the observed final-test results and then
describe that same period as an untouched test set.

## 9. Moving or Backing Up the Project

Copy:

- Source code and notebooks.
- Configuration files and documentation.
- requirements.txt and requirements-lock.txt.
- Raw and processed datasets.
- Saved models, scalers, and manifests.
- Reports, prediction tables, and figures.

Exclude `.venv` and regenerable Python caches.

Git alone will not preserve files excluded by `.gitignore`.
Datasets and model artifacts need a separate backup if ignored.

Some metadata may contain paths from the original computer.
Keep the project-relative directory structure and inspect any
path-resolution error before changing files.

Do not alter fingerprinted files merely to make verification pass.
Document any required migration separately.

## 10. Related Documents

- [Project Plan](project_plan.md)
- [Final Evaluation Protocol](final_evaluation_protocol.md)
- [Final Results](final_results.md)