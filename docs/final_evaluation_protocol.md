# Final Evaluation Protocol — AAPL

## Objective
Evaluate the selected forecasting models on untouched 2023–2025
target dates. LSTM remains the primary project model; naive is
the reference baseline.

## Data
Use the saved AAPL data and feature snapshots.
Do not refresh downloads for this evaluation.
The price target is Yahoo's split-adjusted Close.
Historical downloads are retrospectively adjusted and are not
a point-in-time reconstruction of historical data vintages.

Forecasts are issued after the current session's required data
is available.

## Partitions
Assign examples by target dates:
- Training: 2010–2019.
- Validation: 2020–2022.
- Final test: 2023–2025.

For horizon five, all future sessions t+1 through t+5 must lie
within one partition. Each partition is a continuous period;
internal calendar-year boundaries do not require exclusions.

Input windows may contain earlier observed history.
Never randomly split observations.

This fixed-split evaluation is separate from development
expanding-window experiments.

## Features and preprocessing
Features, in order:
Close, Return_1, RSI_14, Volatility_20,
Relative_Volume_20, Range_Ratio.

Center each Close window on its final observed Close and divide
by the training input Close range.

Standardize the remaining features using unique training input
rows only. Preserve session spacing and exclude incomplete windows.
Do not refit preprocessing or clip later values to training ranges.

Neural targets are future Close minus current Close, divided by
the training target Close range. Reconstruct prices before scoring.

XGBoost predicts the price change directly in dollars.
ARIMA predicts the price directly in dollars.

## Selected models
One-session models:
- Naive.
- Training-mean-change baseline.
- LSTM: 120 sessions, selected epoch 35.
- Attention-LSTM: 120 sessions, selected epoch 10.
- GRU: 60 sessions, selected epoch 9.
- XGBoost: 120 sessions flattened to 720 columns,
  depth 4, prediction iteration range [0, 98).
- ARIMA(1,1,1), without trend.

Five-session models:
- Naive.
- Training-mean-change baseline.
- LSTM: 120 sessions, selected epoch 34.

Neural settings:
32 recurrent units, Adam learning rate 0.001, batch size 32,
maximum 100 epochs, patience 10, seed 42, shuffle=False.
Checkpoints were selected by minimum validation MSE.

Use the exact saved checkpoints identified in the frozen manifest.
Do not retrain on validation observations.

## Baselines
Naive predicts the forecast-date Close for either horizon.

Use one common training-mean-change baseline per horizon:
- h1: training examples from h1_w120.
- h5: training examples from h5_w120.

GRU's different window and early training sample coverage must
be documented; it is not an isolated architecture comparison.

## ARIMA state
Coefficients were fitted through 2019.
The selected saved state incorporates observations through
2022-12-30.

During test, forecast before incorporating each actual observation.
Update filtering state after observation without refitting coefficients.

## Test execution
Verify frozen file fingerprints before evaluation.
Keep weights, coefficients and preprocessing fixed.
Earlier observed test sessions may become later forecast inputs.
Never supply an unrealized target to a model.

Generate five-session forecasts directly without intervening
future actual prices.

Evaluate every listed model and retain unsuccessful results.
Do not use test metrics to select features, models or checkpoints.

## Reporting
Compare models on matching forecast and target dates within
each horizon. Verify actual-price alignment.

Report pooled and yearly:
MAE, RMSE, MSE, MAPE, R2 and mean error.
Save actual-versus-predicted and error plots.

Price-error metrics use dollars.
Do not directly rank different horizons by raw RMSE.

Bootstrap metric intervals are conditional on saved predictions.
Overlapping five-session errors are not independent.
Metric uncertainty intervals are not individual prediction intervals.

## Interpretation and scope
Validation results were used for selection and are not independent
final-test performance.

Report all observed results without assuming LSTM beats naive.
An interval containing zero does not establish equivalence.

Any subsequent test-informed changes are exploratory and cannot
reuse this period as an untouched final test.

This evaluation covers AAPL. Additional stocks, ten-session
forecasts and trading backtests are separate extensions.

Feature experiments compared alternative feature sets, rather
than a complete nested OHLCV-plus-all-indicators sequence.
