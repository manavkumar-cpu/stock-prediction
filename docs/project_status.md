# Project Status

## Completed Scope

An end-to-end AAPL forecasting system was implemented using
2010–2025 daily historical data.

The project includes data validation, exploratory analysis,
feature engineering, chronological preprocessing, LSTM training,
baseline comparisons, development walk-forward evaluation,
feature ablation, regime analysis, multi-horizon forecasting,
and frozen final-test evaluation.

Final-test results and paired block-bootstrap analyses are complete.

## Main Finding

The standard LSTM did not outperform naive forecasting in pooled
final-test results at either evaluated horizon.

Attention-LSTM achieved small observed one-session improvements,
but its bootstrap intervals did not establish superiority over naive.

These findings are retained without tuning against final-test results.

## Future Work — Outside Completed Scope.

- Additional-ticker experiments for MSFT and GOOGL.
- Individual forecast prediction intervals, if included.
- A fully nested feature-group comparison, if included.
- Optional ten-session forecasting.
- Optional trading backtest with transaction costs.
- Git setup, previously deferred.

Documentation files should be saved and checked against the
actual project directory before handover.

## Evaluation Boundary

The AAPL 2023–2025 test period has been observed.

Any subsequent model changes are new development work.
Reusing this period after tuning does not constitute a fresh,
untouched final test.

## Project Positioning

This is a forecasting and evaluation project.

It demonstrates reproducible experimentation, leakage controls,
baseline comparisons, and honest interpretation of results.

It does not establish a profitable trading strategy or a reliable
forecasting advantage from LSTM.