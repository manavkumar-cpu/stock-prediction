# Stock Price Prediction using LSTM — Project Plan

## Objective
Build and evaluate an LSTM forecasting system that predicts the
next trading session's closing price. Compare it against simple
and advanced baselines without data leakage.

## Data
- Source: Yahoo Finance through yfinance
- Primary ticker: AAPL
- Additional tickers later: MSFT and GOOGL
- Period: January 2010–December 2025
- Download start: 2010-01-01
- Download end: 2026-01-01 (exclusive)
- Frequency: Daily
- Download with auto_adjust=False
- Preserve OHLCV and Adj Close
- Main price series: Yahoo Close, which is split-adjusted

## Prediction Task
- Issue predictions after the current session's data is available.
- Main target: Next trading session's Close
- Initial forecast horizon: 1 trading session
- Later horizons: 5 and optionally 10 trading sessions
- Initial input window: 60 trading sessions
- Compare windows of 20, 30, 60, and 120 sessions in Phase 15.
- Select the window using validation performance.
- Keep validation target dates consistent across window comparisons.

## Features
Introduce features progressively:
1. Close only
2. OHLCV
3. OHLCV plus technical indicators
4. Previous features plus optional market context

Candidate indicators:
- Returns
- SMA and EMA
- RSI and MACD
- Rolling volatility
- Relative volume
- High-Low range

Candidate market context:
- S&P 500
- NASDAQ Composite
- VIX

Use only information available when each forecast is issued.

## Models
Primary model:
- TensorFlow/Keras LSTM

Comparison models:
- Naive: Next session's price equals today's Close
- ARIMA
- GRU
- XGBoost
- Attention-LSTM

## Data Splits
- Training: 2010–2019
- Validation: 2020–2022
- Final test: 2023–2025
- Never randomly split time-series observations.
- Assign examples by target dates.
- Exclude examples whose targets cross partition boundaries.
- Input windows may include earlier observed history.
- Fit scalers and other learned preprocessing only on training data.

## Evaluation
- Primary metrics: MAE and RMSE
- Additional metrics: MSE, MAPE, and R²
- Calculate price metrics in the original price scale.
- Compare models on matching forecast dates and horizons.
- Plot actual versus predicted prices.
- Use validation data for all development decisions.
- Keep final-test results out of model and feature selection.
- Perform development walk-forward validation before the test period.
- Evaluate the frozen experimental design on the final test set.

## Advanced Analysis
- Multi-horizon forecasting
- Expanding-window walk-forward validation
- Feature ablation
- Market-regime analysis
- Error analysis
- Uncertainty estimation
- Optional backtesting with transaction costs and no look-ahead bias

## Working Method
- Follow Phases 0–27 in the agreed order.
- Implement one phase at a time.
- Understand concepts before implementing them.
- Review code and outputs before advancing.
- Record actual results, including unsuccessful experiments.
- Keep LSTM central to the project.