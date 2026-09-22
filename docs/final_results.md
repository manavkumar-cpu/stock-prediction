# Stock Price Forecasting — Final Results

## 1. Objective

Predict AAPL’s future closing price using an LSTM and compare it
against simple and advanced forecasting baselines.

The primary horizon is one trading session. A five-session horizon
is evaluated separately.

Forecasts use information available after the forecast session’s
data is available.

## 2. Data and Evaluation Design

- Source: Yahoo Finance through yfinance.
- Ticker: AAPL.
- Period: 2010–2025.
- Raw observations: 4,024 trading sessions.
- Target series: Yahoo Close, on its split-adjusted price basis.
- Training period: 2010–2019.
- Validation period: 2020–2022.
- Final-test period: 2023–2025.

Development included expanding-window evaluation before the final
test period. Within each development fold, an earlier selection
period was used for checkpoint or hyperparameter selection.

Final models were fitted using training data and selected using
validation data. Model weights and learned preprocessing were not
refitted on validation data before final-test evaluation.

Models, preprocessing, and supporting files were fingerprinted
before calculating final-test predictions.

During testing, earlier observed prices could enter later input
windows. ARIMA states were updated after observing each actual
value, while its fitted coefficients remained fixed.

## 3. Features and Models

The final neural and XGBoost experiments used:

1. Close
2. One-session return
3. RSI over 14 sessions
4. Rolling volatility over 20 sessions
5. Relative volume over 20 sessions
6. High–Low range ratio

LSTM and Attention-LSTM used 120-session windows.
GRU used a 60-session window; this protocol difference is retained
when interpreting comparisons.

The residual neural models predicted a price correction:

Predicted future Close = current Close + predicted correction

The Close input channel was centered relative to the last observed
Close in each window. Scaling parameters were estimated using
training data only.

Comparison models:

- Naive: future Close equals current Close.
- Training-mean-change: current Close plus the mean historical
  target change estimated from training examples.
- ARIMA.
- GRU.
- XGBoost.
- Attention-LSTM.

## 4. Development Findings

The initial direct-price LSTM substantially underpredicted higher
validation prices. Its validation MAE was 5.7887 USD and RMSE was
7.3681 USD, compared with 2.0762 USD and 2.7913 USD for naive.

Using relative price history and predicting a correction greatly
reduced this systematic error. However, much of the improvement
came from retaining the current price as the forecast baseline.

Development experiments investigated:

- Input windows of 20, 30, 60, and 120 sessions.
- Close-only, OHLCV, technical features, and market context.
- LSTM, GRU, Attention-LSTM, ARIMA, and XGBoost.
- Feature ablation.
- One-session and five-session targets.
- Trend and volatility regimes.
- Bootstrap uncertainty in metric differences.

These experiments did not establish a consistent development
advantage over naive forecasting. Unsuccessful experiments were
retained as part of the project record.

## 5. Final-Test Results: One Session

All models were evaluated on the same 752 target dates.

| Model | MAE (USD) | RMSE (USD) |
|---|---:|---:|
| Naive | 2.218311 | 3.253173 |
| Training-mean-change | 2.215809 | 3.251719 |
| LSTM | 2.219998 | 3.256531 |
| Attention-LSTM | 2.211940 | 3.250395 |
| GRU | 2.214642 | 3.253802 |
| XGBoost | 2.216673 | 3.254608 |
| ARIMA | 2.218617 | 3.255006 |

Attention-LSTM achieved the lowest observed MAE and RMSE.
Relative to naive, its improvements were approximately 0.287%
in MAE and 0.085% in RMSE.

The standard LSTM had slightly higher pooled MAE and RMSE than
naive.

These observed rankings do not establish statistically reliable
superiority.

## 6. Final-Test Results: Five Sessions

All models were evaluated on the same 748 target dates.

| Model | MAE (USD) | RMSE (USD) |
|---|---:|---:|
| Naive | 5.752500 | 7.745930 |
| Training-mean-change | 5.733150 | 7.730287 |
| LSTM | 5.767080 | 7.779516 |

Examples whose five-session target spans crossed the
validation/test boundary were excluded.

The LSTM improved on naive in 2023 and 2024 but performed worse
in 2025. Its pooled errors exceeded naive at this horizon.

The training-mean-change baseline had the lowest pooled MAE
and RMSE among these three models.

## 7. Bootstrap Uncertainty

A paired circular block bootstrap compared each model with naive.

- Repetitions: 5,000 per block length.
- Block lengths: 5, 10, and 20 sessions.
- Sampling: within target year, preserving yearly sample counts.
- Intervals: individual 95% percentile intervals.
- Difference: model metric minus naive metric.
- Negative differences favor the model.

For all trained models, MAE and RMSE intervals included zero
across all evaluated block lengths.

Attention-LSTM therefore had a small observed improvement, but
the bootstrap did not establish an advantage over naive.

The training-mean-change baseline’s MAE intervals were entirely
below zero at both horizons across all three block lengths.
Its RMSE intervals included zero.

The intervals were not adjusted for multiple comparisons.
Results across block lengths are sensitivity checks, not
independent confirmations.

The bootstrap conditions on the saved predictions. It does not
measure variability from retraining, and it does not provide
prediction intervals for individual future prices.

## 8. Interpretation

A visually close match between predicted and actual prices,
or a high R², is insufficient evidence of useful forecasting
skill. Naive forecasts also closely track price levels.

The residual formulation corrected much of the direct-price
model’s systematic underprediction. Nevertheless, the learned
corrections did not demonstrate a reliable advantage over naive
under the evaluated protocol.

Lower mean error alone did not guarantee lower MAE or RMSE.

The project demonstrates an end-to-end forecasting and evaluation
pipeline, including honest reporting of small and unsuccessful
improvements.

## 9. Limitations

- Only AAPL was evaluated.
- The data is a downloaded historical snapshot, not a
  point-in-time archive of historical data vintages.
- Yahoo’s historical prices incorporate retrospective adjustments.
- Repeated development experiments can overfit validation results.
- Neural results do not characterize variability across many
  independent training seeds.
- GRU and the final LSTM models used different window lengths.
- Overlapping forecast windows create dependent errors.
- Bootstrap intervals depend on the resampling assumptions and
  do not preserve dependence across year boundaries.
- Trading profitability and transaction costs were not evaluated.
- Individual forecast prediction intervals were not implemented.

## 10. Conclusion

Under the evaluated protocol, the standard LSTM did not outperform
naive forecasting in pooled final-test results at either horizon.

Attention-LSTM achieved slightly better observed one-session
metrics, but its bootstrap intervals did not establish superiority.

A simple training-derived constant-change baseline produced small,
consistent MAE improvements under the bootstrap analysis.

Further experiments must be identified as new development work.
The observed 2023–2025 test period must not be reused as an untouched
test set after tuning models against these results.