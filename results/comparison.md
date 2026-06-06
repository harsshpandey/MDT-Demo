# MDT — Hourly vs 15-min vs Paper Comparison

All metrics are on the % of rated capacity scale (predicted × 100, actual × 100), matching Liu et al. 2024 convention.

## Single-twin best result

| Run | Granularity | Best DT | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|---|---|
| Hourly | 60 min | GRU | 4.468 | 6.642 | 0.9318 | 7.266 |
| 15-min | 15 min | GRU | 0.585 | 0.767 | 0.9991 | 1.121 |
| Paper (Liu 2024) | 15 min | GRUCNN | 2.542 | 4.566 | 0.8895 | — (n/a) |

## Method 1 (sliding RMSE) — best fusion

| Run | Combo | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|---|
| Hourly | LSTM&GRU | 4.698 | 6.985 | 0.9248 | 7.243 |
| 15-min | LSTM&GRU | 0.583 | 0.773 | 0.9991 | 1.124 |
| Paper (Liu 2024) | LSTM&GRU&LSTMCNN | 2.435 | 4.357 | 0.8995 | — (n/a) |

## Method 2 (Dempster-Shafer multi-metric) — best fusion

| Run | Combo | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|---|
| Hourly | GRU&LSTMCNN | 4.600 | 6.574 | 0.9334 | 7.392 |
| 15-min | LSTM&GRU | 0.564 | 0.767 | 0.9991 | 1.150 |
| Paper (Liu 2024) | LSTM&GRU&LSTMCNN&GRUCNN | 2.387 | 4.330 | 0.9008 | — (n/a) |

## Interpretation

1. **MAE is a percentage of installed capacity.** Paper used 129.1 MW;
   we use a single 5 MW NREL turbine. Smaller denominator = relative
   error inflates even when absolute kW error is small.
2. **Higher autocorrelation at 15-min** = easier next-step problem.
   Liu et al. 2024 effectively predict `y(t+15m) ≈ y(t) + δ`.
3. **R² is unitless** and the cleanest direct comparison.
