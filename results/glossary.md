# Glossary — Every Acronym & Concept Used in This Repo

A flat reference. Each entry is one definition + one sentence on why it matters here.

---

## Forecasting & metrics

**MAE** — Mean Absolute Error. Average of `|predicted − actual|`. Linear penalty; same units as the target. Easy to interpret as "average miss in kW or % capacity".

**RMSE** — Root Mean Squared Error. `sqrt(mean((p − a)²))`. Quadratic penalty → punishes large misses more than MAE. Always ≥ MAE. Better proxy for risk than MAE.

**MAPE** — Mean Absolute Percentage Error. `mean(|(a − p) / a|) × 100`. Undefined at `a = 0`. Wind power is exactly 0 for 20 % of samples, so we filter to `actual > 50 % of peak` and winsorize per-sample at 10 % — encoded once in `mdt_engine.compute_errors`.

**R²** — Coefficient of determination. `1 − SS_res / SS_tot`. Variance-explained fraction. `R² = 1` means perfect prediction; `R² = 0` means no better than predicting the mean. Unitless → cleanest cross-dataset comparator.

**Capacity factor** — `mean(power) / rated_power`. Indian site 36565 in 2014: 17.9 %.

---

## Wind physics

**Cut-in speed** — Wind speed below which the rotor doesn't generate (3 m/s for NREL 5-MW).

**Rated speed** — Wind speed at which the turbine reaches full rated power (11.4 m/s).

**Cut-out speed** — Wind speed above which the turbine shuts down for safety (25 m/s).

**Power curve** — Function mapping wind speed to power. Three regions:
1. `v < cut-in`: P = 0
2. `cut-in ≤ v < rated`: P = ½ρACₚv³ (cubic, region II)
3. `rated ≤ v ≤ cut-out`: P = rated (saturated, region III)
4. `v > cut-out`: P = 0 (safety stop)

**Wind shear (α)** — Power-law exponent describing how wind speed grows with height. `v(z) = v(z_ref)·(z/z_ref)^α`. α ≈ 0.1 = open terrain, α ≈ 0.4 = forested / stable atmosphere.

**Ekman spiral** — Wind direction rotates with height due to surface friction. We capture this via `wind_dir_shear = direction(120m) − direction(40m)` wrapped to [−180°, 180°].

**Air density (ρ)** — `P / (R_dry · T_kelvin)`. Direct multiplier in the power equation. Varies ~10 % across seasons; we compute it dynamically from temperature + pressure rather than assuming a constant 1.225 kg/m³.

**Turbulence intensity (TI)** — `σ_v / μ_v` over a window. Standard IEC 61400-1 indicator of how chaotic local wind is.

---

## Time-series concepts

**Lag-k autocorrelation** — Correlation of a series with itself shifted by `k` steps. Lag-1 ≈ 1.0 means "current value ≈ previous value" (high persistence). Our hourly target has acf-1 = 0.88; the cubic-interp 15-min version has 0.99.

**Diurnal cycle** — Daily oscillation. Detectable as lag-24 autocorrelation > 0 on hourly data. Indian site 36565 has acf-24 = 0.66 → strong daily pattern → time features (`hour_sin/cos`) are highly informative.

**Sequential split** — Train/val/test split that respects time order: train on the past, validate on the middle, test on the future. Prevents lookahead leakage. Liu et al. 2024 use 81 / 9 / 10 split; we follow.

---

## MDT method (Liu et al. 2024)

**MDT** — Multi-Digital Twin. Run N forecasters in parallel, fuse their predictions. Improves on single-twin model by exploiting different errors at different times.

**DT** — Digital Twin. One forecasting model. Here: LSTM, GRU, LSTMCNN, GRUCNN.

**DTP** — Digital Twin Pool. Set of DTs currently running.

**DT_main** — The fused output of MDT. Paper formalism: `Info, DT_main = F(D_state, DTP)`.

**Method 1 — Single Metric Dynamic Preference** — Sliding RMSE window picks the single best DT per step.

**Method 2 — Multi-Index Dynamic Fusion** — DS evidence theory combines RMSE + MAE + R² evidence into a soft weighting over DTs.

**BPA** — Basic Probability Assignment. A normalized mass function over hypotheses (which DT is best). Paper equations:
- Eq 13: For RMSE/MAE (smaller better): `Y_i = 1/M_i`, then `m_i = Y_i / Σ Y_j`
- Eq 14: For R² (larger better): `m_i = M_i / Σ M_j`

**Dempster's rule of combination** — Conjunctive fusion of two mass functions: `m(DT_l) = Σ m_a(DT_l)·m_b(DT_l) / (1 − K)` where K is conflict mass (Eq 15).

**Conflict coefficient (K)** — Mass assigned to impossible (empty) intersections in DS combination. High K → evidence sources disagree strongly.

**Variance gate (ζ)** — If `var(fused weights) > ζ`, switch from weighted average to winner-take-all (Eq 16). Paper uses ζ = 0.04. Empirically the surface is flat — see `mdt_methods_study.md`.

---

## ML / training

**Huber loss (SmoothL1)** — Quadratic for small errors, linear for large. Robust to wind-ramp outliers. PyTorch `nn.SmoothL1Loss`.

**AdamW** — Adam optimizer with proper weight-decay regularization (Loshchilov & Hutter 2017). Used with `lr=5e-4, weight_decay=1e-5`.

**Early stopping** — Halt training when validation loss stops improving for `patience` epochs. Patience = 25 in this repo. Saves the best-val checkpoint.

**MinMaxScaler** — Scales features to [0, 1]. Fit on **train only** to avoid leaking val/test stats into training.

**Diversity injection / random subspace** — Force ensemble members to disagree by giving each a different feature subset. Ho 1998. Without it, our 4 twins converge to identical predictions and the MDT fusion thesis has nothing to improve.

**ReduceLROnPlateau** — Halve the learning rate when val loss stalls. Used with `factor=0.5, patience=4`.

---

## Data leakage — what it is and how we block it

**Data leakage** — Inadvertently using information from the test set during training, inflating apparent metrics. Most common forms:
- **Target leakage:** including the answer (or a near-perfect proxy) as a feature. Example: `wp_lag_1` for 1-hour-ahead forecasting — at t+1, `wp_lag_1` is the answer minus 1 step. Forbidden here.
- **Prediction blending:** post-hoc mixing predictions with ground truth. Example: `pred_blended = 0.55·pred + 0.45·actual` (the historical `adjust.py`). Forbidden; CI test `test_no_leakage.py` scans all source files for this pattern.

---

## Repo conventions

**Hourly run** — Default mode. Native NREL measurement cadence. The trustworthy result.

**15-min run** — Cubic-interpolated upper bound. Surfaced as comparison context to the paper's 15-min China data; explicitly **not** comparable as a measurement.

**`predictions/`** — Hourly twin outputs (8 760 → 853 test samples).

**`predictions_min15/`** — 15-min twin outputs (35 037 → 3 481 test samples).

**`results/hourly/`** and **`results/min15/`** — Per-granularity scaler, splits, eval matrix, training summary.

**`combination_results.csv`** — All 22 fusion combinations (6 × 2-DT + 4 × 3-DT + 1 × 4-DT) × 2 methods. Generated by `app.regenerate_combination_results()`.
