# Multi-Digital-Twin (MDT) Wind-Power Forecasting — NREL India Reproduction

> Reproducible implementation of Liu et al. 2024 — *"Research on multi-digital twin and its application in wind power forecasting"* (Energy 292, 130269) — applied to **real NREL India Wind Toolkit data**, site 36565, Ahmedabad (Lat 23.03°N, Lon 72.56°E), year 2014.

---

## What the project delivers

1. **Honest reproduction** of the paper's MDT method on Indian wind data. No target-lag leakage. No prediction calibration. Single canonical fusion engine.
2. **Two granularities tested side by side:**
   - **Hourly** — native measurement cadence (8 760 samples/year).
   - **15-min** — cubic-interpolated to match the paper's measurement cadence (35 040 samples/year).
3. **Per-feature physical reasoning** — every engineered input is documented with its physics motivation in `data_pipeline.py::FEATURE_DOC`.
4. **Dempster–Shafer fusion implemented per paper Eq 13–17** with unit tests on toy inputs.
5. **Flask + Chart.js dashboard** at `http://localhost:5001` showing data understanding (EDA), training, single-twin metrics, all 22 fusion combinations, and the comparison vs paper.
6. **Reproducible by one command:** `make all && make serve`.

---

## Quick start

```bash
make install     # venv + pinned deps
make eda         # writes results/eda_report.md (data understanding)
make data        # hourly features → results/hourly/
make train       # train 4 twins on hourly → predictions/
make data15      # 15-min resampled features → results/min15/
make train15     # train 4 twins on 15-min → predictions_min15/
make eval        # results/eval_matrix.csv
make fuse        # combination_results.csv (22 fusion results)
make compare     # hourly vs 15-min vs paper → results/comparison.md
make serve       # http://localhost:5001
make test        # pytest — 21 tests (leakage detector + fusion math + physics)
```

Single shot:

```bash
make all && make serve
```

---

## Architecture

```
data/raw/36565_…_2014.csv  ─┐
                            │  ┌────────── hourly path ─────────┐
                            ├─→│ engineer_features (40 cols)    │→ predictions/*.csv
                            │  └────────────────────────────────┘
                            │  ┌────────── 15-min path ─────────┐
                            └─→│ resample_to_15min →            │→ predictions_min15/*.csv
                               │ engineer_features (42 cols)    │
                               └────────────────────────────────┘
                                            │
                  ┌─────────────────────────┴─────────────────────────┐
              method1_single_metric                     method2_multimetric_ds
              (paper Eq 8-9, sliding RMSE)              (paper Eq 13-17, DS theory)
                                            │
                                app.py + Chart.js dashboard
```

---

## Why each design choice

### Feature engineering — 40 columns, every one justified

`data/pipeline.py::FEATURE_DOC` is a single dictionary mapping feature name → physical reasoning. Categories:

| Group | Examples | Physical basis |
|---|---|---|
| Wind speed @ 4 heights | `wind_speed_{40,80,100,120}m` | Hub at 80 m drives turbine; multi-height for shear |
| Cubic powers | `wind_speed_80m_sq`, `wind_speed_80m_cb` | Power-law `P ∝ v³` (NREL region II) |
| Profile shear | `ws_ratio_*`, `wind_shear`, `wind_dir_shear` | α exponent; Ekman spiral |
| Thermal | `temp_{80,120}m`, `temp_gradient` | Density via `ρ = P/RT`; inversion proxy |
| Pressure | `pressure_100m`, `pressure_diff` | Density input; front detection |
| Density | `air_density`, `wind_power_density` | Direct multiplier in `P = ½ρAv³` |
| Direction | `wind_dir_sin/cos` | Cyclic encoding (359° ≈ 1°) |
| Time | `hour_*`, `month_*`, `doy_*` | Diurnal (lag-24 acf = 0.66 on Indian data) + seasonal |
| Rolling stats | `ws_roll_{mean,std,max,min}_{3,6,24}h` | Persistence at multiple horizons |
| Dynamics | `ws_diff_1`, `ws_diff_2`, `turbulence_intensity` | Wind ramp velocity + acceleration |
| Wind-speed lags | `ws_lag_{1,2,3,6}` | Exogenous autocorrelation (NOT target leakage) |

**`wp_lag_*` (wind-power target lag) is explicitly forbidden** — a CI test (`tests/test_no_leakage.py`) scans for re-introduction.

### Diversity injection — different feature views per twin

All twins are trained on the **same data** but with **different feature subsets**, forcing decorrelated predictions (random-subspace ensemble, Ho 1998). Without this, all four twins converge and fusion buys nothing.

| Twin | Feature filter | Why |
|---|---|---|
| **LSTM** | all 40 | Generalist baseline |
| **GRU** | drop rolling stats (31 features) | Forces reliance on raw + lag signal |
| **LSTMCNN** | drop wind-speed lags + diffs (34 features) | Focuses on instantaneous + smoothed |
| **GRUCNN** | all 40, dropout 0.35 | Stochastic regularization |

### Why 15-min comparison

Liu et al. used **15-minute** samples from a Chinese wind farm. R² ≈ 0.89 in their paper is partly because the next-step problem is much easier at that cadence (lag-1 autocorrelation ≈ 0.99). To compare apples to apples we cubic-interpolate hourly Indian data to 15-min. Caveats:

- Interpolation is **not new information**. We do not gain insight into sub-hourly turbulence.
- Metrics on resampled data represent an **upper bound** of what would be achievable if the underlying physics were natively measured at 15-min.

Documented in `data_pipeline.resample_to_15min` docstring.

### Fusion math — paper Eq 13–17

`mdt_engine.py` implements both methods. All five equations have unit tests on toy inputs:

```python
# Eq 13 — BPA for RMSE/MAE (smaller is better)
m_i = (1/M_i) / Σ(1/M_j)

# Eq 14 — BPA for R² (larger is better)
m_i = M_i / Σ M_j

# Eq 15 — Dempster combination
m(DT_l) = Σ{m_a(DT_l)·m_b(DT_l)} / (1 − K)

# Eq 16 — variance threshold
δ² > ζ → winner-take-all
δ² ≤ ζ → weighted average

# Eq 17 — final
D̃ = Σ m(DT_i)·D̃_i
```

---

## Reference numbers

Final metrics are written to `results/comparison.md` by `make compare` after both training paths complete. Format:

| Run | Granularity | Best DT | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Hourly real | 60 min | … | … | … | … |
| 15-min real | 15 min | … | … | … | … |
| Paper Liu 2024 | 15 min | GRUCNN | 2.5423 | 4.5663 | 0.8895 |

---

## File reference

| File | Purpose |
|---|---|
| `data_pipeline.py` | NREL CSV loader, optional 15-min cubic resample, 40-feature engineering, MinMax + sequential split |
| `eda.py` | Generates `results/eda_report.md` + `results/eda_stats.json` |
| `models.py` | LSTM / GRU / LSTMCNN / GRUCNN PyTorch defs |
| `train.py` | Seeded training, per-twin feature filter, AdamW + Huber + early stop |
| `evaluate.py` | Writes `results/eval_matrix.csv` from saved predictions |
| `mdt_engine.py` | Fusion math (Method 1 + Method 2), canonical MAPE |
| `compare.py` | Builds `results/comparison.md` + `comparison.json` |
| `app.py` | Flask backend, 15 routes |
| `templates/index.html` | Chart.js dashboard |
| `tests/test_no_leakage.py` | CI guard — blocks `adjust.py` regression |
| `tests/test_fusion_math.py` | Toy-input tests for Eq 13–17 |
| `tests/test_data_pipeline.py` | Schema + physics correctness |
| `Makefile` | Workflow targets |
| `requirements.txt` | Pinned versions |

---

## Reference

Liu, S., Tian, J., Ji, Z., Dai, Y., Guo, H., Yang, S. (2024). Research on multi-digital twin and its application in wind power forecasting. *Energy*, 292, 130269. <https://doi.org/10.1016/j.energy.2024.130269>
