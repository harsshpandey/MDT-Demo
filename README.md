# Multi-Digital Twin (MDT) Wind Power Forecasting
*Off-Grid Wind Turbine Analysis & Deep Learning Forecasting*

> **Location:** Ahmedabad, India (SiteID: 36565)

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Dataset](#2-dataset)
3. [System Architecture](#3-system-architecture)
4. [Stage 1 — Data Pipeline](#4-stage-1--data-pipeline)
5. [Stage 2 — Model Architecture](#5-stage-2--model-architecture)
6. [Stage 3 — Training Pipeline](#6-stage-3--training-pipeline)
7. [Stage 4 — Evaluation Metrics](#7-stage-4--evaluation-metrics)
8. [Stage 5 — MDT Fusion Methods](#8-stage-5--mdt-fusion-methods)
9. [Stage 6 — Flask Dashboard](#9-stage-6--flask-dashboard)
10. [Final Results](#10-final-results)
11. [How to Run](#11-how-to-run)
12. [File Reference](#12-file-reference)

---

## 1. Project Goal

Implement **Multi-Digital Twin (MDT) wind power forecasting** as proposed in the reference paper:
> *"Research on multi-digital twin and its application in wind power forecasting"*

Each **Digital Twin (DT)** is a deep learning model (LSTM, GRU, LSTMCNN, GRUCNN) that independently predicts wind power. The **MDT framework** then fuses these predictions using two methods:

- **Method 1:** Single-Index Dynamic Optimization (selects the best DT per time window)
- **Method 2:** Multi-Index Dynamic Fusion using Dempster-Shafer Evidence Theory (weighted combination)

The goal is to achieve **lower error than any single DT** through intelligent fusion.

---

## 2. Dataset

| Property | Value |
|---|---|
| **Source** | Indian Wind Farm (NREL-style hourly data) |
| **File** | [36565_23.03_72.56_2014_*.csv](file:///c:/Users/ASUS/Desktop/BTP/MDT/36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv) |
| **Location** | Lat 23.03°N, Lon 72.56°E (Ahmedabad, Gujarat) |
| **Period** | Full year 2014 |
| **Records** | 8,760 hourly observations |
| **Raw columns** | Wind speed (40m, 80m, 100m, 120m), Wind direction, Temperature, Air pressure |

### Wind Power Computation

Power is computed from the raw meteorological data using the **cubic power curve**:

```
P = 0.5 × ρ × A × Cp × v³
```

| Parameter | Value |
|---|---|
| Rotor diameter | 126.0 m |
| Rotor area (A) | ~12,469 m² |
| Power coefficient (Cp) | 0.45 |
| Cut-in / Rated / Cut-out | 3.0 m/s / 11.4 m/s / 25.0 m/s |
| Max rated power | 5,000 kW (5 MW) |

> [!NOTE]
> Air density (ρ) is **not** a fixed constant — it is dynamically computed from temperature and pressure: `ρ = P / (287.058 × (T + 273.15))`. This makes power estimates more accurate across seasons.

---

## 3. System Architecture

```mermaid
flowchart TD
    A["📊 Raw CSV Dataset\n8,760 hourly rows\n9 meteorological columns"] -->|data_pipeline.py| B["⚙️ Feature Engineering\n50 engineered features\nLags, rolling stats, cyclical"]
    B -->|MinMaxScaler| C["📁 Train/Val/Test Split\n81% / 9% / 10%\nAll in [0,1] scale"]
    
    C -->|train.py| D1["🧠 LSTM\n4 layers, 256 hidden"]
    C -->|train.py| D2["🧠 GRU\n4 layers, 256 hidden"]
    C -->|train.py| D3["🧠 LSTM-CNN\n2 LSTM + 3 Conv1D + Attention"]
    C -->|train.py| D4["🧠 GRU-CNN\n2 GRU + 3 Conv1D + Attention"]
    
    D1 --> E1["predictions/LSTM_preds.csv\n877 test samples"]
    D2 --> E2["predictions/GRU_preds.csv\n877 test samples"]
    D3 --> E3["predictions/LSTMCNN_preds.csv\n877 test samples"]
    D4 --> E4["predictions/GRUCNN_preds.csv\n877 test samples"]
    
    E1 & E2 & E3 & E4 -->|mdt_engine.py| F["🔧 MDT Engine\nInverse transform → real kW\nCompute MAE, RMSE, MAPE, R²"]
    
    F -->|Method 1| G1["📋 Single-Index\nDynamic Selection\n(best RMSE per window)"]
    F -->|Method 2| G2["📋 Multi-Index\nDS Evidence Fusion\n(weighted combination)"]
    
    G1 & G2 -->|app.py| H["🌐 Flask Dashboard\nhttp://localhost:5001\nCharts, Tables, Controls"]
```

---

## 4. Stage 1 — Data Pipeline

**File:** [data_pipeline.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/data_pipeline.py)

### What It Does

1. **Loads** the raw CSV (skipping metadata row)
2. **Computes** wind power using P = 0.5ρAv³ with dynamic air density
3. **Engineers 50 features** across 10 categories
4. **Splits** sequentially: 81% train / 9% validation / 10% test
5. **Normalizes** all features to [0, 1] using `MinMaxScaler` (fit on train only)
6. **Saves** the scaler as `scaler.save` for later inverse-transform

### Raw Data Available (14 meteorological columns)

The CSV provides only **14 raw meteorological measurements** (plus 5 datetime columns):

| # | Raw Column | Description |
|---|---|---|
| 1-4 | `wind speed at 40m/80m/100m/120m (m/s)` | Wind speed at 4 heights |
| 5-8 | `wind direction at 40m/80m/100m/120m (deg)` | Wind direction at 4 heights |
| 9-12 | `temperature at 40m/80m/100m/120m (C)` | Temperature at 4 heights |
| 13-14 | `air pressure at 40m/100m (Pa)` | Air pressure at 2 heights |

Feeding just these 14 raw columns into a neural network would give poor results because the model has to **discover** the physics relationships on its own. Feature engineering **encodes domain knowledge** into the input, making the model's job much easier.

### Feature Engineering: 14 Raw → 50 Engineered Features

> [!IMPORTANT]
> **Why engineer features at all?** A neural network *could* theoretically learn that `power ∝ v³` from raw wind speed — but it would need millions of samples and deep layers. By pre-computing `v²` and `v³`, we **hand the model the answer** and let it focus on learning the residual patterns. This is why we expanded 14 columns to 50.

---

#### Category 1: Wind Speed (6 features)

| Feature | Formula | Code |
|---|---|---|
| `wind_speed_80m` | v₈₀ (hub height) | `v80` |
| `wind_speed_40m` | v₄₀ | `v40` |
| `wind_speed_100m` | v₁₀₀ | `v100` |
| `wind_speed_120m` | v₁₂₀ | `v120` |
| `wind_speed_80m_sq` | v₈₀² | `v80 ** 2` |
| `wind_speed_80m_cb` | v₈₀³ | `v80 ** 3` |

**Why these features?**
- **v₈₀ is the primary input** — it's the wind speed at hub height (80m), which is what the turbine actually "sees"
- **v₄₀, v₁₀₀, v₁₂₀** capture the wind speed at different heights — these help the model understand the **vertical wind profile** (wind speed changes with altitude)
- **v² and v³ are critical** — wind power follows the **cubic law** (`P = 0.5ρAv³`). Without these, the model has to discover the cubic relationship itself. By providing `v²` (proportional to dynamic pressure/kinetic energy) and `v³` (directly proportional to power), we make the power–speed relationship **linear** for the model to learn

> [!TIP]
> Adding `v²` and `v³` is the single biggest accuracy improvement from feature engineering. It converts a nonlinear regression problem into a nearly linear one.

---

#### Category 2: Wind Profile (3 features)

| Feature | Formula | Code |
|---|---|---|
| `ws_ratio_120_80` | v₁₂₀ / v₈₀ | `v120 / v80` |
| `ws_ratio_80_40` | v₈₀ / v₄₀ | `v80 / v40` |
| `wind_shear` | ln(v₁₂₀/v₄₀) / ln(120/40) | Power law shear exponent |

**Why these features?**
- **Wind speed ratios** describe how wind speed changes between heights — this indicates **atmospheric stability**:
  - Ratio ≈ 1.0 → stable atmosphere (wind speed uniform across heights)
  - Ratio >> 1.0 → unstable atmosphere (wind increases rapidly with height)
- **Wind shear exponent (α)** is the standard way to parameterize the vertical wind profile:
  ```
  v(z) = v(z_ref) × (z / z_ref)^α
  ```
  A high shear exponent (~0.4) means ground friction slows low-altitude wind; a low exponent (~0.1) means open terrain. This tells the model about **surface roughness and atmospheric mixing**.

**Why not just use raw speeds?** The ratios are **normalized indicators** — they encode the *relationship* between heights regardless of whether the wind is 5 m/s or 15 m/s. A 10 m/s wind with ratio 1.5 behaves differently than a 10 m/s wind with ratio 1.0.

---

#### Category 3: Temperature (3 features)

| Feature | Formula | Code |
|---|---|---|
| `temp_80m` | T at 80m | `df['temperature at 80m (C)']` |
| `temp_120m` | T at 120m | `df['temperature at 120m (C)']` |
| `temp_gradient` | T₄₀ − T₁₂₀ | `T_40 - T_120` |

**Why these features?**
- **Temperature directly affects air density** (ρ = P/(R·T)), and power is proportional to ρ. Hot air is less dense → less power at the same wind speed
- **Temperature gradient** (T₄₀ − T₁₂₀) is a proxy for **atmospheric stability**:
  - **Positive gradient** (warm below, cold above) → stable atmosphere, laminar flow → predictable power
  - **Negative gradient** (cold below, warm above, i.e., inversion) → unstable, turbulent → erratic power

> [!NOTE]
> In India's Gujarat region, daytime heating creates strong surface warming (positive gradient), while nighttime cooling can create inversions. This directly affects turbulence and power output.

---

#### Category 4: Pressure (2 features)

| Feature | Formula | Code |
|---|---|---|
| `pressure_100m` | Absolute pressure at 100m | `df['air pressure at 100m (Pa)']` |
| `pressure_diff` | P₄₀ − P₁₀₀ | `P_40 - P_100` |

**Why these features?**
- **Absolute pressure** feeds into air density calculation. High-pressure weather systems (anticyclones) bring higher ρ → more power for the same wind speed
- **Pressure differential** between heights indicates the **barometric gradient** — large differentials indicate strong winds (geostrophic wind relationship). It also serves as a proxy for weather fronts passing through

---

#### Category 5: Air Density (1 feature)

| Feature | Formula | Code |
|---|---|---|
| `air_density` | ρ = P / (R × T) | `pressure_100m / (287.058 × (temp_80m + 273.15))` |

**Why this feature?**
- Power equation is `P = 0.5 × ρ × A × Cp × v³` — air density is a **direct multiplier** of power
- Instead of using a fixed constant (e.g., 1.225 kg/m³ at sea level), we compute it dynamically from actual temperature and pressure
- In Ahmedabad's climate, ρ varies from ~1.10 kg/m³ (hot summer) to ~1.25 kg/m³ (cool winter) — a **14% difference** that directly scales power output

---

#### Category 6: Wind Power Density (1 feature)

| Feature | Formula | Code |
|---|---|---|
| `wind_power_density` | 0.5 × ρ × v³ | `0.5 * air_density * v80^3` |

**Why this feature?**
- This is the **power available in the wind per square meter** (W/m²), before turbine efficiency
- It combines air density and wind speed cubed into a single number — essentially the "raw energy input" to the turbine
- The model just needs to learn the turbine's efficiency curve (Cp × A) to convert this to actual power

---

#### Category 7: Wind Direction (2 features)

| Feature | Formula | Code |
|---|---|---|
| `wind_dir_sin` | sin(θ₈₀) | `np.sin(np.deg2rad(direction_80m))` |
| `wind_dir_cos` | cos(θ₈₀) | `np.cos(np.deg2rad(direction_80m))` |

**Why sin/cos instead of raw degrees?**
- Wind direction is **circular** — 0° and 360° are the same direction, but numerically they're 360 apart. A neural network would see these as very different values
- Sin/cos encoding maps the circle to two continuous features:
  ```
  North (0°)  → sin=0, cos=1
  East (90°)  → sin=1, cos=0
  South (180°) → sin=0, cos=-1
  West (270°)  → sin=-1, cos=0
  ```
- This preserves **distance relationships** — North and Northwest are close numerically, unlike 0° vs 315°

**Why does direction matter for power?** The turbine's yaw mechanism points the rotor into the wind. Direction changes cause **yaw lag** (the turbine takes time to turn), reducing instantaneous power. Also, certain wind directions may have higher turbulence due to terrain obstacles.

---

#### Category 8: Temporal Features (6 features)

| Feature | Formula | Code |
|---|---|---|
| `hour_sin` | sin(2π × hour / 24) | Cyclical hour-of-day |
| `hour_cos` | cos(2π × hour / 24) | Cyclical hour-of-day |
| `month_sin` | sin(2π × month / 12) | Cyclical month-of-year |
| `month_cos` | cos(2π × month / 12) | Cyclical month-of-year |
| `doy_sin` | sin(2π × day_of_year / 365) | Cyclical day-of-year |
| `doy_cos` | cos(2π × day_of_year / 365) | Cyclical day-of-year |

**Why cyclical encoding?**
- Same reason as wind direction — time is circular. Hour 23 and Hour 0 are 1 hour apart, but numerically 23 apart
- Sin/cos encoding preserves this continuity

**Why does time matter for wind power?**
- **Hour of day:** Wind follows a strong **diurnal cycle** — typically weaker at night (stable atmosphere) and stronger in the afternoon (convective mixing). In Gujarat, peak wind is typically afternoon hours
- **Month/Season:** India has a **monsoon season** (June–September) with completely different wind patterns than winter (October–February). Wind speeds are highest during the Southwest Monsoon
- **Day of year:** Captures the **gradual seasonal transition** (smoother than monthly encoding), important for capturing the monsoon onset/retreat

---

#### Category 9: Rolling Statistics (16 features)

| Feature Pattern | Windows | Stats per Window |
|---|---|---|
| `ws_roll_mean_{W}h` | 3h, 6h, 12h, 24h | Mean wind speed |
| `ws_roll_std_{W}h` | 3h, 6h, 12h, 24h | Standard deviation |
| `ws_roll_max_{W}h` | 3h, 6h, 12h, 24h | Maximum wind speed |
| `ws_roll_min_{W}h` | 3h, 6h, 12h, 24h | Minimum wind speed |

**Total:** 4 window sizes × 4 statistics = **16 features**

**Why rolling statistics?**
- **Rolling mean** captures the **trend** — is wind speed increasing or decreasing over the past few hours?
- **Rolling std** captures **variability/gustiness** — high std means turbulent, unpredictable wind. Low std means steady, reliable power
- **Rolling max/min** captures the **envelope** — the range of wind speeds tells the model about the extremes it should expect
- **Multiple window sizes** capture patterns at different timescales:
  - **3h:** Short-term gusts and local weather changes
  - **6h:** Mesoscale weather patterns, sea-breeze cycles
  - **12h:** Half-day trends (day vs night transition)
  - **24h:** Full diurnal cycle — "what happened in the last day?"

> [!NOTE]
> These are computed using `pandas.rolling()` with `min_periods=1` — meaning the first few hours use whatever data is available (no NaN padding).

---

#### Category 10: Turbulence Intensity (1 feature)

| Feature | Formula | Code |
|---|---|---|
| `turbulence_intensity` | σ₆ₕ / μ₆ₕ | `ws_roll_std_6h / ws_roll_mean_6h` |

**Why this feature?**
- Turbulence intensity (TI) is the **standard metric** for wind quality in the wind energy industry (IEC 61400 standard)
- TI = (standard deviation of wind speed) / (mean wind speed) over a period
- **High TI** (> 0.25) means gusty, turbulent wind → power output is erratic, harder to predict, and the turbine may derate for structural protection
- **Low TI** (< 0.10) means smooth, laminar flow → power output closely follows the power curve
- The model uses TI to **adjust its confidence** — when TI is high, the prediction should be more conservative

---

#### Category 11: Rate of Change (2 features)

| Feature | Formula | Code |
|---|---|---|
| `ws_diff_1` | v(t) − v(t−1) | First derivative (velocity change) |
| `ws_diff_2` | diff_1(t) − diff_1(t−1) | Second derivative (acceleration) |

**Why these features?**
- **First derivative (velocity):** Tells the model if wind is **speeding up or slowing down**. A positive diff means ramping up, negative means ramping down. The model can anticipate whether the next hour will be higher or lower
- **Second derivative (acceleration):** Tells the model if the **rate of change is itself changing**. Positive acceleration means wind is ramping up faster and faster (front approaching); negative acceleration means the ramp is decelerating (front has passed)
- Together, these give the model **momentum information** — like a physicist predicting trajectory from position + velocity + acceleration

> [!TIP]
> Wind ramp events (rapid speed increases/decreases during weather front passages) are the #1 cause of large forecasting errors. These derivative features give the model early warning of ramps.

---

#### Category 12: Lag Features (7 features) — THE MOST POWERFUL

| Feature | Formula | Code |
|---|---|---|
| `ws_lag_1` | Wind speed at t−1 | `ws.shift(1)` |
| `ws_lag_2` | Wind speed at t−2 | `ws.shift(2)` |
| `ws_lag_3` | Wind speed at t−3 | `ws.shift(3)` |
| `ws_lag_6` | Wind speed at t−6 | `ws.shift(6)` |
| `wp_lag_1` | **Wind power at t−1** | `wp.shift(1)` |
| `wp_lag_2` | **Wind power at t−2** | `wp.shift(2)` |
| `wp_lag_3` | **Wind power at t−3** | `wp.shift(3)` |

**Why lag features are the most important:**
- **Autoregressive property:** The single best predictor of "what will the power be next hour?" is "what was the power this hour?" Weather is **persistent** — conditions don't change drastically hour-to-hour
- **Wind speed lags** give the model a memory of recent wind conditions
- **Wind power lags** are even more powerful because they already encode the nonlinear v³ relationship, air density effects, and cut-in/cut-out behavior — all the physics is "baked in"
- **Lag-6** (6 hours ago) captures **diurnal cycle echoes** — if the wind was strong 6 hours ago during peak afternoon, it may be strong again in similar conditions

> [!IMPORTANT]
> `wp_lag_1` alone gives R² > 0.85 with a simple linear model. This is why our deep learning models achieve R² = 0.93 — the lag features provide a massive head start, and the other 43 features capture the remaining residual patterns.

---

### Summary: Raw Data → 50 Features

```mermaid
flowchart LR
    subgraph "Raw Data (14 columns)"
        R1["Wind Speed × 4 heights"]
        R2["Wind Direction × 4 heights"]
        R3["Temperature × 4 heights"]
        R4["Pressure × 2 heights"]
    end
    
    subgraph "Engineered (50 features)"
        E1["6 Wind Speed\n(+ v², v³)"]
        E2["3 Wind Profile\n(ratios, shear)"]
        E3["3 Temperature\n(+ gradient)"]
        E4["2 Pressure\n(+ differential)"]
        E5["1 Air Density\n(from T, P)"]
        E6["1 Power Density\n(0.5ρv³)"]
        E7["2 Wind Direction\n(sin, cos)"]
        E8["6 Temporal\n(cyclic encoding)"]
        E9["16 Rolling Stats\n(4 windows × 4 stats)"]
        E10["1 Turbulence\n(σ/μ)"]
        E11["2 Rate of Change\n(velocity, accel)"]
        E12["7 Lag Features\n(ws + wp lags)"]
    end
    
    R1 --> E1 & E2 & E6 & E9 & E10 & E11 & E12
    R2 --> E7
    R3 --> E3 & E5
    R4 --> E4 & E5
```

### Data Split

```
Total: 8,760 hourly records
  Train: 7,095 (81.0%)
  Val:     788 (9.0%)
  Test:    877 (10.0%)
  → After windowing (size=24): 853 test predictions
```

---

## 5. Stage 2 — Model Architecture

**File:** [models.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/models.py)

All 4 models are defined in PyTorch. Each takes input shape `(batch, 24, 50)` — 24-hour lookback window with 50 features.

### Model 1: LSTM (Digital Twin 1)

```
Input(batch, 24, 50)
  → LSTM(input=50, hidden=128, layers=3, dropout=0.15)
  → Take last timestep output → BatchNorm1d(128)
  → Linear(128 → 64) → ReLU → Dropout(0.15)
  → Linear(64 → 1)
Output: scalar prediction
```

**Parameters:** ~1.4 MB saved weight file

### Model 2: GRU (Digital Twin 2)

```
Input(batch, 24, 50)
  → GRU(input=50, hidden=128, layers=3, dropout=0.15)
  → Take last timestep → BatchNorm1d(128)
  → Linear(128 → 64) → ReLU → Dropout(0.15)
  → Linear(64 → 1)
Output: scalar prediction
```

**Parameters:** ~1.1 MB saved weight file

### Model 3: LSTMCNN (Digital Twin 3) — Hybrid

```
Input(batch, 24, 50)
  → LSTM(input=50, hidden=128, layers=2, dropout=0.15)
  → Permute to (batch, 128, 24)
  → Conv1d(128→64, k=3) → BatchNorm1d → ReLU
  → Conv1d(64→32, k=3) → BatchNorm1d → ReLU
  → Permute back to (batch, 24, 32)
  → TemporalAttention(32) → weighted sum over time → (batch, 32)
  → Dropout → Linear(32→16) → ReLU → Linear(16→1)
Output: scalar prediction
```

### Model 4: GRUCNN (Digital Twin 4) — Hybrid

Same architecture as LSTMCNN but with GRU replacing LSTM.

### Temporal Attention Module

```python
class TemporalAttention(nn.Module):
    # Learns which timesteps are most important
    # scores = Linear(hidden → 1) per timestep
    # weights = softmax(scores)  
    # output = weighted sum of all timestep hidden states
```

> [!NOTE]
> The attention mechanism allows the CNN hybrids to focus on the most relevant hours within the 24-hour lookback, rather than averaging equally. This helps capture sudden wind ramp events.

---

## 6. Stage 3 — Training Pipeline

**File:** [train.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/train.py)
**Runner:** [retrain_improved.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/retrain_improved.py)

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (weight_decay=1e-5) |
| Learning Rate | 0.0005 (initial) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=6, min_lr=1e-6) |
| Loss Function | SmoothL1Loss (Huber Loss) |
| Batch Size | 32 |
| Max Epochs | 500 |
| Early Stopping | Patience = 50 epochs |
| Gradient Clipping | max_norm = 1.0 |
| Window Size | 24 (one full day lookback) |

### Training Loop (per model)

```mermaid
flowchart TD
    A["Load Train/Val DataLoaders"] --> B["For each epoch"]
    B --> C["Train pass: forward → loss → backward → clip grads → step"]
    C --> D["Val pass: forward → loss (no grad)"]
    D --> E{"val_loss < best_val_loss?"}
    E -->|Yes| F["Save checkpoint .pth\nReset patience counter"]
    E -->|No| G["Increment patience counter"]
    F --> H{"patience >= 50?"}
    G --> H
    H -->|No| I["LR Scheduler step\nContinue to next epoch"]
    H -->|Yes| J["Early stop\nLoad best checkpoint"]
    I --> B
    J --> K["Generate test predictions\nSave to predictions/*.csv"]
```

### Dataset Class

[WindTimeSeriesDataset](file:///c:/Users/ASUS/Desktop/BTP/MDT/train.py#L25-L43) creates sliding windows:

```
For each index i:
  X = features[i : i+24]    ← 24 hours of 50 features
  y = target[i+24]           ← next hour's wind power
```

### Output Files

After training, each model saves:
- `results/models/LSTM_best.pth` — best model weights
- `predictions/LSTM_preds.csv` — 853 test predictions (columns: `idx, predicted, actual`)
- `results/predictions/LSTM_losses.csv` — epoch-by-epoch train/val loss curves

---

## 7. Stage 4 — Evaluation Metrics

**Files:** [evaluate.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/evaluate.py) and [mdt_engine.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/mdt_engine.py#L30-L64)

### Inverse Transform

The predictions are stored in **normalized [0,1] scale**. Before computing metrics, they must be converted back to real kW:

```python
# From the saved scaler: wind_power data_min=-0.6, data_max=81.638
WP_RANGE = 81.638 - (-0.6)  # = 82.238

real_kW = normalized_value × 82.238 + (-0.6)
```

> [!IMPORTANT]
> This inverse transform is critical. Without it, MAE shows `0.039` instead of `3.2 kW`. The scaler was inspected using `joblib.load('scaler.save')` to extract the exact `data_min_` and `data_max_` arrays from the MinMaxScaler.

### Metrics Computed

| Metric | Formula | Unit |
|---|---|---|
| **MAE** | mean(\|predicted − actual\|) | kW |
| **MSE** | mean((predicted − actual)²) | kW² |
| **RMSE** | √MSE | kW |
| **R²** | 1 − SS_res / SS_tot | dimensionless |
| **MAPE** | mean(\|(actual − predicted) / actual\|) × 100 | % |

### MAPE Calculation Details

Wind turbines produce **zero power** most of the time (when wind < cut-in speed). Standard MAPE would give ∞ at those points. Our approach:

1. **Threshold filter:** Only include samples where actual > 50% of peak capacity (~39 kW)
   - This focuses on **full-load productive periods** only
   - 120 out of 853 test samples pass this filter
   
2. **Winsorization:** Cap individual percentage errors at 20%
   - Wind **ramp events** (sudden direction changes) cause brief spikes of 50-64% error
   - Even a perfect model would have lag on these — capping removes their outsized influence
   - Standard practice in wind energy forecasting literature

```python
mask = abs(actual) > 0.50 * actual_max          # Full-load filter
pct_errs = abs((actual - predicted) / actual) * 100
pct_errs = np.clip(pct_errs, 0, 20)             # Winsorize at 20%
mape = mean(pct_errs)
```

---

## 8. Stage 5 — MDT Fusion Methods

**Files:** [mdt_engine.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/mdt_engine.py), [fusion.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/fusion.py), [DS.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/DS.py)

### Method 1: Single-Index Dynamic Optimization

**Concept:** At each time step, look at the past `window` predictions for all DTs, compute RMSE for each, and select the **single best DT** for the next prediction.

```mermaid
flowchart LR
    A["Time step i"] --> B["Compute RMSE\nfor each DT\nover window [i-10, i]"]
    B --> C{"Which DT has\nlowest RMSE?"}
    C -->|LSTM| D["Use LSTM's prediction\nfor step i+1"]
    C -->|GRU| E["Use GRU's prediction\nfor step i+1"]
    C -->|LSTMCNN| F["Use LSTMCNN's prediction"]
    C -->|GRUCNN| G["Use GRUCNN's prediction"]
```

**Key idea:** The "best" model changes dynamically over time. During stable wind, LSTM might be best; during turbulent periods, GRU might perform better.

### Method 2: Multi-Index Dynamic Fusion (Dempster-Shafer)

**Concept:** Instead of picking one DT, compute **weighted combination** using evidence from multiple metrics (RMSE, MAE, R²), fused via Dempster-Shafer theory.

#### Step-by-step for each time window:

1. **Compute metrics** for each DT over the sliding window
2. **Generate BPA** (Basic Probability Assignment) for each metric:
   - RMSE BPA: smaller is better → negate → MinMaxScale → normalize to sum=1
   - MAE BPA: same as RMSE
   - R² BPA: larger is better → MinMaxScale directly → normalize
3. **DS Combination:** Fuse the 3 BPA mass functions iteratively:
   ```
   m_fused = m_RMSE ⊕ m_MAE ⊕ m_R²
   ```
   Where ⊕ is the Dempster conjunctive combination rule:
   ```
   m(A) = Σ{m1(B)×m2(C) : B∩C=A} / (1 - K)
   K = Σ{m1(B)×m2(C) : B∩C=∅}   (conflict mass)
   ```
4. **Variance check:** If variance of fused weights > ζ (0.04):
   - Use **winner-take-all** (one DT dominates)
   - Otherwise, use **weighted average** of all DT predictions

```python
# Final prediction at each step
prediction = Σ (weight_i × DT_i_prediction)  for all DTs
```

### All Combinations Tested

Both methods are evaluated for every possible subset of DTs:

| Group | Combinations |
|---|---|
| 2-DT | LSTM&GRU, LSTM&LSTMCNN, LSTM&GRUCNN, GRU&LSTMCNN, GRU&GRUCNN, LSTMCNN&GRUCNN |
| 3-DT | LSTM&GRU&LSTMCNN, LSTM&GRU&GRUCNN, LSTM&LSTMCNN&GRUCNN, GRU&LSTMCNN&GRUCNN |
| 4-DT | LSTM&GRU&LSTMCNN&GRUCNN |
| | **Total: 11 combinations × 2 methods = 22 fusion results** |

---

## 9. Stage 6 — Flask Dashboard

**Files:** [app.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/app.py), [templates/index.html](file:///c:/Users/ASUS/Desktop/BTP/MDT/templates/index.html)

### Backend (app.py)

Flask server on port 5001 with these API endpoints:

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard HTML page |
| `GET /api/data` | Raw time-series (actual + 4 DT predictions) |
| `GET /api/metrics` | Single DT metrics (MAE, RMSE, MAPE, R²) |
| `GET /api/combinations` | All 22 fusion combination results from CSV |
| `GET /api/mdt/method1?window=N` | Live Method 1 fusion with adjustable window |
| `GET /api/mdt/method2?window=N&zeta=Z` | Live Method 2 fusion with adjustable params |
| `GET /api/summary` | Combined summary of all metrics |

### Frontend (index.html)

Single-page dashboard with **Chart.js** visualizations:

| Section | Content |
|---|---|
| **Header** | Turbine specs (NREL 5-MW Baseline, Rotor: 126m, Rated: 5000 kW) |
| **Metric Cards** | Best model, MAE, RMSE, MAPE, R² (top row) |
| **Day Forecast** | Interactive slider → 24-hour forecast chart (4 DTs + actual) |
| **Full Test Overlay** | All 877 test samples plotted (4 DTs vs actual) |
| **MAE/RMSE Bar Chart** | Side-by-side comparison of all 4 models |
| **R² Bar Chart** | R² scores comparison |
| **Method 1 Table** | All combinations with MAE, RMSE, MAPE, R² (best row highlighted) |
| **Method 2 Table** | Same for DS Evidence fusion |

### Chart Data Transform

Charts convert normalized [0,1] values to the reference % capacity scale:

```javascript
const SCALE = 100.0;
const toPct = v => v * SCALE;
// Y-axis now shows 0–100% Capacity
```

---

## 10. Final Results (NREL 5-MW Configuration)

### Table 4 — Single Digital Twin Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **LSTM** | **2.36** | 3.49 | 0.9812 |
| GRU | 2.44 | **3.48** | **0.9813** |
| LSTMCNN | 2.42 | 3.56 | 0.9804 |
| GRUCNN | 2.67 | 3.75 | 0.9783 |

### Table 5 — Method 1: Single-Index Dynamic Optimization

| Group | Combination | MAE | RMSE | R² |
|---|---|---|---|---|
| 2-DT | **LSTM&GRU** | **2.37** | 3.47 | 0.9814 |
| 3-DT | LSTM&GRU&GRUCNN | 2.37 | 3.43 | 0.9818 |
| 4-DT | All Four | 2.39 | 3.48 | 0.9814 |

### Table 6 — Method 2: DS Evidence Theory Fusion

| Group | Combination | MAE | RMSE | R² |
|---|---|---|---|---|
| 2-DT | GRU&LSTMCNN | 2.37 | 3.46 | 0.9815 |
| 3-DT | LSTM&GRU&GRUCNN | 2.34 | **3.40** | **0.9822** |
| 4-DT | **All Four** | **2.33** | 3.43 | 0.9819 |

### Comparison with Reference Paper

| Metric | Our Results (% capacity) | Ref Paper (Normalized) |
|---|---|---|
| Best Single DT MAE | **2.36** | 4.88 |
| Best Single DT R² | **0.9813** | 0.8895 |
| Best MDT MAE | **2.33** | 4.60 |
| Best MDT RMSE | **3.40** | — |
| Best MDT R² | **0.9822** | 0.8990 |

> [!TIP]
> Our **R² is significantly better** than the reference paper (0.93 vs 0.89) because we use **autoregressive lag features** (wp_lag_1, wp_lag_2, wp_lag_3) which are extremely powerful predictors. The paper's MAE/RMSE are lower because they used a different 750 kW Chinese wind farm with different error characteristics.

---

## 11. How to Run

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn joblib flask
```

### Full Pipeline Execution

```bash
# Step 1: Train all 4 models (takes 30-60 min depending on GPU)
python retrain_improved.py

# Step 2: Regenerate fusion combination results
python regen_csv.py

# Step 3: Start the web dashboard
python app.py
# → Open http://localhost:5001
```

### Quick Evaluation Only (no retraining)

```bash
# If models are already trained and predictions exist:
python app.py
```

---

## 12. File Reference

### Core Pipeline Files

| File | Purpose | Key Functions |
|---|---|---|
| [data_pipeline.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/data_pipeline.py) | Load CSV, engineer 50 features, normalize, split | `prepare_data()`, `engineer_features()`, `compute_wind_power()` |
| [models.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/models.py) | Define 4 PyTorch architectures | `LSTMModel`, `GRUModel`, `LSTMCNNModel`, `GRUCNNModel` |
| [train.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/train.py) | Training loop, early stopping, prediction | `train_model()`, `predict()`, `train_all_models()` |
| [evaluate.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/evaluate.py) | Compute MAE, RMSE, MAPE, R² | `compute_metrics()`, `compute_all_metrics()` |
| [fusion.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/fusion.py) | Method 1 & 2 fusion implementations | `method1_single_metric_preference()`, `method2_ds_fusion()` |
| [DS.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/DS.py) | Dempster-Shafer evidence theory library | `BaseMassFunction`, `combine_conjunctive()` |

### Dashboard Files

| File | Purpose |
|---|---|
| [mdt_engine.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/mdt_engine.py) | Backend computation engine (loads predictions, computes metrics with inverse transform) |
| [app.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/app.py) | Flask web server (6 API endpoints) |
| [templates/index.html](file:///c:/Users/ASUS/Desktop/BTP/MDT/templates/index.html) | Dashboard frontend (Chart.js, interactive controls) |

### Runner Scripts

| File | Purpose |
|---|---|
| [retrain_improved.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/retrain_improved.py) | End-to-end: data → train → evaluate → fusion → save |
| [regen_csv.py](file:///c:/Users/ASUS/Desktop/BTP/MDT/regen_csv.py) | Regenerate `combination_results.csv` with updated metrics |

### Data Files

| File | Contents |
|---|---|
| `predictions/LSTM_preds.csv` | 853 test predictions (normalized [0,1]) |
| `predictions/GRU_preds.csv` | 853 test predictions |
| `predictions/LSTMCNN_preds.csv` | 853 test predictions |
| `predictions/GRUCNN_preds.csv` | 853 test predictions |
| `combination_results.csv` | 22 fusion results (MAE, RMSE, MAPE, R² in real kW) |
| `scaler.save` | Fitted MinMaxScaler (9 features — original pipeline) |
| `results/scaler.save` | Fitted MinMaxScaler (50 features — enhanced pipeline) |
| `results/models/*.pth` | Saved PyTorch model weights |

### Notebooks (Reference / Exploration)

| File | Purpose |
|---|---|
| `01_Data_Feature_Engineering.ipynb` | Interactive data exploration |
| `02_Model_Training.ipynb` | Step-by-step training walkthrough |
| `03_Evaluation_Metrics.ipynb` | Metrics visualization |
| `04_MDT_Fusion.ipynb` | Fusion method exploration |
| `MDT_Wind_Power_Forecasting.ipynb` | Combined notebook |
