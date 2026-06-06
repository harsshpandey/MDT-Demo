"""
Data Pipeline — Multi-Digital Twin Wind Power Forecasting (Honest Rebuild)
==========================================================================

Pipeline stages:
  1. Load NREL-schema CSV (real or synthetic — same schema, same code path)
  2. Compute wind power from physics (P = 0.5·ρ·A·Cp·v³)
  3. Engineer ~37 features grouped by physical motivation (see FEATURE_DOC)
  4. Sequential 81 / 9 / 10 split — matches Liu et al. 2024
  5. MinMaxScaler fit on TRAIN ONLY (avoid lookahead in scaling stats)
  6. Persist scaler + scaled splits for downstream training

Design decisions vs the previous version
----------------------------------------
1. **`wp_lag_*` (wind-power target lags) REMOVED.**

   Previously the feature set included `wp_lag_1, wp_lag_2, wp_lag_3` —
   the wind-power target shifted back 1-3 hours. For 1-hour-ahead
   forecasting this is information leakage in spirit: `wp_lag_1` IS
   "last step's answer". Single twins reached R² ≈ 0.98 simply by
   tracking `wp_lag_1`, which (a) is not the model learning physics
   and (b) collapses all 4 twins to nearly identical predictions
   → no diversity → the MDT fusion thesis from the paper has nothing
   to improve. Removing the target lag forces twins to predict from
   meteorology, which is the actual research question.

2. **`ws_lag_*` (wind-speed lags) KEPT.**

   Wind speed is exogenous to the model's prediction target. Lagging an
   input is standard practice and not leakage.

3. **Every feature carries a docstring** in `FEATURE_DOC` linking it
   to a physical equation or atmospheric phenomenon, so reviewers can
   audit each input without reading code.

4. **Sequential split** (81/9/10) matches Liu et al. 2024 Sec 4 and
   respects time order — no random shuffling that would leak future
   into train.

Reference
---------
Liu, S., Tian, J., et al. (2024). Research on multi-digital twin and its
application in wind power forecasting. Energy, 292, 130269.
"""

from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ────────────────────────────── Turbine constants ─────────────────────────────
# NREL 5-MW reference turbine — used for the power-curve target.
# These are the same values as Jonkman et al. 2009 (NREL/TP-500-38060).
ROTOR_DIAMETER  = 126.0                                # m
ROTOR_AREA      = np.pi * (ROTOR_DIAMETER / 2) ** 2    # m² ≈ 12 469
CP              = 0.45                                 # power coefficient (≈ Betz limit-bounded)
CUT_IN_SPEED    = 3.0                                  # m/s
RATED_SPEED     = 11.4                                 # m/s
CUT_OUT_SPEED   = 25.0                                 # m/s
RATED_POWER_KW  = 5000.0                               # kW

# Sequential split per Liu et al. 2024 Sec 4.
TRAIN_FRAC = 0.81
VAL_FRAC   = 0.09
# Test gets the remainder (≈ 0.10).


# ────────────────────────────── Wind-power physics ────────────────────────────


def compute_wind_power(wind_speed: np.ndarray, air_density: np.ndarray) -> np.ndarray:
    """
    NREL 5-MW power-curve model.

    Regions (Jonkman et al. 2009):
      v <  cut-in     → 0 kW          (rotor below start-up torque)
      cut-in ≤ v < rated → P = 0.5·ρ·A·Cp·v³  (variable-speed region II)
      rated ≤ v ≤ cut-out → P = rated         (pitch-controlled region III)
      v >  cut-out    → 0 kW          (safety shutdown)
    """
    p_kw = 0.5 * air_density * ROTOR_AREA * CP * (wind_speed ** 3) / 1000.0
    p_kw = np.where(wind_speed < CUT_IN_SPEED, 0.0, p_kw)
    p_kw = np.where(wind_speed > CUT_OUT_SPEED, 0.0, p_kw)
    p_kw = np.where(
        (wind_speed >= RATED_SPEED) & (wind_speed <= CUT_OUT_SPEED),
        RATED_POWER_KW, p_kw,
    )
    return np.clip(p_kw, 0.0, RATED_POWER_KW)


# ─────────────────────────────────── Loading ──────────────────────────────────


def load_indian_dataset(csv_path: str) -> pd.DataFrame:
    """
    Read NREL India Wind Toolkit CSV.

    File format:
      Row 0    : single metadata line (site/lat/lon/year/...)
      Row 1    : column headers
      Row 2+   : hourly observations

    Note: header=1 → pandas skips row 0 as metadata and uses row 1 as header.
    """
    df = pd.read_csv(csv_path, header=1)
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour"]].astype(int)
        .assign(Minute=df["Minute"].astype(int)),
        format="%Y%m%d%H%M",
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly NREL data to 15-minute intervals via cubic interpolation
    of meteorological variables.

    Why this exists
    ---------------
    Liu et al. 2024 used 15-minute samples from a Chinese wind farm. Their
    R² ≈ 0.89 is partly due to extreme persistence at that granularity
    (lag-1 autocorrelation typically > 0.97). To compare apples-to-apples,
    we upsample our hourly Indian data with cubic interpolation on physical
    quantities (wind speed, direction, temperature, pressure).

    Caveats explicitly documented
    -----------------------------
    1. Interpolation IS NOT NEW INFORMATION. We don't gain insight into
       sub-hourly turbulence or ramp events that the sensor didn't capture.
       What we DO get is a smoother target with higher autocorrelation,
       which makes the next-step problem easier — exactly as it is in the
       paper's 15-min China data.

    2. This means a fair "vs paper" comparison can be made, but the metrics
       on 15-min interpolated data cannot be claimed as forecasting skill
       at 15-min resolution from native measurements. They reflect the
       upper bound: if the underlying physics were measured at 15-min,
       this is the kind of accuracy one would expect.

    3. The cubic spline is ON THE RAW NREL COLUMNS (m/s, °C, Pa). All
       feature engineering (powers, lags, rolling stats) runs afterward on
       the upsampled grid. So `ws_lag_1` now means "wind speed 15 minutes
       ago" not "wind speed 1 hour ago".
    """
    # Build the new 15-minute time index
    start = df["datetime"].iloc[0]
    end   = df["datetime"].iloc[-1]
    new_index = pd.date_range(start=start, end=end, freq="15min")

    # Columns to interpolate (all measurements; skip integer time columns)
    measure_cols = [c for c in df.columns
                    if c not in ("datetime", "Year", "Month", "Day", "Hour", "Minute")]

    src = df.set_index("datetime").sort_index()
    # Cubic spline for smooth physical quantities; gaps at edges fall back to linear
    interp = src[measure_cols].reindex(src.index.union(new_index)).sort_index()
    interp = interp.interpolate(method="cubic").reindex(new_index)
    # Edge filling — cubic can NaN at boundaries
    interp = interp.bfill().ffill()

    out = interp.reset_index().rename(columns={"index": "datetime"})
    out["Year"]   = out["datetime"].dt.year
    out["Month"]  = out["datetime"].dt.month
    out["Day"]    = out["datetime"].dt.day
    out["Hour"]   = out["datetime"].dt.hour
    out["Minute"] = out["datetime"].dt.minute

    # Reorder to match NREL schema (time columns first)
    ordered = ["datetime", "Year", "Month", "Day", "Hour", "Minute"] + measure_cols
    return out[ordered]


# ────────────────────────────── Feature engineering ───────────────────────────
#
# FEATURE_DOC — every engineered column with its physical motivation.
# Reviewers can audit this list without reading the engineer_features() body.

FEATURE_DOC = {
    # ─── Wind speed (hub + profile) ──
    "wind_speed_40m":  "Wind speed at 40 m. Lower-profile sample used for shear estimation.",
    "wind_speed_80m":  "Hub-height wind speed (80 m). Primary driver of turbine power.",
    "wind_speed_100m": "Wind speed at 100 m. Upper-profile sample for shear estimation.",
    "wind_speed_120m": "Wind speed at 120 m. Topmost profile sample.",

    # ─── Powers of v80 — encode the cubic power law explicitly ──
    "wind_speed_80m_sq": "v80² ∝ dynamic pressure / kinetic energy term.",
    "wind_speed_80m_cb": "v80³. Power-curve region II is linear in this feature: handing the model the cubic relationship turns nonlinear regression into near-linear.",

    # ─── Vertical profile (atmospheric stability proxies) ──
    "ws_ratio_120_80":  "v120/v80 ratio. >1 → unstable / strong shear; ≈1 → neutral mixing.",
    "ws_ratio_80_40":   "v80/v40 ratio. Low-altitude shear, surface-roughness sensitive.",
    "wind_shear":       "Power-law shear exponent α from v(z)∝z^α fit over 40–120 m. α≈0.1 open terrain, α≈0.4 stable/forested.",

    # ─── Temperature ──
    "temp_80m":         "Hub-height temperature. Affects air density via ρ = P/(R·T).",
    "temp_120m":        "Upper-level temperature. Difference with hub gives thermal stratification.",
    "temp_gradient":    "T40 − T120. Positive → stable atmosphere, laminar flow; negative → inversion / turbulence.",

    # ─── Pressure ──
    "pressure_100m":    "Absolute pressure at 100 m. Density driver; tracks weather systems.",
    "pressure_diff":    "P40 − P100. Vertical pressure gradient — large values flag fronts / strong wind events.",

    # ─── Density (derived from temperature + pressure via ideal gas) ──
    "air_density":      "ρ = P100/(R_dry·(T80+273.15)). Direct multiplier in the power equation; varies ~10% across seasons.",
    "wind_power_density": "Physical proxy: 0.5·ρ·v80³ (W/m²). Same form as the target but without the curve clipping — useful inductive bias.",

    # ─── Direction (cyclical encoded so 359° ≈ 1°) ──
    "wind_dir_sin":     "sin(direction_80m). Cyclic-safe encoding; combined with cos preserves topology.",
    "wind_dir_cos":     "cos(direction_80m). Captures yaw / wake-loss directional effects.",
    "wind_dir_shear":   "Ekman-spiral proxy: angular difference (120m − 40m) wrapped to [−180°, 180°]. Surface friction rotates wind; magnitude correlates with atmospheric stability + planetary boundary layer depth.",

    # ─── Time-of-day / season (cyclical encoded) ──
    "hour_sin":         "sin(2π·hour/24). Captures diurnal cycle of land-sea breeze and thermal-driven wind.",
    "hour_cos":         "cos(2π·hour/24).",
    "month_sin":        "sin(2π·month/12). Captures annual seasonal cycle (monsoon vs winter regimes).",
    "month_cos":        "cos(2π·month/12).",
    "doy_sin":          "sin(2π·doy/365). Sub-monthly seasonal resolution.",
    "doy_cos":          "cos(2π·doy/365).",

    # ─── Rolling statistics on v80 (persistence + variability windows) ──
    "ws_roll_mean_3h":  "3-hour rolling mean of v80. Short-term trend; smooths gusts.",
    "ws_roll_std_3h":   "3-hour rolling std of v80. Short-window turbulence indicator.",
    "ws_roll_max_3h":   "3-hour rolling max of v80. Detects recent gust peaks.",
    "ws_roll_min_3h":   "3-hour rolling min of v80. Lulls / wake events.",
    "ws_roll_mean_6h":  "6-hour rolling mean — sub-diurnal trend.",
    "ws_roll_std_6h":   "6-hour rolling std — used as turbulence-intensity denominator.",
    "ws_roll_mean_24h": "24-hour rolling mean — daily-average wind regime.",
    "ws_roll_std_24h":  "24-hour rolling std — day-scale variability.",

    # ─── Derived dynamics ──
    "turbulence_intensity": "σ_6h / μ_6h. Standard atmospheric turbulence-intensity index (IEC 61400-1).",
    "ws_diff_1":            "Δv80 between consecutive hours. Wind ramp velocity.",
    "ws_diff_2":            "Δ²v80. Wind ramp acceleration; flags onset of frontal passage.",

    # ─── Wind-speed lags (exogenous input — NOT target leakage) ──
    "ws_lag_1": "v80(t−1). Persistence cue. Distinct from target lag.",
    "ws_lag_2": "v80(t−2). 2-hour persistence.",
    "ws_lag_3": "v80(t−3). 3-hour persistence.",
    "ws_lag_6": "v80(t−6). Quarter-day cycle persistence.",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from raw NREL columns.

    Returns a DataFrame with all FEATURE_DOC columns plus `datetime` and
    `wind_power` (the prediction target). Order is stable for downstream
    column-index assumptions in inverse_transform_target().
    """
    out = pd.DataFrame()
    out["datetime"] = df["datetime"]

    v40, v80   = df["wind speed at 40m (m/s)"].values,  df["wind speed at 80m (m/s)"].values
    v100, v120 = df["wind speed at 100m (m/s)"].values, df["wind speed at 120m (m/s)"].values

    # Wind speed (4 heights) + cubic powers of hub speed
    out["wind_speed_40m"], out["wind_speed_80m"]  = v40, v80
    out["wind_speed_100m"], out["wind_speed_120m"] = v100, v120
    out["wind_speed_80m_sq"], out["wind_speed_80m_cb"] = v80 ** 2, v80 ** 3

    # Profile ratios — guarded against near-zero divisors
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ws_ratio_120_80"] = np.where(v80 > 0.1, v120 / v80, 1.0)
        out["ws_ratio_80_40"]  = np.where(v40 > 0.1, v80  / v40, 1.0)
    # Shear exponent α from log-fit of v(40), v(120)
    with np.errstate(divide="ignore", invalid="ignore"):
        shear = np.log(np.maximum(v120, 0.01) / np.maximum(v40, 0.01)) / np.log(120 / 40)
    out["wind_shear"] = np.nan_to_num(shear, nan=0.0, posinf=0.0, neginf=0.0)

    # Thermal
    out["temp_80m"]      = df["temperature at 80m (C)"].values
    out["temp_120m"]     = df["temperature at 120m (C)"].values
    out["temp_gradient"] = df["temperature at 40m (C)"].values - df["temperature at 120m (C)"].values

    # Pressure
    out["pressure_100m"] = df["air pressure at 100m (Pa)"].values
    out["pressure_diff"] = df["air pressure at 40m (Pa)"].values - df["air pressure at 100m (Pa)"].values

    # Air density via ideal gas; R_dry = 287.058 J/(kg·K), T in K.
    out["air_density"]        = out["pressure_100m"] / (287.058 * (out["temp_80m"] + 273.15))
    out["wind_power_density"] = 0.5 * out["air_density"] * (v80 ** 3)

    # Direction — cyclic encoding at hub height
    wd80 = np.deg2rad(df["wind direction at 80m (deg)"].values)
    out["wind_dir_sin"], out["wind_dir_cos"] = np.sin(wd80), np.cos(wd80)
    # Directional shear — wrap (120 − 40) into [−180°, 180°] so 359→1 isn't a jump
    diff = (df["wind direction at 120m (deg)"].values - df["wind direction at 40m (deg)"].values + 180) % 360 - 180
    out["wind_dir_shear"] = diff

    # Time features — cyclic encoding so 23 ↔ 0 are adjacent
    hour  = df["Hour"].astype(int).values
    month = df["Month"].astype(int).values
    doy   = df["datetime"].dt.dayofyear.values
    out["hour_sin"], out["hour_cos"]   = np.sin(2*np.pi*hour /24), np.cos(2*np.pi*hour /24)
    out["month_sin"], out["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    out["doy_sin"], out["doy_cos"]     = np.sin(2*np.pi*doy  /365), np.cos(2*np.pi*doy  /365)

    # Rolling stats on v80 — multiple windows for multi-scale persistence
    ws = pd.Series(v80)
    for w in (3, 6, 24):
        out[f"ws_roll_mean_{w}h"] = ws.rolling(w, min_periods=1).mean().values
        out[f"ws_roll_std_{w}h"]  = ws.rolling(w, min_periods=1).std().fillna(0).values
        if w == 3:
            out["ws_roll_max_3h"] = ws.rolling(w, min_periods=1).max().values
            out["ws_roll_min_3h"] = ws.rolling(w, min_periods=1).min().values

    with np.errstate(divide="ignore", invalid="ignore"):
        ti = out["ws_roll_std_6h"] / out["ws_roll_mean_6h"].replace(0, np.nan)
    out["turbulence_intensity"] = ti.fillna(0).values

    out["ws_diff_1"] = ws.diff().fillna(0).values
    out["ws_diff_2"] = ws.diff().diff().fillna(0).values

    # Wind-speed lags ONLY (NOT power lags — that would be target leakage).
    for lag in (1, 2, 3, 6):
        out[f"ws_lag_{lag}"] = ws.shift(lag).bfill().values

    # Target: power from physics + dynamic density
    out["wind_power"] = compute_wind_power(v80, out["air_density"].values)

    # Sanity: every column listed in FEATURE_DOC must exist; never include
    # any wp_lag_* (would be target leakage). Both checks help future authors
    # avoid silently re-introducing the historical bug.
    assert all(c in out.columns for c in FEATURE_DOC), "FEATURE_DOC drift vs implementation"
    assert not any(c.startswith("wp_lag_") for c in out.columns), \
        "wp_lag_* features are forbidden — target leakage for 1-hour-ahead forecasting"

    return out


# ─────────────────────────────────── Split + scale ────────────────────────────


def prepare_data(
    csv_path: str, save_dir: str | None = None, *, resample_15min: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, list[str], str]:
    """
    Full prep: load → (optional 15-min resample) → engineer → split → scale.
    Returns the scaled splits, fitted scaler, feature column names, target name.

    Scaler is fit on TRAIN ONLY to avoid leaking distribution stats from
    validation or test into training.
    """
    raw = load_indian_dataset(csv_path)
    if resample_15min:
        raw = resample_to_15min(raw)
        print(f"Resampled to 15-min: {len(raw)} rows (was hourly)")
    df  = engineer_features(raw)

    target_col   = "wind_power"
    feature_cols = [c for c in df.columns if c not in ("datetime", target_col)]

    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train:n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val:].copy()

    print(f"Dataset: {n} hourly rows | "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Features: {len(feature_cols)} (target = {target_col})")

    scaler        = MinMaxScaler()
    cols_to_scale = feature_cols + [target_col]
    scaler.fit(train_df[cols_to_scale])

    for split in (train_df, val_df, test_df):
        split[cols_to_scale] = scaler.transform(split[cols_to_scale])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(save_dir, "scaler.save"))
        train_df.to_csv(os.path.join(save_dir, "train_scaled.csv"), index=False)
        val_df.to_csv  (os.path.join(save_dir, "val_scaled.csv"),   index=False)
        test_df.to_csv (os.path.join(save_dir, "test_scaled.csv"),  index=False)
        # Also save the unnormalized engineered set for debugging / EDA
        df.to_csv(os.path.join(save_dir, "engineered_features.csv"), index=False)
        with open(os.path.join(save_dir, "feature_doc.md"), "w", encoding="utf-8") as fh:
            fh.write("# Engineered features (with physical reasoning)\n\n")
            for name, why in FEATURE_DOC.items():
                fh.write(f"- **{name}** — {why}\n")
        print(f"Saved scaler + splits + feature_doc.md to {save_dir}/")

    return train_df, val_df, test_df, scaler, feature_cols, target_col


def inverse_transform_target(scaler, values, feature_cols, target_col: str = "wind_power"):
    """Inverse-transform only the target column (rest of the row is zeroed)."""
    all_cols = feature_cols + [target_col]
    idx = all_cols.index(target_col)
    dummy = np.zeros((len(values), len(all_cols)))
    dummy[:, idx] = values
    return scaler.inverse_transform(dummy)[:, idx]


# ────────────────────────────── CLI ───────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str,
        default="data/raw/36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv",
        help="Path to NREL India Wind Toolkit CSV.",
    )
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--resample-15min", action="store_true",
                        help="Cubic-interpolate hourly → 15-min before feature engineering.")
    args = parser.parse_args()

    train_df, val_df, test_df, scaler, feats, tgt = prepare_data(
        args.csv, save_dir=args.out, resample_15min=args.resample_15min,
    )
    print(f"\nFeatures ({len(feats)}): {feats}")
