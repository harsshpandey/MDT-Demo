"""
Data Pipeline for Multi-Digital Twin Wind Power Forecasting
============================================================
Handles:
  - CSV loading (Indian wind dataset)
  - Wind power computation using P = 0.5 * ρ * A * Cp * v³
  - Feature engineering (wind shear, temporal encoding, rolling stats, etc.)
  - MinMax normalization
  - 81% train / 9% validation / 10% test sequential split
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


# ─────────────────────── Constants ───────────────────────
ROTOR_DIAMETER = 28       # m  (Sabarmati Riverfront off-grid turbine)
ROTOR_AREA = np.pi * (ROTOR_DIAMETER / 2) ** 2   # m² (approx 615.75)
CP = 0.40                 # Power coefficient (typical ~0.35-0.45)
CUT_IN_SPEED = 3.0        # m/s
CUT_OUT_SPEED = 25.0      # m/s
RATED_POWER = 250.0       # kW (Estimated rated power for 28m rotor)


def compute_wind_power(wind_speed: np.ndarray, air_density: np.ndarray) -> np.ndarray:
    """
    Compute wind power using the simplified cubic power curve:
        P = 0.5 * ρ * A * Cp * v³
    
    Clamped between 0 and RATED_POWER.
    Below cut-in speed → 0.  Above cut-out speed → 0.
    """
    power = 0.5 * air_density * ROTOR_AREA * CP * (wind_speed ** 3)
    # Convert W to kW
    power = power / 1000.0
    # Apply cut-in / cut-out / rated limits
    power = np.where(wind_speed < CUT_IN_SPEED, 0.0, power)
    power = np.where(wind_speed > CUT_OUT_SPEED, 0.0, power)
    power = np.clip(power, 0.0, RATED_POWER)
    return power


def load_indian_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the Indian wind dataset CSV.
    Row 0 = site metadata, Row 1 = column headers.
    """
    # Read with row 1 as header (0-indexed), skip row 0 metadata
    df = pd.read_csv(csv_path, header=1)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(
        df[['Year', 'Month', 'Day', 'Hour']].astype(int).assign(Minute=df['Minute'].astype(int)),
        format='%Y%m%d%H%M'
    )
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw meteorological data.
    Enhanced version with ~40+ features for high-accuracy forecasting.
    """
    out = pd.DataFrame()
    out['datetime'] = df['datetime']
    
    # ── Primary wind speed (hub height 80m) ──
    v80 = df['wind speed at 80m (m/s)'].values
    v40 = df['wind speed at 40m (m/s)'].values
    v100 = df['wind speed at 100m (m/s)'].values
    v120 = df['wind speed at 120m (m/s)'].values
    
    out['wind_speed_80m'] = v80
    out['wind_speed_40m'] = v40
    out['wind_speed_100m'] = v100
    out['wind_speed_120m'] = v120
    
    # ── Wind speed powers (direct relationship with power) ──
    out['wind_speed_80m_sq'] = v80 ** 2
    out['wind_speed_80m_cb'] = v80 ** 3
    
    # ── Wind speed ratios (profile shape) ──
    with np.errstate(divide='ignore', invalid='ignore'):
        out['ws_ratio_120_80'] = np.where(v80 > 0.1, v120 / v80, 1.0)
        out['ws_ratio_80_40'] = np.where(v40 > 0.1, v80 / v40, 1.0)
    
    # ── Wind shear exponent ──
    with np.errstate(divide='ignore', invalid='ignore'):
        shear = np.log(np.maximum(v120, 0.01) / np.maximum(v40, 0.01)) / np.log(120 / 40)
    shear = np.nan_to_num(shear, nan=0.0, posinf=0.0, neginf=0.0)
    out['wind_shear'] = shear
    
    # ── Temperature ──
    out['temp_80m'] = df['temperature at 80m (C)'].values
    out['temp_120m'] = df['temperature at 120m (C)'].values
    out['temp_gradient'] = df['temperature at 40m (C)'].values - df['temperature at 120m (C)'].values
    
    # ── Pressure ──
    out['pressure_100m'] = df['air pressure at 100m (Pa)'].values
    out['pressure_diff'] = df['air pressure at 40m (Pa)'].values - df['air pressure at 100m (Pa)'].values
    
    # ── Air Density ──
    out['air_density'] = out['pressure_100m'] / (287.058 * (out['temp_80m'] + 273.15))
    
    # ── Wind power density (W/m²) ──
    out['wind_power_density'] = 0.5 * out['air_density'] * (v80 ** 3)
    
    # ── Wind direction (sin/cos encoding at 80m) ──
    wd80 = np.deg2rad(df['wind direction at 80m (deg)'].values)
    out['wind_dir_sin'] = np.sin(wd80)
    out['wind_dir_cos'] = np.cos(wd80)
    
    # ── Temporal features (cyclical encoding) ──
    hour = df['Hour'].astype(int).values
    month = df['Month'].astype(int).values
    day_of_year = df['datetime'].dt.dayofyear.values
    
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    out['month_sin'] = np.sin(2 * np.pi * month / 12)
    out['month_cos'] = np.cos(2 * np.pi * month / 12)
    out['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    out['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)
    
    # ── Rolling statistics on 80m wind speed ──
    ws = pd.Series(v80)
    for window in [3, 6, 12, 24]:
        out[f'ws_roll_mean_{window}h'] = ws.rolling(window, min_periods=1).mean().values
        out[f'ws_roll_std_{window}h'] = ws.rolling(window, min_periods=1).std().fillna(0).values
        out[f'ws_roll_max_{window}h'] = ws.rolling(window, min_periods=1).max().values
        out[f'ws_roll_min_{window}h'] = ws.rolling(window, min_periods=1).min().values
    
    # ── Turbulence intensity (std/mean over 6h window) ──
    with np.errstate(divide='ignore', invalid='ignore'):
        ti = out['ws_roll_std_6h'] / out['ws_roll_mean_6h'].replace(0, np.nan)
    out['turbulence_intensity'] = ti.fillna(0).values
    
    # ── Wind speed rate of change (velocity + acceleration) ──
    out['ws_diff_1'] = ws.diff().fillna(0).values
    out['ws_diff_2'] = ws.diff().diff().fillna(0).values  # acceleration
    
    # ── Lag features (extremely powerful for time series) ──
    for lag in [1, 2, 3, 6]:
        out[f'ws_lag_{lag}'] = ws.shift(lag).bfill().values
    
    # ── Target: Wind Power (kW) using 80m wind speed ──
    out['wind_power'] = compute_wind_power(v80, out['air_density'].values)
    
    # ── Wind power lag (autoregressive — strongest single predictor) ──
    wp = pd.Series(out['wind_power'].values)
    for lag in [1, 2, 3]:
        out[f'wp_lag_{lag}'] = wp.shift(lag).bfill().values
    
    return out


def prepare_data(csv_path: str, save_dir: str = None):
    """
    Full data preparation pipeline.
    Returns: (train_df, val_df, test_df, scaler, feature_cols, target_col)
    """
    # Load and engineer
    raw_df = load_indian_dataset(csv_path)
    df = engineer_features(raw_df)
    
    # Drop datetime for model input
    target_col = 'wind_power'
    feature_cols = [c for c in df.columns if c not in ['datetime', target_col]]
    
    # Sequential split: 81% train, 9% val, 10% test
    n = len(df)
    n_train = int(n * 0.81)
    n_val = int(n * 0.09)
    # n_test = n - n_train - n_val
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    print(f"Dataset size: {n}")
    print(f"Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    # Normalize using MinMaxScaler (fit on train only)
    scaler = MinMaxScaler()
    cols_to_scale = feature_cols + [target_col]
    
    scaler.fit(train_df[cols_to_scale])
    
    train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
    val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
    
    # Save scaler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        scaler_path = os.path.join(save_dir, 'scaler.save')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    
    return train_df, val_df, test_df, scaler, feature_cols, target_col


def inverse_transform_target(scaler, values, feature_cols, target_col='wind_power'):
    """
    Inverse-transform the target column only.
    """
    all_cols = feature_cols + [target_col]
    target_idx = all_cols.index(target_col)
    
    # Create dummy array with correct number of columns
    dummy = np.zeros((len(values), len(all_cols)))
    dummy[:, target_idx] = values
    
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]


if __name__ == '__main__':
    csv_path = r'36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv'
    train_df, val_df, test_df, scaler, feat_cols, tgt = prepare_data(csv_path, save_dir='results')
    print(f"\nFeature columns ({len(feat_cols)}): {feat_cols}")
    print(f"Target: {tgt}")
    print(f"\nTrain sample:\n{train_df.head()}")
