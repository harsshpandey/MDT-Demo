"""
Compute proper inverse-transformed metrics for the dashboard.
Uses the actual scaler's wind_power min/max to convert from [0,1] back to kW.
"""
import pandas as pd
import numpy as np

# From the scaler: wind_power data_min = -0.6, data_max = 81.638
# inverse_transform: real_kW = normalized * (max - min) + min
WP_MIN = -0.6
WP_MAX = 81.638
WP_RANGE = WP_MAX - WP_MIN  # 82.238

models = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']

def inverse_transform(values):
    """Convert from [0,1] normalized back to real kW."""
    return np.array(values) * WP_RANGE + WP_MIN

print("=" * 60)
print("SINGLE DT METRICS (Table 4 equivalent)")
print("=" * 60)
print(f"{'Model':>10} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE':>8}")

for m in models:
    df = pd.read_csv(f'predictions/{m}_preds.csv')
    df.columns = ['idx', 'p', 'a']
    
    # Inverse transform to real kW
    p_kw = inverse_transform(df['p'].values)
    a_kw = inverse_transform(df['a'].values)
    
    mae = np.mean(np.abs(p_kw - a_kw))
    rmse = np.sqrt(np.mean((p_kw - a_kw) ** 2))
    ss_res = np.sum((a_kw - p_kw) ** 2)
    ss_tot = np.sum((a_kw - np.mean(a_kw)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # MAPE (filtered > 30% of max actual)
    actual_max = np.max(np.abs(a_kw))
    mask = np.abs(a_kw) > 0.30 * actual_max
    mape = np.mean(np.abs((a_kw[mask] - p_kw[mask]) / a_kw[mask])) * 100 if mask.sum() > 0 else 0
    
    print(f"{m:>10} {mae:8.4f} {rmse:8.4f} {r2:8.4f} {mape:7.2f}%")

print("\n=== Reference Paper Table 4 ===")
print(f"{'LSTM':>10} {'2.6466':>8} {'4.6799':>8} {'0.8839':>8}")
print(f"{'GRU':>10} {'2.6590':>8} {'4.5664':>8} {'0.8895':>8}")
print(f"{'LSTMCNN':>10} {'2.6263':>8} {'4.5088':>8} {'0.8923':>8}")
print(f"{'GRUCNN':>10} {'2.5423':>8} {'4.5663':>8} {'0.8895':>8}")
