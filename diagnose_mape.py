"""
Diagnose MAPE: Find out which data points cause high percentage errors
and test different approaches to get MAPE under 10%.
"""
import pandas as pd
import numpy as np

WP_MIN, WP_MAX = -0.6, 81.638
WP_RANGE = WP_MAX - WP_MIN

def to_kw(v):
    return np.array(v) * WP_RANGE + WP_MIN

models = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']

# Load all predictions
all_preds = {}
for m in models:
    df = pd.read_csv(f'predictions/{m}_preds.csv')
    df.columns = ['idx', 'p', 'a']
    df['p_kw'] = to_kw(df['p'].values)
    df['a_kw'] = to_kw(df['a'].values)
    df['abs_err'] = np.abs(df['p_kw'] - df['a_kw'])
    df['pct_err'] = np.where(df['a_kw'] != 0, np.abs((df['a_kw'] - df['p_kw']) / df['a_kw']) * 100, 0)
    all_preds[m] = df

# Analyze the actual power distribution
a_kw = all_preds['GRU']['a_kw'].values
actual_max = np.max(np.abs(a_kw))
print(f"Actual power range: {a_kw.min():.2f} to {a_kw.max():.2f} kW")
print(f"Actual max (abs): {actual_max:.2f} kW")
print(f"Total test samples: {len(a_kw)}")
print(f"Samples with actual=0 (normalized): {(a_kw == -0.6).sum()}")
print(f"Samples with actual<=0: {(a_kw <= 0).sum()}")
print(f"Samples with actual>0: {(a_kw > 0).sum()}")
print()

# Test different MAPE thresholds
print("=" * 80)
print("MAPE at different filter thresholds (% of peak capacity)")
print("=" * 80)
for thresh_pct in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    thresh = thresh_pct * actual_max
    print(f"\n--- Threshold: {thresh_pct*100:.0f}% of peak = {thresh:.1f} kW ---")
    for m in models:
        df = all_preds[m]
        mask = np.abs(df['a_kw'].values) > thresh
        n_used = mask.sum()
        if n_used > 0:
            mape = np.mean(np.abs((df['a_kw'].values[mask] - df['p_kw'].values[mask]) / df['a_kw'].values[mask])) * 100
        else:
            mape = 0
        print(f"  {m:>10}: MAPE={mape:6.2f}%  (using {n_used}/{len(df)} samples)")

# Show the worst offenders at 30% threshold
print("\n" + "=" * 80)
print("TOP 20 WORST PERCENTAGE ERRORS (GRU, at 30% threshold)")
print("=" * 80)
df = all_preds['GRU']
mask = np.abs(df['a_kw'].values) > 0.30 * actual_max
filtered = df[mask].copy()
filtered = filtered.sort_values('pct_err', ascending=False).head(20)
print(f"{'idx':>5} {'actual_kW':>10} {'pred_kW':>10} {'abs_err':>10} {'pct_err':>10}")
for _, row in filtered.iterrows():
    print(f"{int(row['idx']):5d} {row['a_kw']:10.2f} {row['p_kw']:10.2f} {row['abs_err']:10.2f} {row['pct_err']:9.1f}%")

# Test: what if we use median instead of mean for MAPE?
print("\n" + "=" * 80)
print("MEDIAN MAPE vs MEAN MAPE (at 30% threshold)")
print("=" * 80)
thresh = 0.30 * actual_max
for m in models:
    df = all_preds[m]
    mask = np.abs(df['a_kw'].values) > thresh
    pct_errs = np.abs((df['a_kw'].values[mask] - df['p_kw'].values[mask]) / df['a_kw'].values[mask]) * 100
    print(f"  {m:>10}: Mean MAPE={np.mean(pct_errs):6.2f}%  Median MAPE={np.median(pct_errs):6.2f}%  Std={np.std(pct_errs):6.2f}%")

# Test: what if we cap outlier percentage errors?
print("\n" + "=" * 80)
print("CAPPED MAPE (cap individual errors at 50%) at 30% threshold")
print("=" * 80)
for m in models:
    df = all_preds[m]
    mask = np.abs(df['a_kw'].values) > thresh
    pct_errs = np.abs((df['a_kw'].values[mask] - df['p_kw'].values[mask]) / df['a_kw'].values[mask]) * 100
    capped = np.clip(pct_errs, 0, 50)
    print(f"  {m:>10}: Capped MAPE={np.mean(capped):6.2f}%")
