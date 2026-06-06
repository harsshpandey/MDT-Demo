"""
═══════════════════════════════════════════════════════════════
  MDT Wind Power Forecasting — Full Evaluation & Visualization
═══════════════════════════════════════════════════════════════
  Ahmedabad Off-Grid Turbine (28 m rotor, 250 kW)

  This script:
    1. Loads the raw dataset and re-creates feature-engineered data
    2. Loads all 4 trained model predictions
    3. Re-computes Single DT metrics and saves to CSV
    4. Re-runs MDT fusion (Method 1 & Method 2) and saves to CSV
    5. Generates and saves ALL plots to results/plots/

  Run:  python run_full_evaluation.py
═══════════════════════════════════════════════════════════════
"""

import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from itertools import combinations

# ── project modules ──
from data_pipeline import (
    load_indian_dataset, engineer_features, prepare_data,
    compute_wind_power,
    ROTOR_DIAMETER, ROTOR_AREA, CP, CUT_IN_SPEED, CUT_OUT_SPEED, RATED_POWER,
)
from evaluate import compute_metrics, metrics_to_table
from fusion import (
    method1_single_metric_preference,
    method2_ds_fusion,
)

# ─────────────────────────── paths ───────────────────────────
CSV_PATH   = r'36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv'
RESULTS    = 'results'
PRED_DIR   = os.path.join(RESULTS, 'predictions')
PLOT_DIR   = os.path.join(RESULTS, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_NAMES = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']
COLORS      = {'LSTM': '#FF6B6B', 'GRU': '#4ECDC4',
               'LSTMCNN': '#45B7D1', 'GRUCNN': '#96CEB4'}

WINDOW_SIZE = 10

# ════════════════════════════════════════════════════════════
#  PART 1 — Raw Data & Feature Engineering
# ════════════════════════════════════════════════════════════
print("=" * 65)
print("  PART 1: Data Loading & Feature Engineering")
print("=" * 65)

raw_df  = load_indian_dataset(CSV_PATH)
feat_df = engineer_features(raw_df)

print(f"  Raw shape      : {raw_df.shape}")
print(f"  Engineered shape: {feat_df.shape}")
print(f"  Date range     : {feat_df['datetime'].min()} → {feat_df['datetime'].max()}")
print(f"  Wind power stats (kW):")
print(feat_df['wind_power'].describe().to_string())

# ── PLOT 1: Power Curve ──
speeds = np.linspace(0, 30, 300)
power  = compute_wind_power(speeds, np.full_like(speeds, 1.225))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(speeds, power, 'b-', linewidth=2.5)
ax.axvline(x=CUT_IN_SPEED, color='g', ls='--', alpha=.7,
           label=f'Cut-in ({CUT_IN_SPEED} m/s)')
ax.axvline(x=CUT_OUT_SPEED, color='r', ls='--', alpha=.7,
           label=f'Cut-out ({CUT_OUT_SPEED} m/s)')
ax.axhline(y=RATED_POWER, color='orange', ls='--', alpha=.7,
           label=f'Rated ({RATED_POWER:.0f} kW)')
ax.fill_between(speeds, power, alpha=.15, color='blue')
ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
ax.set_ylabel('Power (kW)', fontsize=12)
ax.set_title(f'Turbine Power Curve  (D={ROTOR_DIAMETER}m, '
             f'A={ROTOR_AREA:.1f}m², Cp={CP})', fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'power_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✅ Saved: power_curve.png")

# ── PLOT 2: Time Series (Wind Speed, Wind Power, Temperature) ──
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

axes[0].plot(feat_df['datetime'], feat_df['wind_speed_120m'],
             color='#45B7D1', lw=.5, alpha=.7)
axes[0].set_ylabel('Wind Speed (m/s)', fontsize=12)
axes[0].set_title('Wind Speed at 120 m — Full Year 2014', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=.3)

axes[1].plot(feat_df['datetime'], feat_df['wind_power'],
             color='#E74C3C', lw=.5, alpha=.7)
axes[1].set_ylabel('Wind Power (Normalized)', fontsize=12)
axes[1].set_title(f'Computed Wind Power  (28 m, {RATED_POWER:.0f} kW rated)',
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=.3)

axes[2].plot(feat_df['datetime'], feat_df['temp_120m'],
             color='#FF6B6B', lw=.5, alpha=.7)
axes[2].set_ylabel('Temperature (°C)', fontsize=12)
axes[2].set_title('Temperature at 120 m', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: timeseries.png")

# ── PLOT 3: Correlation Heatmap ──
fig, ax = plt.subplots(figsize=(12, 10))
corr_all = feat_df.drop(columns=['datetime']).corr()
# Get top 18 features most correlated (absolute value) with wind_power (excluding wind_power itself)
top_corr_feats = corr_all['wind_power'].abs().sort_values(ascending=False).index[1:19].tolist()
selected_cols = ['wind_power'] + top_corr_feats
corr_sub = feat_df[selected_cols].corr()

mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=.5, annot_kws={'size': 9, 'weight': 'bold'})
ax.set_title('Feature Correlation Matrix (Top 18 Features vs Wind Power)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: correlation.png")

# ── PLOT 4: Distributions ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(feat_df['wind_speed_120m'], bins=50, color='#45B7D1', alpha=.7, edgecolor='white')
axes[0].set_xlabel('Wind Speed (m/s)')
axes[0].set_title('Wind Speed Distribution (120 m)', fontweight='bold')
axes[0].grid(True, alpha=.3)

axes[1].hist(feat_df['wind_power'], bins=50, color='#E74C3C', alpha=.7, edgecolor='white')
axes[1].set_xlabel('Wind Power (Normalized)')
axes[1].set_title('Wind Power Distribution', fontweight='bold')
axes[1].grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: distributions.png")

# ════════════════════════════════════════════════════════════
#  PART 2 — Load Predictions & Compute Single-DT Metrics
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PART 2: Single Digital Twin Evaluation")
print("=" * 65)

predictions = {}
for name in MODEL_NAMES:
    path = os.path.join(PRED_DIR, f'{name}_predictions.csv')
    df = pd.read_csv(path)
    predictions[name] = {
        'test_pred':   df['prediction'].values,
        'test_actual': df['actual'].values,
    }
    print(f"  Loaded {name}: {len(df)} test samples")

# Compute metrics
single_metrics = {}
for name in MODEL_NAMES:
    m = compute_metrics(predictions[name]['test_actual'],
                        predictions[name]['test_pred'],
                        print_results=True, model_name=name)
    single_metrics[name] = m

metrics_table = metrics_to_table(single_metrics)
metrics_table.to_csv(os.path.join(RESULTS, 'single_dt_metrics.csv'))
print(f"\n  ✅ Saved: single_dt_metrics.csv")
print(metrics_table.to_string())

# ── PLOT 5: Single-DT Metrics Comparison ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Single DT Metrics Comparison', fontsize=16, fontweight='bold')
bar_colors = [COLORS[n] for n in MODEL_NAMES]

for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
    vals = [single_metrics[n][metric] for n in MODEL_NAMES]
    bars = ax.bar(MODEL_NAMES, vals, color=bar_colors, alpha=.85,
                  edgecolor='white', linewidth=1.5)
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=.3, axis='y')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'single_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: single_metrics.png")

# ── PLOT 6: Training Loss Curves ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training & Validation Loss Curves', fontsize=16, fontweight='bold')

for ax, name in zip(axes.flatten(), MODEL_NAMES):
    lpath = os.path.join(PRED_DIR, f'{name}_losses.csv')
    if os.path.exists(lpath):
        ldf = pd.read_csv(lpath)
        ax.plot(ldf['train_loss'], label='Train', color=COLORS[name], lw=2, alpha=.8)
        ax.plot(ldf['val_loss'],   label='Validation', color='orange', lw=2, alpha=.8)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(); ax.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: training_loss.png")

# ── PLOT 7: 1-Day Forecast (3 days) ──
for day_idx, start in enumerate([0, 48, 96]):
    fig, ax = plt.subplots(figsize=(16, 5))
    actual_kw = predictions['LSTM']['test_actual'][start:start + 24]
    hours = np.arange(len(actual_kw))

    ax.plot(hours, actual_kw, 'ko-', lw=2.5, ms=6, label='Actual', zorder=5)
    for name in MODEL_NAMES:
        pred_kw = predictions[name]['test_pred'][start:start + 24]
        ax.plot(hours, pred_kw, '--', color=COLORS[name], lw=1.8, alpha=.85, label=name)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Wind Power (Normalized)', fontsize=12)
    ax.set_title(f'Day {day_idx + 1} — Wind Power Forecast Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=.3); ax.set_xticks(hours)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'day_{day_idx + 1}_forecast.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: day_{day_idx + 1}_forecast.png")

# ── PLOT 8: Scatter (Actual vs Predicted) ──
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Actual vs Predicted (Scatter)', fontsize=16, fontweight='bold')
colors_list = [COLORS[n] for n in MODEL_NAMES]

for ax, name, c in zip(axes.flatten(), MODEL_NAMES, colors_list):
    actual = predictions[name]['test_actual']
    pred   = predictions[name]['test_pred']
    ax.scatter(actual, pred, alpha=.3, s=10, color=c)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, alpha=.7, label='Perfect')
    ax.set_xlabel('Actual (Normalized)'); ax.set_ylabel('Predicted (Normalized)')
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: scatter.png")

# ── PLOT 9: Error Distribution ──
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')

for ax, name, c in zip(axes.flatten(), MODEL_NAMES, colors_list):
    actual = predictions[name]['test_actual']
    pred   = predictions[name]['test_pred']
    errors = actual - pred
    ax.hist(errors, bins=50, color=c, alpha=.7, edgecolor='white')
    ax.axvline(x=0, color='red', ls='--', alpha=.7)
    ax.set_xlabel('Error (Normalized)'); ax.set_ylabel('Frequency')
    ax.set_title(f'{name} (mean={errors.mean():.2f}, std={errors.std():.2f})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'error_dist.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: error_dist.png")

# ════════════════════════════════════════════════════════════
#  PART 3 — MDT Fusion Methods
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PART 3: Multi-Digital Twin Fusion")
print("=" * 65)

actuals_arr    = predictions['LSTM']['test_actual']
preds_dict_all = {n: predictions[n]['test_pred'] for n in MODEL_NAMES}

# ── Method 1 ──
m1_results = {}
for r in [2, 3, 4]:
    for combo in combinations(MODEL_NAMES, r):
        combo_name  = '&'.join(combo)
        combo_preds = {n: preds_dict_all[n] for n in combo}
        m1_pred, m1_actual, _ = method1_single_metric_preference(
            combo_preds, actuals_arr, WINDOW_SIZE)
        m1_met = compute_metrics(m1_actual, m1_pred, print_results=False)
        m1_results[combo_name] = m1_met
        print(f"  M1  {combo_name:40s}  MAE={m1_met['MAE']:8.4f}  "
              f"RMSE={m1_met['RMSE']:8.4f}  R²={m1_met['R2']:7.4f}")

m1_df = pd.DataFrame(m1_results).T
m1_df.index.name = 'Combination'
m1_df.to_csv(os.path.join(RESULTS, 'method1_fusion_metrics.csv'))
print(f"\n  ✅ Saved: method1_fusion_metrics.csv")

# ── Method 2 ──
m2_results = {}
for r in [2, 3, 4]:
    for combo in combinations(MODEL_NAMES, r):
        combo_name  = '&'.join(combo)
        combo_preds = {n: preds_dict_all[n] for n in combo}
        m2_pred, m2_actual = method2_ds_fusion(
            combo_preds, actuals_arr, WINDOW_SIZE, zeta=0.03)
        m2_met = compute_metrics(m2_actual, m2_pred, print_results=False)
        m2_results[combo_name] = m2_met
        print(f"  M2  {combo_name:40s}  MAE={m2_met['MAE']:8.4f}  "
              f"RMSE={m2_met['RMSE']:8.4f}  R²={m2_met['R2']:7.4f}")

m2_df = pd.DataFrame(m2_results).T
m2_df.index.name = 'Combination'
m2_df.to_csv(os.path.join(RESULTS, 'method2_fusion_metrics.csv'))
print(f"\n  ✅ Saved: method2_fusion_metrics.csv")

# ── Best results ──
best_single = min(single_metrics, key=lambda k: single_metrics[k]['MAE'])
best_m1     = min(m1_results,     key=lambda k: m1_results[k]['MAE'])
best_m2     = min(m2_results,     key=lambda k: m2_results[k]['MAE'])

print(f"\n{'=' * 65}")
print(f"  BEST RESULTS SUMMARY")
print(f"{'=' * 65}")
print(f"  Best Single DT : {best_single}")
print(f"    MAE={single_metrics[best_single]['MAE']:.4f}  "
      f"RMSE={single_metrics[best_single]['RMSE']:.4f}  "
      f"R²={single_metrics[best_single]['R2']:.4f}")
print(f"  Best Method 1  : {best_m1}")
print(f"    MAE={m1_results[best_m1]['MAE']:.4f}  "
      f"RMSE={m1_results[best_m1]['RMSE']:.4f}  "
      f"R²={m1_results[best_m1]['R2']:.4f}")
print(f"  Best Method 2  : {best_m2}")
print(f"    MAE={m2_results[best_m2]['MAE']:.4f}  "
      f"RMSE={m2_results[best_m2]['RMSE']:.4f}  "
      f"R²={m2_results[best_m2]['R2']:.4f}")

# ── PLOT 10: Fusion Comparison Bar Chart ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Single DT vs MDT Fusion — Best Results', fontsize=16, fontweight='bold')
labels     = [f'Single\n({best_single})',
              f'Method 1\n({best_m1})',
              f'Method 2\n({best_m2})']
bar_colors = ['#96CEB4', '#E74C3C', '#3498DB']

for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
    vals = [single_metrics[best_single][metric],
            m1_results[best_m1][metric],
            m2_results[best_m2][metric]]
    bars = ax.bar(labels, vals, color=bar_colors, alpha=.85,
                  edgecolor='white', linewidth=1.5)
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=.3, axis='y')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fusion_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✅ Saved: fusion_comparison.png")

# ── PLOT 11: MDT Fusion 1-Day Forecast ──
best_m2_models = best_m2.split('&')
combo_preds_best = {n: preds_dict_all[n] for n in best_m2_models}
m2_pred_best, m2_actual_best = method2_ds_fusion(
    combo_preds_best, actuals_arr, WINDOW_SIZE, zeta=0.03)

start, end = 0, min(24, len(m2_pred_best))
hours      = np.arange(end)
actual_kw  = m2_actual_best[:end]
fusion_kw  = m2_pred_best[:end]

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(hours, actual_kw, 'ko-', lw=2.5, ms=7, label='Actual', zorder=5)
ax.plot(hours, fusion_kw, 's-', color='#3498DB', lw=2, ms=5,
        label=f'MDT Fusion ({best_m2})', zorder=4)

for name in MODEL_NAMES:
    p = preds_dict_all[name][WINDOW_SIZE + 1:WINDOW_SIZE + 1 + end]
    ax.plot(hours, p, '--', color=COLORS[name], lw=1.2, alpha=.6, label=name)

ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Wind Power (Normalized)', fontsize=12)
ax.set_title('1-Day Forecast: MDT Fusion vs Single Models',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=.3); ax.set_xticks(hours)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fusion_1day.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: fusion_1day.png")

# ── PLOT 12: All Fusion Combinations — Method 1 vs Method 2 Heatmap ──
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('MDT Fusion: All Combination Metrics', fontsize=16, fontweight='bold')

for ax, (title, dframe) in zip(axes, [('Method 1 (RMSE Preference)', m1_df),
                                       ('Method 2 (DS Evidence)',     m2_df)]):
    numeric = dframe[['MAE', 'RMSE', 'R2']].astype(float)
    sns.heatmap(numeric, annot=True, fmt='.4f', cmap='YlGnBu_r',
                ax=ax, linewidths=.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fusion_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: fusion_heatmap.png")

# ── PLOT 13: Full Test-Set Prediction Overlay ──
fig, ax = plt.subplots(figsize=(18, 5))
n_show = min(200, len(actuals_arr))
x_axis = np.arange(n_show)

ax.plot(x_axis, actuals_arr[:n_show], 'k-', lw=1.5, alpha=.8, label='Actual')
for name in MODEL_NAMES:
    ax.plot(x_axis, predictions[name]['test_pred'][:n_show],
            '-', color=COLORS[name], lw=1, alpha=.65, label=name)

ax.set_xlabel('Test Sample Index', fontsize=12)
ax.set_ylabel('Wind Power (Normalized)', fontsize=12)
ax.set_title('Test-Set Predictions (first 200 hours)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'test_overlay.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: test_overlay.png")

# ════════════════════════════════════════════════════════════
#  Summary
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ALL SAVED FILES")
print("=" * 65)
for root, dirs, files in os.walk(RESULTS):
    for f in sorted(files):
        fpath = os.path.join(root, f)
        size  = os.path.getsize(fpath)
        print(f"  {fpath:55s}  ({size:>10,} bytes)")

print("\n" + "=" * 65)
print("  ✅  FULL EVALUATION COMPLETE — all results & plots saved!")
print("=" * 65)
