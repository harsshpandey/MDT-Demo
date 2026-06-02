"""
Retrain all models with enhanced features + larger architectures.
Target: MAE < 2, MAPE < 10%, R² > 0.98
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import joblib

from data_pipeline import prepare_data, inverse_transform_target
from train import train_all_models
from evaluate import compute_metrics, metrics_to_table
from fusion import method1_single_metric_preference, method2_ds_fusion

CSV_PATH    = r'36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv'
RESULTS     = 'results'
WINDOW_SIZE = 24
BATCH_SIZE  = 32
LR          = 0.0003
EPOCHS      = 800
PATIENCE    = 80

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(42)
torch.manual_seed(42)

print("=" * 65)
print("  STEP 1: Enhanced Data Pipeline")
print("=" * 65)
train_df, val_df, test_df, scaler, feature_cols, target_col = prepare_data(CSV_PATH, save_dir=RESULTS)
print(f"\n  Features ({len(feature_cols)}): {feature_cols}")
print(f"  Target: {target_col}")
print(f"  Device: {DEVICE}")

# Save config
import json
config = {'feature_cols': feature_cols, 'target_col': target_col, 'window_size': WINDOW_SIZE}
with open(os.path.join(RESULTS, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 65)
print("  STEP 2: Train All Models (Enhanced)")
print("=" * 65)

results = train_all_models(
    train_df=train_df, val_df=val_df, test_df=test_df,
    feature_cols=feature_cols, target_col=target_col,
    window_size=WINDOW_SIZE, batch_size=BATCH_SIZE,
    lr=LR, epochs=EPOCHS, patience=PATIENCE,
    save_dir=RESULTS, device=DEVICE, scaler=scaler
)

print("\n" + "=" * 65)
print("  STEP 3: Evaluation Metrics")
print("=" * 65)

MODEL_NAMES = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']
single_metrics = {}
for name in MODEL_NAMES:
    m = compute_metrics(results[name]['test_actual'], results[name]['test_pred'],
                        print_results=True, model_name=name)
    single_metrics[name] = m

metrics_table = metrics_to_table(single_metrics)
metrics_table.to_csv(os.path.join(RESULTS, 'single_dt_metrics.csv'))
print(f"\n{metrics_table.to_string()}")

# Save predictions & losses
os.makedirs(os.path.join(RESULTS, 'predictions'), exist_ok=True)
for name in MODEL_NAMES:
    pd.DataFrame({'prediction': results[name]['test_pred'],
                  'actual': results[name]['test_actual']
                  }).to_csv(f'{RESULTS}/predictions/{name}_predictions.csv', index=False)
    pd.DataFrame({'train_loss': results[name]['train_losses'],
                  'val_loss': results[name]['val_losses']
                  }).to_csv(f'{RESULTS}/predictions/{name}_losses.csv', index=False)

# Update predictions/ for Flask dashboard
os.makedirs('predictions', exist_ok=True)
for name in MODEL_NAMES:
    n = len(results[name]['test_pred'])
    pd.DataFrame({'idx': range(n), 'predicted': results[name]['test_pred'],
                  'actual': results[name]['test_actual']
                  }).to_csv(f'predictions/{name}_preds.csv', index=False)

print("\n" + "=" * 65)
print("  STEP 4: MDT Fusion")
print("=" * 65)

actuals = results['LSTM']['test_actual']
preds_dict = {n: results[n]['test_pred'] for n in MODEL_NAMES}

from itertools import combinations

m1_results, m2_results = {}, {}
for r in [2, 3, 4]:
    for combo in combinations(MODEL_NAMES, r):
        cn = '&'.join(combo)
        cp = {n: preds_dict[n] for n in combo}
        p1, a1, _ = method1_single_metric_preference(cp, actuals, WINDOW_SIZE)
        m1_results[cn] = compute_metrics(a1, p1, print_results=False)
        p2, a2 = method2_ds_fusion(cp, actuals, WINDOW_SIZE, zeta=0.03)
        m2_results[cn] = compute_metrics(a2, p2, print_results=False)
        print(f"  {cn:40s}  M1 MAE={m1_results[cn]['MAE']:7.4f}  M2 MAE={m2_results[cn]['MAE']:7.4f}")

pd.DataFrame(m1_results).T.to_csv(f'{RESULTS}/method1_fusion_metrics.csv', index_label='Combination')
pd.DataFrame(m2_results).T.to_csv(f'{RESULTS}/method2_fusion_metrics.csv', index_label='Combination')

# Update combination_results.csv for Flask
rows = []
for method_name, mres in [('method1', m1_results), ('method2', m2_results)]:
    for combo, met in mres.items():
        n = combo.count('&') + 1
        rows.append({'group': f'{n}-DT', 'method': method_name, 'combo': combo,
                     'mae': met['MAE'], 'rmse': met['RMSE'], 'r2': met['R2']})
pd.DataFrame(rows).to_csv('combination_results.csv', index=False)

best_s = min(single_metrics, key=lambda k: single_metrics[k]['MAE'])
best_m1 = min(m1_results, key=lambda k: m1_results[k]['MAE'])
best_m2 = min(m2_results, key=lambda k: m2_results[k]['MAE'])

print(f"\n{'='*65}")
print(f"  FINAL RESULTS")
print(f"{'='*65}")
print(f"  Best Single: {best_s}  MAE={single_metrics[best_s]['MAE']:.4f}  MAPE={single_metrics[best_s]['MAPE']:.2f}%  R²={single_metrics[best_s]['R2']:.4f}")
print(f"  Best M1:     {best_m1}  MAE={m1_results[best_m1]['MAE']:.4f}  R²={m1_results[best_m1]['R2']:.4f}")
print(f"  Best M2:     {best_m2}  MAE={m2_results[best_m2]['MAE']:.4f}  R²={m2_results[best_m2]['R2']:.4f}")
print(f"\n  ✅ All results saved!")
