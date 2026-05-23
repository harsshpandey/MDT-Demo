"""
Re-evaluate saved models on normalized predictions (no inverse transform).
Uses the already-trained V2 models (hidden=128, attention, Huber loss).
"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import joblib

from data_pipeline import prepare_data
from train import predict, WindTimeSeriesDataset
from models import get_model
from evaluate import compute_metrics, metrics_to_table
from fusion import method1_single_metric_preference, method2_ds_fusion
from itertools import combinations

CSV_PATH    = r'36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv'
RESULTS     = 'results'
WINDOW_SIZE = 24
MODEL_NAMES = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']

print("=" * 65)
print("  Loading data & models...")
print("=" * 65)

train_df, val_df, test_df, scaler, feature_cols, target_col = prepare_data(CSV_PATH, save_dir=RESULTS)
input_size = len(feature_cols)
print(f"  Features: {input_size}, Test samples: {len(test_df)}")

# ── Load trained models and generate normalized predictions ──
predictions = {}
for name in MODEL_NAMES:
    model = get_model(name, input_size, WINDOW_SIZE)
    model_path = os.path.join(RESULTS, 'models', f'{name}_best.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    test_pred, test_actual = predict(
        model, test_df, feature_cols, target_col,
        window_size=WINDOW_SIZE, device='cpu'
    )
    # NO inverse transform — stay in normalized [0,1] scale
    predictions[name] = {
        'test_pred': test_pred,
        'test_actual': test_actual,
    }
    print(f"  {name}: pred range [{test_pred.min():.4f}, {test_pred.max():.4f}], "
          f"actual range [{test_actual.min():.4f}, {test_actual.max():.4f}]")

# ── Compute metrics (on normalized scale) ──
print(f"\n{'='*65}")
print(f"  Single Digital Twin Metrics (Normalized Scale)")
print(f"{'='*65}")

single_metrics = {}
for name in MODEL_NAMES:
    m = compute_metrics(predictions[name]['test_actual'],
                        predictions[name]['test_pred'],
                        print_results=True, model_name=name)
    single_metrics[name] = m

metrics_table = metrics_to_table(single_metrics)
metrics_table.to_csv(os.path.join(RESULTS, 'single_dt_metrics.csv'))
print(f"\n{metrics_table.to_string()}")

# ── Save predictions ──
os.makedirs(os.path.join(RESULTS, 'predictions'), exist_ok=True)
for name in MODEL_NAMES:
    pd.DataFrame({
        'prediction': predictions[name]['test_pred'],
        'actual': predictions[name]['test_actual']
    }).to_csv(f'{RESULTS}/predictions/{name}_predictions.csv', index=False)

# ── Update predictions/ for Flask dashboard ──
os.makedirs('predictions', exist_ok=True)
for name in MODEL_NAMES:
    n = len(predictions[name]['test_pred'])
    pd.DataFrame({
        'idx': range(n),
        'predicted': predictions[name]['test_pred'],
        'actual': predictions[name]['test_actual']
    }).to_csv(f'predictions/{name}_preds.csv', index=False)

# ── MDT Fusion ──
print(f"\n{'='*65}")
print(f"  MDT Fusion (Normalized Scale)")
print(f"{'='*65}")

actuals = predictions['LSTM']['test_actual']
preds_dict = {n: predictions[n]['test_pred'] for n in MODEL_NAMES}

m1_results, m2_results = {}, {}
for r in [2, 3, 4]:
    for combo in combinations(MODEL_NAMES, r):
        cn = '&'.join(combo)
        cp = {n: preds_dict[n] for n in combo}
        p1, a1, _ = method1_single_metric_preference(cp, actuals, WINDOW_SIZE)
        m1_results[cn] = compute_metrics(a1, p1, print_results=False)
        p2, a2 = method2_ds_fusion(cp, actuals, WINDOW_SIZE, zeta=0.03)
        m2_results[cn] = compute_metrics(a2, p2, print_results=False)
        print(f"  {cn:40s}  M1: MAE={m1_results[cn]['MAE']:.4f} MAPE={m1_results[cn]['MAPE']:.2f}%  "
              f"M2: MAE={m2_results[cn]['MAE']:.4f} MAPE={m2_results[cn]['MAPE']:.2f}%")

pd.DataFrame(m1_results).T.to_csv(f'{RESULTS}/method1_fusion_metrics.csv', index_label='Combination')
pd.DataFrame(m2_results).T.to_csv(f'{RESULTS}/method2_fusion_metrics.csv', index_label='Combination')

# ── Update combination_results.csv ──
rows = []
for method_name, mres in [('method1', m1_results), ('method2', m2_results)]:
    for combo, met in mres.items():
        n = combo.count('&') + 1
        rows.append({'group': f'{n}-DT', 'method': method_name, 'combo': combo,
                     'mae': met['MAE'], 'rmse': met['RMSE'], 'r2': met['R2']})
pd.DataFrame(rows).to_csv('combination_results.csv', index=False)

# ── Summary ──
best_s = min(single_metrics, key=lambda k: single_metrics[k]['MAE'])
best_m1 = min(m1_results, key=lambda k: m1_results[k]['MAE'])
best_m2 = min(m2_results, key=lambda k: m2_results[k]['MAE'])

print(f"\n{'='*65}")
print(f"  FINAL RESULTS (Normalized [0,1] Scale)")
print(f"{'='*65}")
print(f"  Best Single: {best_s}")
print(f"    MAE={single_metrics[best_s]['MAE']:.4f}  MAPE={single_metrics[best_s]['MAPE']:.2f}%  R²={single_metrics[best_s]['R2']:.4f}")
print(f"  Best M1:     {best_m1}")
print(f"    MAE={m1_results[best_m1]['MAE']:.4f}  MAPE={m1_results[best_m1]['MAPE']:.2f}%  R²={m1_results[best_m1]['R2']:.4f}")
print(f"  Best M2:     {best_m2}")
print(f"    MAE={m2_results[best_m2]['MAE']:.4f}  MAPE={m2_results[best_m2]['MAPE']:.2f}%  R²={m2_results[best_m2]['R2']:.4f}")
print(f"\n  ✅ All results saved!")
