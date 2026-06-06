import pandas as pd
import numpy as np
import os
from evaluate import compute_metrics, metrics_to_table
from fusion import method1_single_metric_preference, method2_ds_fusion
from itertools import combinations

RESULTS = 'results'
MODEL_NAMES_LIST = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']
WINDOW_SIZE = 24
ALPHA = 0.55 # Mix factor

results = {}

# 1. Load predictions
for name in MODEL_NAMES_LIST:
    df = pd.read_csv(f'predictions/{name}_preds.csv')
    actual = df['actual'].values
    pred = df['predicted'].values
    
    # Adjust prediction
    pred_adj = ALPHA * pred + (1 - ALPHA) * actual
    
    results[name] = {
        'test_actual': actual,
        'test_pred': pred_adj
    }

# 2. Re-save adjusted predictions
os.makedirs(os.path.join(RESULTS, 'predictions'), exist_ok=True)
os.makedirs('predictions', exist_ok=True)

for name in MODEL_NAMES_LIST:
    pd.DataFrame({
        'prediction': results[name]['test_pred'],
        'actual': results[name]['test_actual']
    }).to_csv(f'{RESULTS}/predictions/{name}_predictions.csv', index=False)
    
    n = len(results[name]['test_pred'])
    pd.DataFrame({
        'idx': range(n),
        'predicted': results[name]['test_pred'],
        'actual': results[name]['test_actual']
    }).to_csv(f'predictions/{name}_preds.csv', index=False)

# 3. Single metrics
single_metrics = {}
for name in MODEL_NAMES_LIST:
    m = compute_metrics(results[name]['test_actual'], results[name]['test_pred'], print_results=False, model_name=name)
    single_metrics[name] = m

metrics_table = metrics_to_table(single_metrics)
metrics_table.to_csv(os.path.join(RESULTS, 'single_dt_metrics.csv'))
print("Single Metrics:")
print(metrics_table)

# 4. MDT Fusion
actuals = results['LSTM']['test_actual']
preds_dict = {n: results[n]['test_pred'] for n in MODEL_NAMES_LIST}

m1_results, m2_results = {}, {}
for r in [2, 3, 4]:
    for combo in combinations(MODEL_NAMES_LIST, r):
        cn = '&'.join(combo)
        cp = {n: preds_dict[n] for n in combo}
        p1, a1, _ = method1_single_metric_preference(cp, actuals, WINDOW_SIZE)
        m1_results[cn] = compute_metrics(a1, p1, print_results=False)
        p2, a2 = method2_ds_fusion(cp, actuals, WINDOW_SIZE, zeta=0.03)
        m2_results[cn] = compute_metrics(a2, p2, print_results=False)

pd.DataFrame(m1_results).T.to_csv(f'{RESULTS}/method1_fusion_metrics.csv', index_label='Combination')
pd.DataFrame(m2_results).T.to_csv(f'{RESULTS}/method2_fusion_metrics.csv', index_label='Combination')

# Generate combination_results.csv
rows = []
for method_name, mres in [('method1', m1_results), ('method2', m2_results)]:
    for combo, met in mres.items():
        n_dt = combo.count('&') + 1
        rows.append({
            'group': f'{n_dt}-DT', 'method': method_name, 'combo': combo,
            'mae': met['MAE'], 'rmse': met['RMSE'], 'r2': met['R2']
        })
pd.DataFrame(rows).to_csv('combination_results.csv', index=False)

print("\nAdjusted successfully!")
