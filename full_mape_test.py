from mdt_engine import load_twin_data, method1_single_metric, method2_multimetric_ds, get_single_dt_metrics, compute_errors
import itertools

twins = load_twin_data()
models = list(twins.keys())

print("=== SINGLE DT METRICS ===")
single = get_single_dt_metrics(twins)
for k, v in single.items():
    flag = "OK" if v['mape'] < 10 else "HIGH"
    print(f"  {k:>10}: MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}  MAPE={v['mape']:.2f}%  R2={v['r2']:.4f}  {flag}")

print("\n=== METHOD 1 (Single Metric Preference) ===")
for r in range(2, 5):
    for combo in itertools.combinations(models, r):
        sub = {k: twins[k] for k in combo}
        m1 = method1_single_metric(sub, window=10)
        m = m1['metrics']
        flag = "OK" if m['mape'] < 10 else "HIGH"
        name = "&".join(combo)
        print(f"  {name:>35}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MAPE={m['mape']:.2f}%  R2={m['r2']:.4f}  {flag}")

print("\n=== METHOD 2 (DS Evidence Fusion) ===")
for r in range(2, 5):
    for combo in itertools.combinations(models, r):
        sub = {k: twins[k] for k in combo}
        m2 = method2_multimetric_ds(sub, window=10, zeta=0.04)
        m = m2['metrics']
        flag = "OK" if m['mape'] < 10 else "HIGH"
        name = "&".join(combo)
        print(f"  {name:>35}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MAPE={m['mape']:.2f}%  R2={m['r2']:.4f}  {flag}")
