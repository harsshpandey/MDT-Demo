import pandas as pd
from mdt_engine import load_twin_data, method1_single_metric, method2_multimetric_ds
import itertools

twins = load_twin_data()
models = list(twins.keys())

results = []

# Generate all combinations of length 2, 3, 4
for r in range(2, 5):
    for combo in itertools.combinations(models, r):
        group = f"{r}-DT"
        combo_name = "&".join(combo)
        sub_twins = {k: twins[k] for k in combo}
        
        # Method 1
        m1 = method1_single_metric(sub_twins, window=10)
        metrics1 = m1["metrics"]
        results.append({
            "group": group, "method": "method1", "combo": combo_name,
            "mae": metrics1["mae"], "rmse": metrics1["rmse"], 
            "mape": metrics1.get("mape", 0.0), "r2": metrics1["r2"]
        })
        
        # Method 2
        m2 = method2_multimetric_ds(sub_twins, window=10, zeta=0.04)
        metrics2 = m2["metrics"]
        results.append({
            "group": group, "method": "method2", "combo": combo_name,
            "mae": metrics2["mae"], "rmse": metrics2["rmse"], 
            "mape": metrics2.get("mape", 0.0), "r2": metrics2["r2"]
        })

df = pd.DataFrame(results)
# Sort to match original order (Method 1 then Method 2)
df = df.sort_values(by=["method", "group", "combo"]).reset_index(drop=True)
df.to_csv("combination_results.csv", index=False)
print("Updated combination_results.csv with MAPE.")
