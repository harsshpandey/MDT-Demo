import urllib.request, json

# Check single DT metrics
d = json.loads(urllib.request.urlopen('http://localhost:5001/api/metrics').read())
print("=== Single DT Metrics ===")
for k, v in d.items():
    print(f"  {k}: MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}  MAPE={v['mape']:.2f}%  R2={v['r2']:.4f}")

# Check combinations
c = json.loads(urllib.request.urlopen('http://localhost:5001/api/combinations').read())
print("\n=== Method 1 (Single Metric) ===")
for group in ['2-DT', '3-DT', '4-DT']:
    if group in c and 'method1' in c[group]:
        for row in c[group]['method1']:
            print(f"  {row['combo']:>30}: MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}  MAPE={row['mape']:.2f}%  R2={row['r2']:.4f}")

print("\n=== Method 2 (DS Evidence) ===")
for group in ['2-DT', '3-DT', '4-DT']:
    if group in c and 'method2' in c[group]:
        for row in c[group]['method2']:
            print(f"  {row['combo']:>30}: MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}  MAPE={row['mape']:.2f}%  R2={row['r2']:.4f}")
