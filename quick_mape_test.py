from mdt_engine import get_single_dt_metrics, load_twin_data
twins = load_twin_data()
m = get_single_dt_metrics(twins)
for k, v in m.items():
    print(f"{k}: MAE={v['mae']:.4f} RMSE={v['rmse']:.4f} MAPE={v['mape']:.2f}% R2={v['r2']:.4f}")
