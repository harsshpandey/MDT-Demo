import sys

with open('create_4_notebooks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update notebook 1 markdown & plot
content = content.replace('| Air density (rho) | 1.225 kg/m^3 |', '| Air density (rho) | Dynamic (from P and T) |')
content = content.replace('| Cut-in speed | 3.5 m/s |', '| Cut-in speed | 3.0 m/s |')
content = content.replace('power = compute_wind_power(speeds)', 'power = compute_wind_power(speeds, np.full_like(speeds, 1.225))')
content = content.replace("ax.axvline(x=3.5, color='g', linestyle='--', alpha=0.7, label='Cut-in (3.5 m/s)')", "ax.axvline(x=3.0, color='g', linestyle='--', alpha=0.7, label='Cut-in (3.0 m/s)')")

# 2. Update feature list
old_feat = """| 1-4 | wind_speed_120m/100m/80m/40m | Wind speed at 4 heights |
| 5 | wind_dir_120m | Wind direction at 120m |
| 6 | temp_120m | Temperature at 120m |
| 7 | pressure_100m | Air pressure at 100m |
| 8 | wind_shear | Wind shear exponent |
| 9-10 | hour_sin, hour_cos | Cyclical hour encoding |
| 11-12 | month_sin, month_cos | Cyclical month encoding |
| 13 | temp_gradient | Temperature diff (40m - 120m) |
| 14 | pressure_diff | Pressure diff (40m - 100m) |"""
new_feat = """| 1-2 | wind_speed_120m/100m | Wind speed at 120m and 100m |
| 3 | wind_dir_120m | Wind direction at 120m |
| 4 | temp_120m | Temperature at 120m |
| 5 | pressure_100m | Air pressure at 100m |
| 6 | air_density | Dynamic air density (rho) |
| 7 | wind_speed_cubed | v^3 at 120m |
| 8 | wind_shear | Wind shear exponent |
| 9-10 | hour_sin, hour_cos | Cyclical hour encoding |
| 11-12 | month_sin, month_cos | Cyclical month encoding |
| 13 | temp_gradient | Temperature diff (40m - 120m) |
| 14 | pressure_diff | Pressure diff (40m - 100m) |"""
content = content.replace(old_feat, new_feat)

# 3. Update train_all_models call in Notebook 2
content = content.replace("    save_dir='results',\n    device=DEVICE", "    save_dir='results',\n    device=DEVICE,\n    scaler=scaler")

# 4. Notebook 3 metrics and inverse transform removal
content = content.replace("- **MAE** - Mean Absolute Error\n- **RMSE** - Root Mean Squared Error\n- **R^2** - Coefficient of Determination\n- **MAPE** - Mean Absolute Percentage Error", "- **MAE** - Mean Absolute Error\n- **RMSE** - Root Mean Squared Error\n- **NMAE** - Normalized Mean Absolute Error\n- **MAPE** - Mean Absolute Percentage Error\n- **R^2** - Coefficient of Determination")

content = content.replace("actual_raw = predictions['LSTM']['test_actual'][start:start+24]\n    actual_kw = inverse_transform_target(scaler, actual_raw, feature_cols, target_col)", "actual_kw = predictions['LSTM']['test_actual'][start:start+24]")
content = content.replace("hours = np.arange(len(actual_raw))", "hours = np.arange(len(actual_kw))")
content = content.replace("pred_raw = predictions[name]['test_pred'][start:start+24]\n        pred_kw = inverse_transform_target(scaler, pred_raw, feature_cols, target_col)", "pred_kw = predictions[name]['test_pred'][start:start+24]")

content = content.replace("actual = inverse_transform_target(scaler, predictions[name]['test_actual'], feature_cols, target_col)\n    pred = inverse_transform_target(scaler, predictions[name]['test_pred'], feature_cols, target_col)", "actual = predictions[name]['test_actual']\n    pred = predictions[name]['test_pred']")

# 5. Notebook 4 inverse transform removal
content = content.replace("actual_kw = inverse_transform_target(scaler, m2_actual_best[start:end], feature_cols, target_col)\nfusion_kw = inverse_transform_target(scaler, m2_pred_best[start:end], feature_cols, target_col)", "actual_kw = m2_actual_best[start:end]\nfusion_kw = m2_pred_best[start:end]")

content = content.replace("pred_raw = predictions_dict[name][WINDOW_SIZE+1:WINDOW_SIZE+1+end]\n    pred_kw = inverse_transform_target(scaler, pred_raw, feature_cols, target_col)", "pred_kw = predictions_dict[name][WINDOW_SIZE+1:WINDOW_SIZE+1+end]")

with open('create_4_notebooks.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated create_4_notebooks.py successfully')
