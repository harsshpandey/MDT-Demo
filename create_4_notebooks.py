"""Generate 4 separate Jupyter notebooks for MDT Wind Power Forecasting."""
import nbformat as nbf

def make_nb(cells_list, path):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells_list
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"  Created: {path}")

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

# ════════════════════════════════════════════════════════════════
# NOTEBOOK 1: Data & Feature Engineering
# ════════════════════════════════════════════════════════════════
nb1 = [
md("""# 1. Data Loading & Feature Engineering
**Indian Wind Dataset** | SiteID: 36565 | Lat: 23.03 N | Lon: 72.56 E | Year: 2014

- 8,760 hourly meteorological records
- Wind speed at 40m, 80m, 100m, 120m
- Wind direction, temperature, air pressure
- **Target**: Wind power computed using P = 0.5 * rho * A * Cp * v^3 (120m wind speed)
"""),

code("""import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(f"Python: {sys.version}")
np.random.seed(42)
os.makedirs('results/predictions', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
"""),

md("## 1.1 Load Raw Dataset"),

code("""from data_pipeline import load_indian_dataset, engineer_features, prepare_data, compute_wind_power

CSV_PATH = r'36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv'

raw_df = load_indian_dataset(CSV_PATH)
print(f"Shape: {raw_df.shape}")
print(f"\\nColumns: {list(raw_df.columns)}")
print(f"\\nDate range: {raw_df['datetime'].min()} to {raw_df['datetime'].max()}")
raw_df.head(10)
"""),

md("""## 1.2 Wind Power Computation

Using the simplified cubic power curve formula:

**P = 0.5 * rho * A * Cp * v^3**

| Parameter | Value |
|---|---|
| Air density (rho) | Dynamic (from P and T) |
| Rotor diameter | 28 m |
| Rotor area (A) | 615.75 m^2 |
| Power coefficient (Cp) | 0.40 |
| Cut-in speed | 3.0 m/s |
| Cut-out speed | 25.0 m/s |
| Rated power | 250 kW |
"""),

code("""# Plot the power curve
speeds = np.linspace(0, 30, 300)
power = compute_wind_power(speeds, np.full_like(speeds, 1.225))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(speeds, power, 'b-', linewidth=2.5)
ax.axvline(x=3.0, color='g', linestyle='--', alpha=0.7, label='Cut-in (3.0 m/s)')
ax.axvline(x=25.0, color='r', linestyle='--', alpha=0.7, label='Cut-out (25 m/s)')
ax.axhline(y=250, color='orange', linestyle='--', alpha=0.7, label='Rated (250 kW)')
ax.fill_between(speeds, power, alpha=0.15, color='blue')
ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
ax.set_ylabel('Power (kW)', fontsize=12)
ax.set_title('Wind Turbine Power Curve: P = 0.5 * rho * A * Cp * v^3', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),

md("""## 1.3 Feature Engineering

15 features from raw meteorological data (plus derived air density):

| # | Feature | Description |
|---|---|---|
| 1-4 | wind_speed_40m/80m/100m/120m | Wind speed at 4 heights |
| 5-8 | wind_dir_40m/80m/100m/120m | Wind direction at 4 heights |
| 9-12 | temp_40m/80m/100m/120m | Temperature at 4 heights |
| 13-14 | pressure_40m/100m | Air pressure at 40m and 100m |
| 15 | air_density | Dynamic air density (rho) |
"""),

code("""feat_df = engineer_features(raw_df)
print(f"Engineered dataset shape: {feat_df.shape}")
print(f"\\nColumns ({len(feat_df.columns)}):")
for i, col in enumerate(feat_df.columns):
    print(f"  {i+1:2d}. {col}")

print("\\n--- Wind Power Statistics (kW) ---")
print(feat_df['wind_power'].describe())
"""),

md("## 1.4 Data Exploration"),

code("""# Time series plots
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

axes[0].plot(feat_df['datetime'], feat_df['wind_speed_120m'], color='#45B7D1', linewidth=0.5, alpha=0.7)
axes[0].set_ylabel('Wind Speed (m/s)', fontsize=12)
axes[0].set_title('Wind Speed at 120m - Full Year 2014', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(feat_df['datetime'], feat_df['wind_power'], color='#E74C3C', linewidth=0.5, alpha=0.7)
axes[1].set_ylabel('Wind Power (Normalized)', fontsize=12)
axes[1].set_title('Computed Wind Power (P = 0.5 * rho * A * Cp * v^3) [Sabarmati 28m]', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

axes[2].plot(feat_df['datetime'], feat_df['temp_120m'], color='#FF6B6B', linewidth=0.5, alpha=0.7)
axes[2].set_ylabel('Temperature (C)', fontsize=12)
axes[2].set_title('Temperature at 120m', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/timeseries.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

code("""# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
corr = feat_df.drop(columns=['datetime']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/correlation.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

code("""# Distribution plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(feat_df['wind_speed_120m'], bins=50, color='#45B7D1', alpha=0.7, edgecolor='white')
axes[0].set_xlabel('Wind Speed (m/s)')
axes[0].set_title('Wind Speed Distribution (120m)', fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(feat_df['wind_power'], bins=50, color='#E74C3C', alpha=0.7, edgecolor='white')
axes[1].set_xlabel('Wind Power (Normalized)')
axes[1].set_title('Wind Power Distribution', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/distributions.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 1.5 Data Splitting & Normalization"),

code("""WINDOW_SIZE = 10
train_df, val_df, test_df, scaler, feature_cols, target_col = prepare_data(CSV_PATH, save_dir='results')

print(f"\\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols):
    print(f"  {i+1:2d}. {col}")
print(f"Target: {target_col}")
"""),

code("""print("=== Train (first 5 rows) ===")
print(train_df[feature_cols + [target_col]].head())
print("\\n=== Validation (first 5 rows) ===")
print(val_df[feature_cols + [target_col]].head())
print("\\n=== Test (first 5 rows) ===")
print(test_df[feature_cols + [target_col]].head())
"""),

md("## 1.6 Save Processed Data"),

code("""# Save processed dataframes
train_df.to_csv('results/train_data.csv', index=False)
val_df.to_csv('results/val_data.csv', index=False)
test_df.to_csv('results/test_data.csv', index=False)

# Save config
config = {
    'feature_cols': feature_cols,
    'target_col': target_col,
    'window_size': WINDOW_SIZE
}
with open('results/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("[OK] Saved files:")
print("  results/train_data.csv")
print("  results/val_data.csv")
print("  results/test_data.csv")
print("  results/config.json")
print("  results/scaler.save")
"""),
]

# ════════════════════════════════════════════════════════════════
# NOTEBOOK 2: Model Training
# ════════════════════════════════════════════════════════════════
nb2 = [
md("""# 2. Model Training
**Models**: LSTM, GRU, LSTMCNN, GRUCNN

Training with early stopping, window size = 10, batch size = 64
"""),

code("""import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
np.random.seed(42)
torch.manual_seed(42)

# Load processed data from Notebook 1
train_df = pd.read_csv('results/train_data.csv')
val_df = pd.read_csv('results/val_data.csv')
test_df = pd.read_csv('results/test_data.csv')
scaler = joblib.load('results/scaler.save')

with open('results/config.json', 'r') as f:
    config = json.load(f)
feature_cols = config['feature_cols']
target_col = config['target_col']
WINDOW_SIZE = config['window_size']

print(f"\\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Features: {len(feature_cols)}, Window: {WINDOW_SIZE}")
"""),

md("""## 2.1 Model Architectures

| Model | Architecture |
|---|---|
| **LSTM** | LSTM(hidden=10, layers=3) -> Linear(10,1) |
| **GRU** | GRU(hidden=10, layers=3) -> Linear(10,1) |
| **LSTMCNN** | LSTM(hidden=16, layers=1) -> Conv1d(16,8,k=1) -> MaxPool(4) -> FC(16) -> FC(1) |
| **GRUCNN** | GRU(hidden=16, layers=1) -> Conv1d(16,8,k=1) -> MaxPool(4) -> FC(16) -> FC(1) |
"""),

code("""from models import get_model

input_size = len(feature_cols)
for name in ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']:
    model = get_model(name, input_size, WINDOW_SIZE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\\n{'='*50}")
    print(f"{name} ({total_params:,} parameters)")
    print(f"{'='*50}")
    print(model)
"""),

md("""## 2.2 Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | MSE |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Window size | 10 |
"""),

md("## 2.3 Train All Models"),

code("""from train import train_all_models

results = train_all_models(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    feature_cols=feature_cols,
    target_col=target_col,
    window_size=WINDOW_SIZE,
    batch_size=64,
    lr=0.001,
    epochs=100,
    patience=10,
    save_dir='results',
    device=DEVICE,
    scaler=scaler
)

print("\\n[OK] All 4 models trained successfully!")
"""),

md("## 2.4 Training Loss Curves"),

code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training & Validation Loss Curves', fontsize=16, fontweight='bold')
colors = {'LSTM': '#FF6B6B', 'GRU': '#4ECDC4', 'LSTMCNN': '#45B7D1', 'GRUCNN': '#96CEB4'}

for ax, (name, data) in zip(axes.flatten(), results.items()):
    ax.plot(data['train_losses'], label='Train', color=colors[name], linewidth=2, alpha=0.8)
    ax.plot(data['val_losses'], label='Validation', color='orange', linewidth=2, alpha=0.8)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/training_loss.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 2.5 Save Predictions"),

code("""import json as json2

os.makedirs('results/predictions', exist_ok=True)

for name, data in results.items():
    # Save predictions
    pred_df = pd.DataFrame({
        'prediction': data['test_pred'],
        'actual': data['test_actual']
    })
    pred_df.to_csv(f'results/predictions/{name}_predictions.csv', index=False)
    
    # Save loss history
    loss_df = pd.DataFrame({
        'train_loss': data['train_losses'],
        'val_loss': data['val_losses']
    })
    loss_df.to_csv(f'results/predictions/{name}_losses.csv', index=False)
    
    print(f"  {name}: {len(data['test_pred'])} predictions saved, {len(data['train_losses'])} epochs")

print("\\n[OK] All predictions and loss histories saved!")
"""),
]

# ════════════════════════════════════════════════════════════════
# NOTEBOOK 3: Evaluation Metrics
# ════════════════════════════════════════════════════════════════
nb3 = [
md("""# 3. Evaluation Metrics

Evaluating each single Digital Twin on the **test set** (10% of data):
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Squared Error
- **NMAE** - Normalized Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error
- **R^2** - Coefficient of Determination
"""),

code("""import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from evaluate import compute_metrics, metrics_to_table
from data_pipeline import # inverse_transform_target

# Load config
with open('results/config.json', 'r') as f:
    config = json.load(f)
feature_cols = config['feature_cols']
target_col = config['target_col']
scaler = joblib.load('results/scaler.save')

# Load predictions
model_names = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']
predictions = {}
for name in model_names:
    df = pd.read_csv(f'results/predictions/{name}_predictions.csv')
    predictions[name] = {'test_pred': df['prediction'].values, 'test_actual': df['actual'].values}

print(f"[OK] Loaded predictions for {len(predictions)} models")
print(f"Test samples per model: {len(predictions['LSTM']['test_pred'])}")
"""),

md("## 3.1 Single Digital Twin Metrics"),

code("""single_metrics = {}
for name in model_names:
    m = compute_metrics(
        predictions[name]['test_actual'],
        predictions[name]['test_pred'],
        print_results=True,
        model_name=name
    )
    single_metrics[name] = m

# Display as table
metrics_table = metrics_to_table(single_metrics)
print("\\n" + "="*60)
print("  SINGLE DIGITAL TWIN - TEST SET METRICS")
print("="*60)
print(metrics_table.to_string())
"""),

md("## 3.2 Metrics Comparison Chart"),

code("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Single DT Metrics Comparison', fontsize=16, fontweight='bold')
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
    vals = [single_metrics[n][metric] for n in model_names]
    bars = ax.bar(model_names, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/single_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 3.3 Actual vs Predicted - 1-Day Forecast"),

code("""colors_map = {'LSTM': '#FF6B6B', 'GRU': '#4ECDC4', 'LSTMCNN': '#45B7D1', 'GRUCNN': '#96CEB4'}

for day_idx, start in enumerate([0, 48, 96]):
    fig, ax = plt.subplots(figsize=(16, 5))
    
    actual_kw = predictions['LSTM']['test_actual'][start:start+24]
    hours = np.arange(len(actual_kw))
    
    ax.plot(hours, actual_kw, 'ko-', linewidth=2.5, markersize=6, label='Actual', zorder=5)
    
    for name in model_names:
        pred_kw = predictions[name]['test_pred'][start:start+24]
        ax.plot(hours, pred_kw, '--', color=colors_map[name], linewidth=1.8, alpha=0.85, label=name)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Wind Power (Normalized)', fontsize=12)
    ax.set_title(f'Day {day_idx+1} - Wind Power Forecast Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(hours)
    plt.tight_layout()
    plt.savefig(f'results/plots/day_{day_idx+1}_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
"""),

md("## 3.4 Scatter Plots"),

code("""fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Actual vs Predicted (Scatter)', fontsize=16, fontweight='bold')

for ax, name, c in zip(axes.flatten(), model_names, colors):
    actual = predictions[name]['test_actual']
    pred = predictions[name]['test_pred']
    
    ax.scatter(actual, pred, alpha=0.3, s=10, color=c)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.7, label='Perfect')
    ax.set_xlabel('Actual (Normalized)')
    ax.set_ylabel('Predicted (Normalized)')
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/scatter.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 3.5 Error Distribution"),

code("""fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')

for ax, name, c in zip(axes.flatten(), model_names, colors):
    actual = predictions[name]['test_actual']
    pred = predictions[name]['test_pred']
    errors = actual - pred
    
    ax.hist(errors, bins=50, color=c, alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Error (Normalized)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} (mean={errors.mean():.2f}, std={errors.std():.2f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/error_dist.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 3.6 Save Metrics"),

code("""metrics_table.to_csv('results/single_dt_metrics.csv')
print("[OK] Saved: results/single_dt_metrics.csv")
print(metrics_table)
"""),
]

# ════════════════════════════════════════════════════════════════
# NOTEBOOK 4: MDT Fusion
# ════════════════════════════════════════════════════════════════
nb4 = [
md("""# 4. Multi-Digital Twin Fusion

Two fusion methods under the multi-twin cooperative operation mechanism:

**Method 1: Single Metric Dynamic Preference (Time Window)**
- Selects the best model (lowest RMSE) in each sliding window of size 10
- Window slides one step at a time

**Method 2: Multi-Metrics Dynamic Fusion (DS Evidence Theory)**
- Computes BPAs from RMSE, MAE, R^2 for each model
- Fuses using Dempster-Shafer combination rule
- Threshold zeta = 0.03 for winner-take-all

Tests all combinations: 2-DT, 3-DT, and 4-DT
"""),

code("""import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from evaluate import compute_metrics, metrics_to_table
from fusion import method1_single_metric_preference, method2_ds_fusion, run_all_fusion
from data_pipeline import # inverse_transform_target

# Load config and data
with open('results/config.json', 'r') as f:
    config = json.load(f)
feature_cols = config['feature_cols']
target_col = config['target_col']
WINDOW_SIZE = config['window_size']
scaler = joblib.load('results/scaler.save')

# Load predictions
model_names = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']
predictions_dict = {}
for name in model_names:
    df = pd.read_csv(f'results/predictions/{name}_predictions.csv')
    predictions_dict[name] = df['prediction'].values

actuals = pd.read_csv(f'results/predictions/LSTM_predictions.csv')['actual'].values

print(f"[OK] Loaded {len(model_names)} model predictions ({len(actuals)} test samples)")
print(f"Window size: {WINDOW_SIZE}")
"""),

md("""## 4.1 Method 1: Single Metric Dynamic Preference

For each prediction point:
1. Look at the past 10 points (time window)
2. Compute RMSE for each model in that window
3. Select the model with **lowest RMSE** for the next prediction
"""),

code("""from itertools import combinations

m1_results = {}
for r in [2, 3, 4]:
    for combo in combinations(model_names, r):
        combo_name = '&'.join(combo)
        combo_preds = {n: predictions_dict[n] for n in combo}
        
        m1_pred, m1_actual, selections = method1_single_metric_preference(
            combo_preds, actuals, WINDOW_SIZE
        )
        m1_metrics = compute_metrics(m1_actual, m1_pred, print_results=False)
        m1_results[combo_name] = m1_metrics
        print(f"  {combo_name}: MAE={m1_metrics['MAE']}, RMSE={m1_metrics['RMSE']}, R2={m1_metrics['R2']}")

print(f"\\n[OK] Method 1 computed for {len(m1_results)} combinations")
"""),

code("""# Method 1 Results Table
m1_df = pd.DataFrame(m1_results).T
m1_df.index.name = 'Combination'
print("="*70)
print("  METHOD 1: Single Metric Dynamic Preference (RMSE-based)")
print("="*70)
print(m1_df.to_string())
"""),

md("""## 4.2 Method 2: Multi-Metrics DS Evidence Fusion

For each prediction point:
1. Compute BPAs from **RMSE, MAE, R^2** for each model
2. Fuse using **Dempster-Shafer combination rule**
3. If variance of weights > zeta (0.03), use **winner-take-all**
4. Otherwise, use **weighted combination** of predictions
"""),

code("""m2_results = {}
for r in [2, 3, 4]:
    for combo in combinations(model_names, r):
        combo_name = '&'.join(combo)
        combo_preds = {n: predictions_dict[n] for n in combo}
        
        m2_pred, m2_actual = method2_ds_fusion(
            combo_preds, actuals, WINDOW_SIZE, zeta=0.03
        )
        m2_metrics = compute_metrics(m2_actual, m2_pred, print_results=False)
        m2_results[combo_name] = m2_metrics
        print(f"  {combo_name}: MAE={m2_metrics['MAE']}, RMSE={m2_metrics['RMSE']}, R2={m2_metrics['R2']}")

print(f"\\n[OK] Method 2 computed for {len(m2_results)} combinations")
"""),

code("""# Method 2 Results Table
m2_df = pd.DataFrame(m2_results).T
m2_df.index.name = 'Combination'
print("="*70)
print("  METHOD 2: Multi-Metrics Dynamic Fusion (DS Evidence Theory)")
print("="*70)
print(m2_df.to_string())
"""),

md("## 4.3 Comparison: Single DT vs MDT"),

code("""# Load single DT metrics
single_df = pd.read_csv('results/single_dt_metrics.csv', index_col=0)
single_metrics = single_df.to_dict('index')

best_single = min(single_metrics, key=lambda k: single_metrics[k]['MAE'])
best_m1 = min(m1_results, key=lambda k: m1_results[k]['MAE'])
best_m2 = min(m2_results, key=lambda k: m2_results[k]['MAE'])

print("="*70)
print("  BEST RESULTS SUMMARY")
print("="*70)
print(f"  Best Single DT: {best_single}")
print(f"    MAE={single_metrics[best_single]['MAE']}, RMSE={single_metrics[best_single]['RMSE']}, R2={single_metrics[best_single]['R2']}")
print(f"")
print(f"  Best Method 1:  {best_m1}")
print(f"    MAE={m1_results[best_m1]['MAE']}, RMSE={m1_results[best_m1]['RMSE']}, R2={m1_results[best_m1]['R2']}")
print(f"")
print(f"  Best Method 2:  {best_m2}")
print(f"    MAE={m2_results[best_m2]['MAE']}, RMSE={m2_results[best_m2]['RMSE']}, R2={m2_results[best_m2]['R2']}")
"""),

code("""# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Single DT vs MDT Fusion - Best Results', fontsize=16, fontweight='bold')

labels = [f'Single\\n({best_single})', f'Method 1\\n({best_m1})', f'Method 2\\n({best_m2})']
bar_colors = ['#96CEB4', '#E74C3C', '#3498DB']

for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
    vals = [
        single_metrics[best_single][metric],
        m1_results[best_m1][metric],
        m2_results[best_m2][metric]
    ]
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/fusion_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 4.4 1-Day Forecast: Best Fusion vs Single Models"),

code("""# Run best fusion on test data for plotting
best_m2_models = best_m2.split('&')
combo_preds = {n: predictions_dict[n] for n in best_m2_models}
m2_pred_best, m2_actual_best = method2_ds_fusion(combo_preds, actuals, WINDOW_SIZE, zeta=0.03)

# Plot 1 day
start = 0
end = min(24, len(m2_pred_best))
hours = np.arange(end)

actual_kw = m2_actual_best[start:end]
fusion_kw = m2_pred_best[start:end]

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(hours, actual_kw, 'ko-', linewidth=2.5, markersize=7, label='Actual', zorder=5)
ax.plot(hours, fusion_kw, 's-', color='#3498DB', linewidth=2, markersize=5, label=f'MDT Fusion ({best_m2})', zorder=4)

# Also plot individual models
colors_map = {'LSTM': '#FF6B6B', 'GRU': '#4ECDC4', 'LSTMCNN': '#45B7D1', 'GRUCNN': '#96CEB4'}
for name in model_names:
    pred_kw = predictions_dict[name][WINDOW_SIZE+1:WINDOW_SIZE+1+end]
    ax.plot(hours, pred_kw, '--', color=colors_map[name], linewidth=1.2, alpha=0.6, label=name)

ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Wind Power (Normalized)', fontsize=12)
ax.set_title('1-Day Forecast: MDT Fusion vs Single Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(hours)
plt.tight_layout()
plt.savefig('results/plots/fusion_1day.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 4.5 Save All Results"),

code("""m1_df.to_csv('results/method1_fusion_metrics.csv')
m2_df.to_csv('results/method2_fusion_metrics.csv')

print("[OK] Saved:")
print("  results/method1_fusion_metrics.csv")
print("  results/method2_fusion_metrics.csv")
print()
print("=== All Results Files ===")
for root, dirs, files in os.walk('results'):
    for f in sorted(files):
        print(f"  {os.path.join(root, f)}")
"""),
]

# ════════════════════════════════════════════════════════════════
# Generate all 4 notebooks
# ════════════════════════════════════════════════════════════════
print("Generating 4 notebooks...")
make_nb(nb1, '01_Data_Feature_Engineering.ipynb')
make_nb(nb2, '02_Model_Training.ipynb')
make_nb(nb3, '03_Evaluation_Metrics.ipynb')
make_nb(nb4, '04_MDT_Fusion.ipynb')
print("\n[OK] All 4 notebooks created!")
