"""
Quick end-to-end test: train 1 model for 5 epochs to verify pipeline.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from data_pipeline import prepare_data
from train import train_model, predict
from evaluate import compute_metrics

CSV_PATH = r'36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv'

print("Loading data...")
train_df, val_df, test_df, scaler, feature_cols, target_col = prepare_data(CSV_PATH, save_dir='results')

print("\nTraining LSTM (5 epochs quick test)...")
model, tl, vl = train_model(
    model_name='LSTM',
    train_df=train_df, val_df=val_df,
    feature_cols=feature_cols, target_col=target_col,
    window_size=10, batch_size=64, lr=0.01,
    epochs=5, patience=10,
    save_dir='results/models'
)

print("\nGenerating predictions...")
pred, actual = predict(model, test_df, feature_cols, target_col, window_size=10)
print(f"Predictions: {len(pred)} samples")

metrics = compute_metrics(actual, pred, print_results=True, model_name='LSTM')
print("\n[OK] End-to-end pipeline verified successfully!")
