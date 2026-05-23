"""
Training Pipeline for Wind Power Forecasting Models
=====================================================
Includes:
  - Custom time-series dataset for PyTorch
  - Training loop with early stopping
  - Model checkpointing
  - Prediction generation for test set
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import get_model
from data_pipeline import inverse_transform_target


# ─────────────────────── Dataset ───────────────────────

class WindTimeSeriesDataset(Dataset):
    """
    Time-series dataset with sliding window.
    Returns (features_window, target_at_next_step).
    """
    def __init__(self, dataframe, feature_cols, target_col, window_size=10):
        self.features = dataframe[feature_cols].values.astype(np.float32)
        self.targets = dataframe[target_col].values.astype(np.float32)
        self.window_size = window_size
    
    def __len__(self):
        return len(self.features) - self.window_size
    
    def __getitem__(self, idx):
        # Window of features: [idx : idx+window_size]
        x = self.features[idx:idx + self.window_size]
        # Target: the value right after the window
        y = self.targets[idx + self.window_size]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ─────────────────────── Training ───────────────────────

def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    window_size: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    epochs: int = 100,
    patience: int = 10,
    save_dir: str = 'results/models',
    device: str = 'cpu'
):
    """
    Train a single model with early stopping.
    Returns: (model, train_losses, val_losses)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = WindTimeSeriesDataset(train_df, feature_cols, target_col, window_size)
    val_dataset = WindTimeSeriesDataset(val_df, feature_cols, target_col, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = len(feature_cols)
    model = get_model(model_name, input_size, window_size)
    model = model.to(device)
    
    criterion = nn.SmoothL1Loss()   # Huber loss — robust to outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"  Input size: {input_size}, Window: {window_size}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}, LR: {lr}, Max epochs: {epochs}")
    
    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        
        train_loss = epoch_loss / len(train_dataset)
        train_losses.append(train_loss)
        
        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch).squeeze()
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * len(X_batch)
        
        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)
        
        # Step the LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or patience_counter == patience:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | "
                  f"LR: {current_lr:.1e} | "
                  f"Pat: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"  Best model saved to {best_model_path}")
    
    return model, train_losses, val_losses


def predict(model, dataframe, feature_cols, target_col, window_size=10, 
            batch_size=64, device='cpu'):
    """
    Generate predictions on a dataset.
    Returns: (predictions, actuals) as numpy arrays
    """
    dataset = WindTimeSeriesDataset(dataframe, feature_cols, target_col, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).squeeze()
            predictions.extend(pred.cpu().numpy().tolist())
            actuals.extend(y_batch.numpy().tolist())
    
    return np.array(predictions), np.array(actuals)


def train_all_models(
    train_df, val_df, test_df, feature_cols, target_col,
    window_size=10, batch_size=64, lr=0.001, epochs=100,
    patience=10, save_dir='results', device='cpu', scaler=None
):
    """
    Train all 4 models and generate test predictions.
    If scaler is provided, inverse-transforms predictions to real kW values.
    Returns: dict of {model_name: {model, train_losses, val_losses, test_pred, test_actual}}
    """
    model_names = ['LSTM', 'GRU', 'LSTMCNN', 'GRUCNN']
    results = {}
    
    for name in model_names:
        model, train_losses, val_losses = train_model(
            model_name=name,
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col=target_col,
            window_size=window_size,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            patience=patience,
            save_dir=os.path.join(save_dir, 'models'),
            device=device
        )
        
        # Generate test predictions
        test_pred, test_actual = predict(
            model, test_df, feature_cols, target_col, 
            window_size=window_size, device=device
        )
        
        # Predictions stay in normalized [0, 1] scale for metric computation
        
        # Save predictions to CSV
        pred_dir = os.path.join(save_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        pred_df = pd.DataFrame({
            'prediction': test_pred,
            'actual': test_actual
        })
        pred_df.to_csv(os.path.join(pred_dir, f'{name}_predictions.csv'), index=False)
        
        results[name] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_pred': test_pred,
            'test_actual': test_actual,
        }
        
        print(f"  Test predictions saved: {len(test_pred)} samples")
    
    return results

