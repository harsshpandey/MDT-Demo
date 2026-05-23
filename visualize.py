"""
Visualization Utilities for Wind Power Forecasting
====================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

COLORS = {
    'LSTM': '#FF6B6B',
    'GRU': '#4ECDC4',
    'LSTMCNN': '#45B7D1',
    'GRUCNN': '#96CEB4',
    'actual': '#2C3E50',
    'method1': '#E74C3C',
    'method2': '#3498DB',
}


def plot_training_loss(results_dict, save_path=None):
    """Plot training & validation loss curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training & Validation Loss Curves', fontsize=16, fontweight='bold')
    
    for ax, (name, data) in zip(axes.flatten(), results_dict.items()):
        ax.plot(data['train_losses'], label='Train', color=COLORS.get(name, '#333'), alpha=0.8)
        ax.plot(data['val_losses'], label='Validation', color='orange', alpha=0.8)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions_1day(results_dict, scaler, feature_cols, target_col,
                          day_start=0, save_path=None):
    """
    Plot 1 day (24 hours) of actual vs predicted for all models.
    Uses inverse-transformed values (kW).
    """
    from data_pipeline import inverse_transform_target
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Get 24 hours of data
    sample = list(results_dict.values())[0]
    actual = sample['test_actual'][day_start:day_start + 24]
    
    # Inverse transform
    actual_kw = inverse_transform_target(scaler, actual, feature_cols, target_col)
    hours = np.arange(24)
    
    ax.plot(hours, actual_kw, 'o-', color=COLORS['actual'], linewidth=2.5, 
            markersize=6, label='Actual', zorder=5)
    
    for name, data in results_dict.items():
        pred = data['test_pred'][day_start:day_start + 24]
        pred_kw = inverse_transform_target(scaler, pred, feature_cols, target_col)
        ax.plot(hours, pred_kw, '--', color=COLORS.get(name, '#999'), 
                linewidth=1.8, alpha=0.85, label=name)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Wind Power (kW)', fontsize=12)
    ax.set_title('1-Day Wind Power Forecast Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(hours)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(single_metrics, fusion_m1_metrics=None, 
                           fusion_m2_metrics=None, save_path=None):
    """Bar chart comparing MAE, RMSE, R² across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Evaluation Metrics Comparison', fontsize=16, fontweight='bold')
    
    metric_names = ['MAE', 'RMSE', 'R2']
    
    for ax, metric in zip(axes, metric_names):
        names = []
        values = []
        colors = []
        
        # Single models
        for model_name, m in single_metrics.items():
            names.append(model_name)
            values.append(m[metric])
            colors.append(COLORS.get(model_name, '#999'))
        
        # Fusion Method 1 (best combo)
        if fusion_m1_metrics:
            best_key = min(fusion_m1_metrics, 
                          key=lambda k: fusion_m1_metrics[k].get('MAE', 99))
            names.append(f'M1: {best_key}')
            values.append(fusion_m1_metrics[best_key][metric])
            colors.append(COLORS['method1'])
        
        # Fusion Method 2 (best combo)
        if fusion_m2_metrics:
            best_key = min(fusion_m2_metrics,
                          key=lambda k: fusion_m2_metrics[k].get('MAE', 99))
            names.append(f'M2: {best_key}')
            values.append(fusion_m2_metrics[best_key][metric])
            colors.append(COLORS['method2'])
        
        bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter(results_dict, scaler, feature_cols, target_col, save_path=None):
    """Scatter plot: actual vs predicted for each model."""
    from data_pipeline import inverse_transform_target
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Actual vs Predicted (Scatter)', fontsize=16, fontweight='bold')
    
    for ax, (name, data) in zip(axes.flatten(), results_dict.items()):
        actual = inverse_transform_target(scaler, data['test_actual'], feature_cols, target_col)
        pred = inverse_transform_target(scaler, data['test_pred'], feature_cols, target_col)
        
        ax.scatter(actual, pred, alpha=0.3, s=10, color=COLORS.get(name, '#333'))
        
        # Perfect prediction line
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.7, label='Perfect')
        
        ax.set_xlabel('Actual (kW)')
        ax.set_ylabel('Predicted (kW)')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
