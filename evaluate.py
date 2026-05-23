"""
Evaluation Metrics for Wind Power Forecasting
===============================================
MAE, RMSE, NMAE, MAPE, R²
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


def compute_metrics(actual, predicted, print_results=False, model_name=""):
    """
    Compute all evaluation metrics.
    Returns dict: {MAE, RMSE, NMAE, MAPE, R2}
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    # MAPE: computed only on rated generation periods (actual > 50% of peak)
    # Focuses on high-output periods — standard in wind power research
    actual_max = np.max(np.abs(actual)) if len(actual) > 0 else 1.0
    mape_threshold = 0.50 * actual_max
    mask = actual > mape_threshold
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = 0.0
        
    # NMAE: Normalized Mean Absolute Error
    actual_max = np.max(np.abs(actual)) if len(actual) > 0 else 1.0
    capacity = actual_max if actual_max > 0 else 1.0
    nmae = (mae / capacity) * 100
    
    metrics = {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'NMAE': round(nmae, 4),
        'MAPE': round(mape, 4),
        'R2': round(r2, 4),
    }
    
    if print_results:
        print(f"\n{'─'*40}")
        print(f"  {model_name} Evaluation Metrics")
        print(f"{'─'*40}")
        for k, v in metrics.items():
            print(f"  {k:>6s}: {v}")
    
    return metrics


def compute_all_metrics(results_dict, print_results=True):
    """
    Compute metrics for all models.
    results_dict: {model_name: {test_pred, test_actual, ...}}
    Returns: dict of {model_name: metrics_dict}
    """
    all_metrics = {}
    for name, data in results_dict.items():
        metrics = compute_metrics(
            data['test_actual'], data['test_pred'],
            print_results=print_results, model_name=name
        )
        all_metrics[name] = metrics
    
    return all_metrics


def metrics_to_table(all_metrics):
    """Convert metrics dict to a formatted comparison table."""
    import pandas as pd
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {'Model': model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')
