"""
Multi-Digital Twin Fusion Methods
===================================
Method 1: Single Metric Dynamic Preference (based on time window)
Method 2: Multi-Metrics Dynamic Fusion (DS Evidence Theory, based on time window)

Window size = 10 always.
"""

import numpy as np
from itertools import combinations
from evaluate import compute_metrics
from DS import BaseMassFunction


# ─────────────── Helper: errors ───────────────

def _calc_errors(pred, actual):
    """Quick error calculation for a window segment."""
    mae = np.mean(np.abs(pred - actual))
    mse = np.mean((pred - actual) ** 2)
    rmse = np.sqrt(mse)
    
    # R2
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return mae, rmse, r2


# ─────────────── BPA functions for DS ───────────────

from sklearn.preprocessing import MinMaxScaler

def bpa_rmse_mae(values):
    """Basic probability assignment for RMSE/MAE using MinMaxScaler (smaller is better)."""
    values = np.array(values, dtype=float).reshape(-1, 1)
    if values.max() == values.min():
        return np.ones(len(values)) / len(values)
    
    # Smaller is better, so we invert by negating before scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(-values).flatten()
    scaled += 1e-6
    
    if scaled.sum() == 0:
        return np.ones(len(values)) / len(values)
    return scaled / scaled.sum()


def bpa_r2(values):
    """Basic probability assignment for R² using MinMaxScaler (larger is better)."""
    values = np.array(values, dtype=float).reshape(-1, 1)
    if values.max() == values.min():
        return np.ones(len(values)) / len(values)
        
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    scaled += 1e-6
    
    if scaled.sum() == 0:
        return np.ones(len(values)) / len(values)
    return scaled / scaled.sum()


# ─────────────── Method 1: Single Metric Dynamic Preference ───────────────

def method1_single_metric_preference(
    predictions_dict,   # {model_name: pred_array}
    actuals,            # actual values array (same for all models)
    window_size=10
):
    """
    Single metric dynamic preference method based on time window.
    
    For each point, look at the past `window_size` points, compute RMSE
    for each model in that window, and select the model with lowest RMSE
    to produce the prediction for the next point.
    
    Returns: (fused_predictions, fused_actuals, selection_indices)
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    n_points = len(actuals)
    
    fused_preds = []
    fused_actuals = []
    selections = []
    
    for i in range(n_points - window_size - 1):
        rmse_list = []
        for name in model_names:
            pred_window = predictions_dict[name][i:i + window_size]
            actual_window = actuals[i:i + window_size]
            _, rmse, _ = _calc_errors(pred_window, actual_window)
            rmse_list.append(rmse)
        
        best_idx = np.argmin(rmse_list)
        next_point = i + window_size
        
        fused_preds.append(predictions_dict[model_names[best_idx]][next_point])
        fused_actuals.append(actuals[next_point])
        selections.append(model_names[best_idx])
    
    return np.array(fused_preds), np.array(fused_actuals), selections


# ─────────────── Method 2: Multi-Metrics DS Fusion ───────────────

def method2_ds_fusion(
    predictions_dict,
    actuals,
    window_size=10,
    zeta=0.03
):
    """
    Multi-metrics dynamic fusion method based on time window using DS Evidence Theory.
    
    For each point, compute BPAs from RMSE, MAE, and R² metrics,
    fuse them using Dempster's rule, then produce a weighted prediction.
    
    If variance of fused weights > zeta, use winner-take-all instead.
    
    Returns: (fused_predictions, fused_actuals)
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    n_points = len(actuals)
    
    # Assign codes to models
    codes = {name: str(i + 1) for i, name in enumerate(model_names)}
    
    fused_preds = []
    fused_actuals = []
    
    for i in range(n_points - window_size - 1):
        mae_list = []
        rmse_list = []
        r2_list = []
        
        for name in model_names:
            pred_window = predictions_dict[name][i:i + window_size]
            actual_window = actuals[i:i + window_size]
            mae, rmse, r2 = _calc_errors(pred_window, actual_window)
            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)
        
        # Compute BPAs
        bpa_rmse = bpa_rmse_mae(rmse_list)
        bpa_mae = bpa_rmse_mae(mae_list)
        bpa_r2_vals = bpa_r2(r2_list)
        
        # Create mass functions for DS combination
        rmse_mass = {}
        mae_mass = {}
        r2_mass = {}
        for j, name in enumerate(model_names):
            code = codes[name]
            rmse_mass[code] = bpa_rmse[j]
            mae_mass[code] = bpa_mae[j]
            r2_mass[code] = bpa_r2_vals[j]
        
        m_rmse = BaseMassFunction(rmse_mass)
        m_mae = BaseMassFunction(mae_mass)
        m_r2 = BaseMassFunction(r2_mass)
        
        # DS combination
        ds_result = m_rmse & m_mae & m_r2
        
        # Get fused weights
        ds_weights = {}
        ds_values = []
        for name in model_names:
            w = ds_result[codes[name]]
            ds_weights[name] = w
            ds_values.append(w)
        
        # Check variance threshold
        ds_var = np.var(ds_values)
        if ds_var > zeta:
            # Winner-take-all
            best_name = max(ds_weights, key=ds_weights.get)
            for name in model_names:
                ds_weights[name] = 1.0 if name == best_name else 0.0
        
        # Weighted prediction
        next_point = i + window_size
        result = sum(
            predictions_dict[name][next_point] * ds_weights[name]
            for name in model_names
        )
        
        fused_preds.append(result)
        fused_actuals.append(actuals[next_point])
    
    return np.array(fused_preds), np.array(fused_actuals)


# ─────────────── Run All Combinations ───────────────

def run_all_fusion(
    predictions_dict,
    actuals,
    window_size=10,
    zeta=0.03,
    print_results=True
):
    """
    Run both fusion methods for all combinations of 2, 3, and 4 digital twins.
    
    Returns: {
        'method1': {combo_name: metrics},
        'method2': {combo_name: metrics}
    }
    """
    model_names = list(predictions_dict.keys())
    
    all_results = {'method1': {}, 'method2': {}}
    
    for r in [2, 3, 4]:
        for combo in combinations(model_names, r):
            combo_name = '&'.join(combo)
            combo_preds = {name: predictions_dict[name] for name in combo}
            
            if print_results:
                print(f"\n{'='*50}")
                print(f"  Combination: {combo_name}")
                print(f"{'='*50}")
            
            # Method 1
            m1_pred, m1_actual, _ = method1_single_metric_preference(
                combo_preds, actuals, window_size
            )
            m1_metrics = compute_metrics(m1_actual, m1_pred, print_results, 
                                         f"Method 1 ({combo_name})")
            all_results['method1'][combo_name] = m1_metrics
            
            # Method 2
            m2_pred, m2_actual = method2_ds_fusion(
                combo_preds, actuals, window_size, zeta
            )
            m2_metrics = compute_metrics(m2_actual, m2_pred, print_results,
                                         f"Method 2 ({combo_name})")
            all_results['method2'][combo_name] = m2_metrics
    
    return all_results
