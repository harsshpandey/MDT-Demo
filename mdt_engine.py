"""
MDT Engine - Multi-Digital Twin Computation Backend
Implements both MDT methods from the research paper:
  Method 1: Single Metric Dynamic Preference
  Method 2: Multi-Metrics Dynamic Fusion (Dempster-Shafer)

Adapted from mdt_enginedemo.py to work with Indian dataset predictions.
"""

import os
import numpy as np
import pandas as pd
from math import fsum

# ── Data loading points to predictions/ directory ──
PRED_DIR = os.path.join(os.path.dirname(__file__), "predictions")


def load_twin_data():
    """Load prediction CSVs for all 4 digital twins from predictions/ dir."""
    twins = {}
    for name in ["LSTM", "GRU", "LSTMCNN", "GRUCNN"]:
        path = os.path.join(PRED_DIR, f"{name}_preds.csv")
        df = pd.read_csv(path, encoding="utf-8")
        df.columns = ["idx", "predicted", "actual"]
        twins[name] = df
    return twins


def compute_errors(predicted, actual):
    """Compute MAE, MSE, RMSE, R² for arrays."""
    predicted = np.array(predicted, dtype=float)
    actual = np.array(actual, dtype=float)
    n = len(actual)
    if n == 0:
        return dict(mae=0, mse=0, rmse=0, r2=0)
    mae = float(np.mean(np.abs(predicted - actual)))
    mse = float(np.mean((predicted - actual) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    return dict(mae=mae, mse=mse, rmse=rmse, r2=r2)


from sklearn.preprocessing import MinMaxScaler

def _bpa_rmse_mae(values):
    """Basic Probability Assignment for RMSE/MAE using MinMaxScaler (smaller is better)."""
    values = np.array(values, dtype=float).reshape(-1, 1)
    if values.max() == values.min():
        return list(np.ones(len(values)) / len(values))
    
    # Smaller is better, so we invert by negating before scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(-values).flatten()
    scaled += 1e-6
    
    if scaled.sum() == 0:
        return list(np.ones(len(values)) / len(values))
    return list(scaled / scaled.sum())


def _bpa_r2(values):
    """Basic Probability Assignment for R² using MinMaxScaler (larger is better)."""
    values = np.array(values, dtype=float).reshape(-1, 1)
    if values.max() == values.min():
        return list(np.ones(len(values)) / len(values))
        
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    scaled += 1e-6
    
    if scaled.sum() == 0:
        return list(np.ones(len(values)) / len(values))
    return list(scaled / scaled.sum())


def _ds_combine_two(m1, m2):
    """Dempster-Shafer conjunctive combination of two single-element mass functions."""
    combined = {}
    K = 0.0
    for h1, v1 in m1.items():
        for h2, v2 in m2.items():
            inter = h1 if h1 == h2 else None
            if inter is None:
                K += v1 * v2
            else:
                combined[inter] = combined.get(inter, 0.0) + v1 * v2
    denom = 1 - K
    if denom <= 0:
        return {k: 1.0 / len(combined) for k in combined} if combined else m1
    return {k: v / denom for k, v in combined.items()}


def _ds_combine_list(mass_functions):
    """Combine a list of mass function dicts using D-S rule iteratively."""
    result = mass_functions[0]
    for m in mass_functions[1:]:
        result = _ds_combine_two(result, m)
    return result


def method1_single_metric(twins, window=10):
    """
    Single Metric Dynamic Preference Method.
    For each time step i (after the initial window), select the DT with
    minimum RMSE over [i-window, i] and use its predicted value for step i+1.

    Returns:
        predictions: list of predicted values for MDT
        actual: corresponding actual values
        winners: list of DT name chosen at each step
        metrics: overall metrics dict
    """
    names = list(twins.keys())
    n = len(twins[names[0]])
    predictions = []
    actual_vals = []
    winners = []

    for i in range(n - window - 1):
        rmse_per_dt = []
        for name in names:
            df = twins[name]
            pred_w = df["predicted"].iloc[i: i + window].values
            act_w = df["actual"].iloc[i: i + window].values
            err = compute_errors(pred_w, act_w)
            rmse_per_dt.append(err["rmse"])

        best_idx = int(np.argmin(rmse_per_dt))
        best_name = names[best_idx]
        next_pred = float(twins[best_name]["predicted"].iloc[i + window])
        next_actual = float(twins[best_name]["actual"].iloc[i + window])
        predictions.append(next_pred)
        actual_vals.append(next_actual)
        winners.append(best_name)

    metrics = compute_errors(predictions, actual_vals)
    return dict(predictions=predictions, actual=actual_vals, winners=winners, metrics=metrics)


def method2_multimetric_ds(twins, window=10, zeta=0.04):
    """
    Multi-Metrics Dynamic Fusion Method (D-S Evidence Theory).
    For each time step, compute BPA from RMSE, MAE, R², fuse with D-S rule,
    then apply variance threshold to switch between preference and weighted fusion.

    Returns:
        predictions, actual, weights_per_step, metrics
    """
    names = list(twins.keys())
    n = len(twins[names[0]])
    predictions = []
    actual_vals = []
    weights_per_step = []

    for i in range(n - window - 1):
        rmse_list, mae_list, r2_list = [], [], []
        for name in names:
            df = twins[name]
            pred_w = df["predicted"].iloc[i: i + window].values
            act_w = df["actual"].iloc[i: i + window].values
            err = compute_errors(pred_w, act_w)
            rmse_list.append(err["rmse"])
            mae_list.append(err["mae"])
            r2_list.append(err["r2"])

        bpa_rmse = _bpa_rmse_mae(rmse_list)
        bpa_mae = _bpa_rmse_mae(mae_list)
        bpa_r2 = _bpa_r2(r2_list)

        mf_rmse = {names[j]: bpa_rmse[j] for j in range(len(names))}
        mf_mae  = {names[j]: bpa_mae[j]  for j in range(len(names))}
        mf_r2   = {names[j]: bpa_r2[j]   for j in range(len(names))}

        fused = _ds_combine_list([mf_rmse, mf_mae, mf_r2])

        fused_vals = list(fused.values())
        variance = float(np.var(fused_vals))

        if variance > zeta:
            best_key = max(fused, key=fused.get)
            adjusted = {k: (1.0 if k == best_key else 0.0) for k in fused}
        else:
            adjusted = fused

        result_pred = 0.0
        for name in names:
            result_pred += float(twins[name]["predicted"].iloc[i + window]) * adjusted.get(name, 0.0)

        next_actual = float(twins[names[0]]["actual"].iloc[i + window])
        predictions.append(result_pred)
        actual_vals.append(next_actual)
        weights_per_step.append({k: round(adjusted.get(k, 0.0), 4) for k in names})

    metrics = compute_errors(predictions, actual_vals)
    return dict(predictions=predictions, actual=actual_vals, weights=weights_per_step, metrics=metrics)


def get_single_dt_metrics(twins):
    """Return MAE, RMSE, R² for each single DT."""
    result = {}
    for name, df in twins.items():
        result[name] = compute_errors(df["predicted"].values, df["actual"].values)
    return result
