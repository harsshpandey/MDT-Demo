"""
Verify MDT fusion math matches Liu et al. 2024 (Energy 292, 130269).

Paper equations covered:
  Eq (13): BPA for RMSE/MAE  →  Y_i = 1/M_i,  m_i = Y_i / Σ Y_j
  Eq (14): BPA for R²        →  m_i = M_i / Σ M_j
  Eq (15): Dempster combination rule (conjunctive, singleton hypotheses)
  Eq (16): variance threshold adjustment (winner-take-all when δ² > ζ)
  Eq (17): final weighted prediction

These tests use small toy inputs with known closed-form answers — no model needed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mdt_engine import (
    _bpa_rmse_mae,
    _bpa_r2,
    _ds_combine_two,
    _ds_combine_list,
    method1_single_metric,
    method2_multimetric_ds,
)


# ─────────────────────────── Eq 13: BPA for RMSE / MAE ───────────────────────────


def test_bpa_rmse_reciprocal_normalization():
    """Eq 13: m_i ∝ 1/M_i, sum to 1, smallest metric → largest mass."""
    rmse = [0.1, 0.2, 0.4, 0.5]
    bpa = _bpa_rmse_mae(rmse)
    # Sum normalized
    assert math.isclose(sum(bpa), 1.0, abs_tol=1e-9)
    # Order preserved: smallest RMSE → largest mass
    assert bpa[0] > bpa[1] > bpa[2] > bpa[3]
    # Exact value: 1/0.1 / (1/0.1 + 1/0.2 + 1/0.4 + 1/0.5) = 10 / 19.5
    expected_0 = (1 / 0.1) / (1 / 0.1 + 1 / 0.2 + 1 / 0.4 + 1 / 0.5)
    assert math.isclose(bpa[0], expected_0, rel_tol=1e-6)


def test_bpa_rmse_all_equal_returns_uniform():
    """If all twins tie, BPA is uniform — no twin should be favored."""
    bpa = _bpa_rmse_mae([0.3, 0.3, 0.3, 0.3])
    assert all(math.isclose(b, 0.25, abs_tol=1e-6) for b in bpa)


def test_bpa_rmse_zero_safe():
    """Zero metric (perfect window) must not produce NaN / Inf."""
    bpa = _bpa_rmse_mae([0.0, 0.1, 0.2, 0.3])
    assert all(math.isfinite(b) for b in bpa)
    assert math.isclose(sum(bpa), 1.0, abs_tol=1e-6)


# ─────────────────────────── Eq 14: BPA for R² ───────────────────────────


def test_bpa_r2_direct_normalization():
    """Eq 14: m_i ∝ M_i (larger R² → larger mass), sum to 1."""
    r2 = [0.9, 0.8, 0.7, 0.6]
    bpa = _bpa_r2(r2)
    assert math.isclose(sum(bpa), 1.0, abs_tol=1e-9)
    assert bpa[0] > bpa[1] > bpa[2] > bpa[3]
    # All positive, so shifting by min(0, min)=0 is a no-op
    assert math.isclose(bpa[0], 0.9 / sum(r2), rel_tol=1e-6)


def test_bpa_r2_negative_handled():
    """
    Paper assumes R² ≥ 0. On noisy windows R² can go negative.
    Implementation shifts by min(0, min(R²)) so smallest maps to 0.
    Order must still be preserved.
    """
    r2 = [-0.2, 0.3, 0.5, 0.8]
    bpa = _bpa_r2(r2)
    assert math.isclose(sum(bpa), 1.0, abs_tol=1e-9)
    # Order preserved (smallest → 0 mass, largest → most mass)
    assert bpa[0] < bpa[1] < bpa[2] < bpa[3]
    # The most-negative entry gets mass 0 after shift
    assert math.isclose(bpa[0], 0.0, abs_tol=1e-9)


# ─────────────────────────── Eq 15: Dempster combination ───────────────────────────


def test_ds_combine_singletons_no_conflict():
    """Two identical singleton mass functions → product, renormalize."""
    m1 = {"A": 0.6, "B": 0.4}
    m2 = {"A": 0.6, "B": 0.4}
    out = _ds_combine_two(m1, m2)
    # Singleton intersection: A∩A = A, B∩B = B. A∩B = ∅ (conflict).
    # K = 0.6*0.4 + 0.4*0.6 = 0.48
    # m(A) = 0.6*0.6 / (1-0.48) = 0.36 / 0.52
    # m(B) = 0.4*0.4 / (1-0.48) = 0.16 / 0.52
    assert math.isclose(out["A"], 0.36 / 0.52, rel_tol=1e-6)
    assert math.isclose(out["B"], 0.16 / 0.52, rel_tol=1e-6)
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)


def test_ds_combine_chain_associative():
    """Chaining 3 mass functions = combining sequentially."""
    m1 = {"X": 0.5, "Y": 0.5}
    m2 = {"X": 0.7, "Y": 0.3}
    m3 = {"X": 0.6, "Y": 0.4}
    chained = _ds_combine_list([m1, m2, m3])
    pair = _ds_combine_two(_ds_combine_two(m1, m2), m3)
    for k in chained:
        assert math.isclose(chained[k], pair[k], rel_tol=1e-9)


# ─────────────────────────── Eq 16: variance threshold ───────────────────────────


def test_method2_winner_take_all_above_threshold():
    """
    Build a synthetic case where one twin dominates: window where LSTM is
    perfect and others are very noisy. After fusion δ² > ζ → winner-take-all.
    Expected: MDT prediction = LSTM prediction at all steps post-warmup.
    """
    n = 60
    rng = np.random.default_rng(0)
    actual = np.sin(np.linspace(0, 6 * np.pi, n))
    # LSTM: perfect. Others: actual + heavy noise.
    twins = {
        "LSTM":    pd.DataFrame({"idx": range(n), "predicted": actual,                   "actual": actual}),
        "GRU":     pd.DataFrame({"idx": range(n), "predicted": actual + rng.normal(0, 1, n), "actual": actual}),
        "LSTMCNN": pd.DataFrame({"idx": range(n), "predicted": actual + rng.normal(0, 1, n), "actual": actual}),
        "GRUCNN":  pd.DataFrame({"idx": range(n), "predicted": actual + rng.normal(0, 1, n), "actual": actual}),
    }
    result = method2_multimetric_ds(twins, window=10, zeta=0.04)
    weights = result["weights"]
    # LSTM should dominate the vast majority of post-warmup steps
    lstm_wins = sum(1 for w in weights if w["LSTM"] == 1.0)
    assert lstm_wins / len(weights) > 0.85, (
        f"LSTM should dominate when it is the only accurate twin; got {lstm_wins}/{len(weights)}"
    )


# ─────────────────────────── Eq 17: final weighted prediction ───────────────────────────


def test_method1_picks_best_rmse_per_window():
    """Method 1 must pick the twin with lowest sliding-window RMSE."""
    n = 30
    actual = np.linspace(0, 1, n)
    # Two twins: GRU perfect everywhere; LSTM very wrong.
    twins = {
        "LSTM": pd.DataFrame({"idx": range(n), "predicted": np.zeros(n),   "actual": actual}),
        "GRU":  pd.DataFrame({"idx": range(n), "predicted": actual,        "actual": actual}),
    }
    result = method1_single_metric(twins, window=5)
    # All winners should be GRU once the window populates
    assert result["winners"].count("GRU") == len(result["winners"])
    # MDT prediction tracks GRU's prediction = actual
    assert np.allclose(result["predictions"], result["actual"], atol=1e-9)
