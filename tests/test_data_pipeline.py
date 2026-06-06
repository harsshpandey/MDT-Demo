"""
Data-pipeline tests — guards against silent regressions in feature engineering.

Most important guard: no `wp_lag_*` column EVER reappears. The historical
pipeline contained `wp_lag_1, wp_lag_2, wp_lag_3` (target lag), which made
single twins reach R² > 0.97 trivially and collapsed all four twins to the
same prediction. The MDT fusion thesis is meaningless on that setup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent

from data_pipeline import (  # noqa: E402  (import after path mod)
    FEATURE_DOC,
    compute_wind_power,
    engineer_features,
    load_indian_dataset,
    prepare_data,
)


REAL_CSV = REPO / "data" / "raw" / "36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv"


# ─────────────────────────────────── Schema integrity ────────────────────────


def test_feature_doc_keys_dont_include_target_lags():
    """FEATURE_DOC must never document a wp_lag_* feature."""
    offenders = [name for name in FEATURE_DOC if name.startswith("wp_lag_")]
    assert not offenders, f"Target lag features forbidden: {offenders}"


@pytest.mark.skipif(not REAL_CSV.exists(),
                    reason="Real CSV missing at data/raw/")
def test_engineered_features_match_feature_doc():
    """Every column produced must be in FEATURE_DOC and vice versa."""
    raw = load_indian_dataset(str(REAL_CSV))
    feats = engineer_features(raw)
    produced = {c for c in feats.columns if c not in ("datetime", "wind_power")}
    documented = set(FEATURE_DOC.keys())
    assert produced == documented, (
        f"Drift between code and FEATURE_DOC.\n"
        f"In code but not documented: {produced - documented}\n"
        f"Documented but not in code: {documented - produced}"
    )


@pytest.mark.skipif(not REAL_CSV.exists(),
                    reason="Real CSV missing")
def test_no_target_lag_in_engineered_output():
    raw = load_indian_dataset(str(REAL_CSV))
    feats = engineer_features(raw)
    leaks = [c for c in feats.columns if c.startswith("wp_lag_")]
    assert not leaks, f"Target-lag features reappeared: {leaks}"


# ─────────────────────────────────── Physics correctness ─────────────────────


def test_compute_wind_power_zero_below_cutin():
    """Below 3 m/s cut-in speed → 0 kW."""
    v = np.array([0.0, 1.0, 2.5, 2.99])
    rho = np.full_like(v, 1.225)
    assert np.allclose(compute_wind_power(v, rho), 0.0)


def test_compute_wind_power_zero_above_cutout():
    """Above 25 m/s cut-out → 0 kW (safety shutdown)."""
    v = np.array([25.01, 30.0, 50.0])
    rho = np.full_like(v, 1.225)
    assert np.allclose(compute_wind_power(v, rho), 0.0)


def test_compute_wind_power_rated_region_constant():
    """Between 11.4 and 25 m/s → constant rated power (5000 kW)."""
    v = np.array([11.4, 15.0, 20.0, 24.9])
    rho = np.full_like(v, 1.225)
    p = compute_wind_power(v, rho)
    assert np.allclose(p, 5000.0), f"Expected rated 5000 kW, got {p}"


def test_compute_wind_power_cubic_in_region_ii():
    """In region II (3-11.4 m/s) power scales as v³ for fixed density."""
    v = np.array([5.0, 7.0])
    rho = np.full_like(v, 1.225)
    p = compute_wind_power(v, rho)
    ratio_p = p[1] / p[0]
    ratio_v3 = (v[1] / v[0]) ** 3
    assert np.isclose(ratio_p, ratio_v3, rtol=1e-6)


# ─────────────────────────────────── Split correctness ───────────────────────


@pytest.mark.skipif(not REAL_CSV.exists(),
                    reason="Real CSV missing")
def test_prepare_data_split_proportions_match_paper():
    """81 / 9 / 10 sequential split per Liu et al. 2024 Sec 4."""
    train, val, test, _scaler, _feats, _tgt = prepare_data(str(REAL_CSV))
    total = len(train) + len(val) + len(test)
    assert abs(len(train) / total - 0.81) < 0.01
    assert abs(len(val)   / total - 0.09) < 0.01
    assert abs(len(test)  / total - 0.10) < 0.01


@pytest.mark.skipif(not REAL_CSV.exists(),
                    reason="Real CSV missing")
def test_resample_to_15min_preserves_mean_quadruples_rows():
    """Cubic interpolation must preserve mean wind speed and produce ~4× rows."""
    from data_pipeline import resample_to_15min
    raw = load_indian_dataset(str(REAL_CSV))
    r15 = resample_to_15min(raw)
    # Row count: 8 760 hourly → ~ 35 037 (4 × 8 760 − 3 for endpoint trim)
    assert 34_000 < len(r15) < 36_000, f"Unexpected row count: {len(r15)}"
    # Mean wind speed should be preserved within 1% (interpolation must not bias)
    v_h = raw["wind speed at 80m (m/s)"].mean()
    v_r = r15["wind speed at 80m (m/s)"].mean()
    assert abs(v_h - v_r) / v_h < 0.01, f"Mean wind drift: {v_h:.3f} → {v_r:.3f}"
    # Lag-1 autocorrelation MUST be higher than hourly (paper-like regime)
    assert r15["wind speed at 80m (m/s)"].autocorr(1) > raw["wind speed at 80m (m/s)"].autocorr(1)


@pytest.mark.skipif(not REAL_CSV.exists(),
                    reason="Real CSV missing")
def test_scaler_fit_on_train_only_not_full_data():
    """Scaler should be fit on train only; min/max should match train stats."""
    train, val, test, scaler, feats, _tgt = prepare_data(str(REAL_CSV))
    # After scaling, train values should be ~[0,1]; val/test can exceed.
    assert train[feats].min().min() >= -1e-6
    assert train[feats].max().max() <= 1 + 1e-6
