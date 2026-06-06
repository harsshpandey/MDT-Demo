"""
Flask API server — Multi-Digital Twin Wind Power Forecasting Dashboard.

Honest rebuild:
  • Predictions loaded from predictions/ — if missing, dashboard shows a
    pipeline-setup screen instead of crashing.
  • combination_results.csv generated lazily on first /api/combinations hit
    so cold start is fast.
  • No ALPHA blend, no calibration — single source of truth in mdt_engine.py.
"""

from __future__ import annotations

import itertools
import os
from typing import Dict

import pandas as pd
from flask import Flask, jsonify, render_template, request

from mdt_engine import (
    compute_errors,
    get_single_dt_metrics,
    method1_single_metric,
    method2_multimetric_ds,
)

REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
PRED_DIR   = os.path.join(REPO_DIR, "predictions")
COMBO_CSV  = os.path.join(REPO_DIR, "combination_results.csv")
MODEL_KEYS = ["LSTM", "GRU", "LSTMCNN", "GRUCNN"]

app = Flask(__name__)


# ── Lazy state ───────────────────────────────────────────────────────────────

_state: Dict[str, object] = {"twins": None, "loaded": False, "error": None}


def load_twins() -> Dict[str, pd.DataFrame]:
    """Read 4 prediction CSVs. Missing → dict with error, no exception."""
    twins: Dict[str, pd.DataFrame] = {}
    missing = []
    for name in MODEL_KEYS:
        path = os.path.join(PRED_DIR, f"{name}_preds.csv")
        if not os.path.exists(path):
            missing.append(os.path.relpath(path, REPO_DIR))
            continue
        df = pd.read_csv(path)
        df.columns = ["idx", "predicted", "actual"]
        twins[name] = df
    if missing:
        _state["error"] = (
            "Predictions not generated yet. Run the pipeline:\n"
            "  make data && make train\n"
            "Missing files: " + ", ".join(missing)
        )
        return {}
    _state["error"] = None
    return twins


def get_twins() -> Dict[str, pd.DataFrame]:
    if not _state["loaded"]:
        _state["twins"] = load_twins()
        _state["loaded"] = True
    return _state["twins"] or {}


def regenerate_combination_results() -> int:
    """Compute all 11 combos × 2 methods, write combination_results.csv. Returns row count."""
    twins = get_twins()
    if not twins:
        return 0
    rows = []
    models = list(twins.keys())
    for r in range(2, len(models) + 1):
        for combo in itertools.combinations(models, r):
            sub = {k: twins[k] for k in combo}
            combo_name = "&".join(combo)
            group = f"{r}-DT"
            m1 = method1_single_metric(sub, window=10)["metrics"]
            m2 = method2_multimetric_ds(sub, window=10, zeta=0.04)["metrics"]
            rows.append({"group": group, "method": "method1", "combo": combo_name, **m1})
            rows.append({"group": group, "method": "method2", "combo": combo_name, **m2})
    df = pd.DataFrame(rows).sort_values(["method", "group", "combo"]).reset_index(drop=True)
    df.to_csv(COMBO_CSV, index=False)
    return len(df)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    twins = get_twins()
    n_points = len(next(iter(twins.values()))) if twins else 0
    return render_template(
        "index.html",
        n_points=n_points,
        pipeline_error=_state.get("error"),
    )


@app.route("/api/status")
def status():
    twins = get_twins()
    return jsonify({
        "ready": bool(twins),
        "error": _state.get("error"),
        "models": list(twins.keys()),
        "n_points": len(next(iter(twins.values()))) if twins else 0,
    })


@app.route("/api/data")
def get_data():
    twins = get_twins()
    if not twins:
        return jsonify({"error": _state["error"]}), 503
    first = next(iter(twins.values()))
    return jsonify({
        "actual": first["actual"].tolist(),
        "twins": {name: df["predicted"].tolist() for name, df in twins.items()},
    })


@app.route("/api/metrics")
def get_metrics():
    twins = get_twins()
    if not twins:
        return jsonify({"error": _state["error"]}), 503
    return jsonify(get_single_dt_metrics(twins))


@app.route("/api/mdt/method1")
def mdt_method1():
    twins = get_twins()
    if not twins:
        return jsonify({"error": _state["error"]}), 503
    window = max(2, min(int(request.args.get("window", 20)), 40))
    return jsonify(method1_single_metric(twins, window=window))


@app.route("/api/mdt/method2")
def mdt_method2():
    twins = get_twins()
    if not twins:
        return jsonify({"error": _state["error"]}), 503
    window = max(2, min(int(request.args.get("window", 10)), 40))
    zeta = max(0.0, min(float(request.args.get("zeta", 0.04)), 0.2))
    return jsonify(method2_multimetric_ds(twins, window=window, zeta=zeta))


@app.route("/api/combinations")
def combinations_route():
    twins = get_twins()
    if not twins:
        return jsonify({"error": _state["error"]}), 503
    if not os.path.exists(COMBO_CSV):
        regenerate_combination_results()
    df = pd.read_csv(COMBO_CSV)
    out: Dict[str, Dict[str, list]] = {}
    for group in df["group"].unique():
        g = df[df["group"] == group]
        out[group] = {}
        for method in g["method"].unique():
            out[group][method] = g[g["method"] == method][
                ["combo", "mae", "rmse", "mape", "r2"]
            ].to_dict("records")
    return jsonify(out)


@app.route("/api/summary")
def summary():
    twins = get_twins()
    if not twins:
        return jsonify({"error": _state["error"]}), 503
    return jsonify({
        "n_points": len(next(iter(twins.values()))),
        "single_dts": get_single_dt_metrics(twins),
        "method1":    method1_single_metric(twins, window=20)["metrics"],
        "method2":    method2_multimetric_ds(twins, window=10, zeta=0.04)["metrics"],
    })


@app.route("/api/feature-doc")
def feature_doc():
    """Per-feature physical reasoning, sourced from data_pipeline.FEATURE_DOC."""
    from data_pipeline import FEATURE_DOC
    return jsonify(FEATURE_DOC)


@app.route("/api/twin-configs")
def twin_configs():
    """Diversity-injection recipe per twin (what feature subset, why)."""
    from train import TWIN_CONFIGS, select_features
    # Best-effort: load full feature list from the engineered CSV if present.
    feats: list[str] = []
    try:
        sample = pd.read_csv(os.path.join("results", "test_scaled.csv"), nrows=0)
        feats = [c for c in sample.columns if c not in ("datetime", "wind_power")]
    except Exception:
        pass
    out = []
    for cfg in TWIN_CONFIGS:
        sub = select_features(feats, cfg.feature_filter) if feats else []
        out.append({
            "name":   cfg.name,
            "arch":   cfg.arch,
            "filter": cfg.feature_filter,
            "note":   cfg.note,
            "n_features": len(sub),
            "features":   sub,
        })
    return jsonify(out)


@app.route("/api/training-summary")
def training_summary():
    """Return results/training_summary.json if training has produced one."""
    path = os.path.join(REPO_DIR, "results", "training_summary.json")
    if not os.path.exists(path):
        return jsonify({"error": "Training has not produced a summary yet."}), 404
    import json
    with open(path) as fh:
        return jsonify(json.load(fh))


@app.route("/api/glossary")
def glossary():
    """Project glossary — every acronym + concept used."""
    path = os.path.join(REPO_DIR, "results", "glossary.md")
    if not os.path.exists(path):
        return jsonify({"error": "glossary.md not generated yet."}), 404
    return app.response_class(open(path).read(), mimetype="text/markdown")


@app.route("/api/mdt-study")
def mdt_study():
    """Return the detailed MDT methods study (markdown)."""
    path = os.path.join(REPO_DIR, "results", "mdt_methods_study.md")
    if not os.path.exists(path):
        return jsonify({"error": "Run `make all` to generate."}), 404
    return app.response_class(open(path).read(), mimetype="text/markdown")


@app.route("/api/method-sweep")
def method_sweep():
    """Window length + zeta sensitivity sweep results (JSON)."""
    import json
    path = os.path.join(REPO_DIR, "results", "method_sweep.json")
    if not os.path.exists(path):
        return jsonify({"error": "Run the sweep first."}), 404
    with open(path) as fh:
        return jsonify(json.load(fh))


@app.route("/api/final-analysis")
def final_analysis():
    """Return the senior-engineer markdown summary."""
    path = os.path.join(REPO_DIR, "results", "final_analysis.md")
    if not os.path.exists(path):
        return jsonify({"error": "Run `make all && make compare` to generate."}), 404
    return app.response_class(open(path).read(), mimetype="text/markdown")


@app.route("/api/comparison")
def comparison():
    """Hourly vs 15-min vs paper, from results/comparison.json."""
    import json
    path = os.path.join(REPO_DIR, "results", "comparison.json")
    if not os.path.exists(path):
        return jsonify({"error": "Run `make compare` after training both granularities."}), 404
    with open(path) as fh:
        return jsonify(json.load(fh))


@app.route("/api/eda")
def eda_report():
    """Return EDA stats (results/eda_stats.json) — data understanding panel."""
    import json
    path = os.path.join(REPO_DIR, "results", "eda_stats.json")
    if not os.path.exists(path):
        return jsonify({"error": "Run `python eda.py --csv <path>` first."}), 404
    with open(path) as fh:
        return jsonify(json.load(fh))


@app.route("/api/loss-curves")
def loss_curves():
    """Per-twin train/val loss curves from results/predictions/*_losses.csv."""
    out: Dict[str, list] = {}
    losses_dir = os.path.join(REPO_DIR, "results", "predictions")
    if not os.path.exists(losses_dir):
        return jsonify({"error": "No loss curves yet."}), 404
    for name in MODEL_KEYS:
        p = os.path.join(losses_dir, f"{name}_losses.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            out[name] = df.to_dict("records")
    return jsonify(out)


@app.route("/api/window-sweep")
def window_sweep():
    """Replicate paper Fig 10: metrics vs window length for the 4-DT MDT."""
    twins = get_twins()
    if not twins:
        return jsonify({"error": _state["error"]}), 503
    method = request.args.get("method", "method1")
    out = []
    for w in [3, 5, 8, 10, 15, 20, 25, 30, 35, 40]:
        if method == "method2":
            m = method2_multimetric_ds(twins, window=w, zeta=0.04)["metrics"]
        else:
            m = method1_single_metric(twins, window=w)["metrics"]
        out.append({"window": w, **m})
    return jsonify(out)


if __name__ == "__main__":
    print("MDT Wind Power Forecasting — Indian Dataset Reproducibility Demo")
    print("Open http://localhost:5001")
    app.run(debug=False, port=5001, host="127.0.0.1")
