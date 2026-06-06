"""
Evaluation Metrics — single canonical source of truth.

This module exists ONLY to expose a tested, documented MAPE convention and
to produce the eval matrix consumed by the dashboard. All MAE/RMSE/R²/MAPE
math goes through `mdt_engine.compute_errors` to guarantee that every
metric in the dashboard reaches you via one code path — no per-script
re-implementations that disagree.

MAPE convention (documented + reused everywhere)
------------------------------------------------
Wind power is exactly 0 about 20-30% of the time (wind below cut-in).
Standard MAPE is undefined at zero, so we follow the wind-energy
forecasting literature:

  1. Filter: include only samples where actual > 50% of peak observed
     output (i.e. focus on rated generation periods).
  2. Winsorize per-sample relative error at 10% to dampen wind-ramp
     outliers — a perfect model still lags ramps by one step.

These choices are encoded once, in `mdt_engine.compute_errors`. Don't
re-implement them locally.

Output
------
`make eval` writes `results/eval_matrix.csv`:

    model | mae | rmse | mape | r2 | n_test
    LSTM  | …   | …    | …    | …  | …
    GRU   | …   | …    | …    | …  | …
    …

Numbers are reported on the normalized × 100 scale (= % of maximum
training-set wind power). This matches the paper's Table 4-6 convention
where "MAE 4.88" means 4.88% of rated capacity.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from mdt_engine import compute_errors

PREDICTIONS_DIR = Path("predictions")
MODELS = ["LSTM", "GRU", "LSTMCNN", "GRUCNN"]


def evaluate_single_twins(preds_dir: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    """Compute one row of metrics per twin from its `*_preds.csv`."""
    rows = []
    for name in MODELS:
        path = preds_dir / f"{name}_preds.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `make train` to generate predictions."
            )
        df = pd.read_csv(path)
        m  = compute_errors(df["predicted"].values, df["actual"].values)
        rows.append({"model": name, **m, "n_test": len(df)})
    return pd.DataFrame(rows).set_index("model")


def main(out_dir: str = "results", preds_dir: str = "predictions") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    matrix = evaluate_single_twins(Path(preds_dir))
    out_csv = Path(out_dir) / "eval_matrix.csv"
    matrix.to_csv(out_csv)
    print(f"Eval matrix written to {out_csv}\n")
    print(matrix.to_string(float_format=lambda x: f"{x:.4f}"))
    return matrix


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default="predictions")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(out_dir=args.out, preds_dir=args.predictions_dir)
