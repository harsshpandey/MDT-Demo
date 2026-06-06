"""
Side-by-side comparison: hourly vs 15-min vs paper Liu et al. 2024.

Reads prediction CSVs from `predictions/` (hourly) and `predictions_min15/`
(15-minute resampled), runs them through `mdt_engine` to compute single-DT
metrics AND both MDT fusion methods, then assembles a markdown + JSON
report ready for the dashboard.

Why this artifact exists
------------------------
A single number is meaningless without its denominator. Paper Liu et al.
report "MAE 2.39" — but that's % of a 129 MW farm output at 15-min
intervals. We must surface the **granularity** and **scale** alongside
the metric so reviewers can compare apples to apples.

Run:
  make compare              # default paths
  python compare.py --out results/comparison.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from mdt_engine import (
    compute_errors,
    get_single_dt_metrics,
    method1_single_metric,
    method2_multimetric_ds,
)

MODELS = ["LSTM", "GRU", "LSTMCNN", "GRUCNN"]


# Paper-reported best numbers (Liu et al. 2024, Tables 4-6, 2017 dataset).
# Source: Energy 292 (2024) 130269.
# Note: paper Table 4-6 reports MAE/RMSE/R² only; MAPE not reported, so paper
# rows show MAPE = None (rendered as "—" in tables).
PAPER = {
    "data":            "Ningxia, China — 129.1 MW wind farm",
    "granularity_min": 15,
    "n_features":      9,
    "single_best": {
        "name": "GRUCNN", "MAE": 2.5423, "RMSE": 4.5663, "R2": 0.8895, "MAPE": None,
    },
    "method1_best": {
        "combo": "LSTM&GRU&LSTMCNN", "MAE": 2.4353, "RMSE": 4.3569, "R2": 0.8995, "MAPE": None,
    },
    "method2_best": {
        "combo": "LSTM&GRU&LSTMCNN&GRUCNN", "MAE": 2.3874, "RMSE": 4.3298, "R2": 0.9008, "MAPE": None,
    },
}


def load_twins(preds_dir: Path) -> Dict[str, pd.DataFrame]:
    twins = {}
    for name in MODELS:
        path = preds_dir / f"{name}_preds.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        df.columns = ["idx", "predicted", "actual"]
        twins[name] = df
    return twins


def compute_block(label: str, granularity_min: int, twins: Dict[str, pd.DataFrame]) -> Dict:
    """Return dict with single + fusion metrics for one twin set."""
    single = get_single_dt_metrics(twins)
    best_single_name = min(single, key=lambda n: single[n]["mae"])

    method1_rows = []
    method2_rows = []
    for r in range(2, len(twins) + 1):
        for combo in itertools.combinations(twins.keys(), r):
            sub = {k: twins[k] for k in combo}
            m1 = method1_single_metric(sub, window=10)["metrics"]
            m2 = method2_multimetric_ds(sub, window=10, zeta=0.04)["metrics"]
            method1_rows.append({"combo": "&".join(combo), **m1})
            method2_rows.append({"combo": "&".join(combo), **m2})

    m1_best = min(method1_rows, key=lambda r: r["mae"])
    m2_best = min(method2_rows, key=lambda r: r["mae"])

    return {
        "label":            label,
        "granularity_min":  granularity_min,
        "n_test":           int(len(next(iter(twins.values())))),
        "single":           single,
        "best_single_name": best_single_name,
        "best_single":      single[best_single_name],
        "method1_best":     m1_best,
        "method2_best":     m2_best,
    }


def to_markdown(blocks: List[Dict]) -> str:
    lines = [
        "# MDT — Hourly vs 15-min vs Paper Comparison",
        "",
        "All metrics are on the % of rated capacity scale "
        "(predicted × 100, actual × 100), matching Liu et al. 2024 convention.",
        "",
        "## Single-twin best result",
        "",
        "| Run | Granularity | Best DT | MAE | RMSE | R² | MAPE |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in blocks:
        s = b["best_single"]
        lines.append(
            f"| {b['label']} | {b['granularity_min']} min | {b['best_single_name']} | "
            f"{s['mae']:.3f} | {s['rmse']:.3f} | {s['r2']:.4f} | {s['mape']:.3f} |"
        )
    # Paper row — MAPE not reported in paper
    p = PAPER["single_best"]
    lines.append(
        f"| Paper (Liu 2024) | {PAPER['granularity_min']} min | {p['name']} | "
        f"{p['MAE']:.3f} | {p['RMSE']:.3f} | {p['R2']:.4f} | — (n/a) |"
    )

    lines += [
        "",
        "## Method 1 (sliding RMSE) — best fusion",
        "",
        "| Run | Combo | MAE | RMSE | R² | MAPE |",
        "|---|---|---|---|---|---|",
    ]
    for b in blocks:
        m = b["method1_best"]
        lines.append(
            f"| {b['label']} | {m['combo']} | "
            f"{m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.4f} | {m.get('mape', 0):.3f} |"
        )
    p = PAPER["method1_best"]
    lines.append(
        f"| Paper (Liu 2024) | {p['combo']} | {p['MAE']:.3f} | {p['RMSE']:.3f} | {p['R2']:.4f} | — (n/a) |"
    )

    lines += [
        "",
        "## Method 2 (Dempster-Shafer multi-metric) — best fusion",
        "",
        "| Run | Combo | MAE | RMSE | R² | MAPE |",
        "|---|---|---|---|---|---|",
    ]
    for b in blocks:
        m = b["method2_best"]
        lines.append(
            f"| {b['label']} | {m['combo']} | "
            f"{m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.4f} | {m.get('mape', 0):.3f} |"
        )
    p = PAPER["method2_best"]
    lines.append(
        f"| Paper (Liu 2024) | {p['combo']} | {p['MAE']:.3f} | {p['RMSE']:.3f} | {p['R2']:.4f} | — (n/a) |"
    )

    lines += [
        "",
        "## Interpretation",
        "",
        "1. **MAE is a percentage of installed capacity.** Paper used 129.1 MW;",
        "   we use a single 5 MW NREL turbine. Smaller denominator = relative",
        "   error inflates even when absolute kW error is small.",
        "2. **Higher autocorrelation at 15-min** = easier next-step problem.",
        "   Liu et al. 2024 effectively predict `y(t+15m) ≈ y(t) + δ`.",
        "3. **R² is unitless** and the cleanest direct comparison.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourly-dir", default="predictions")
    parser.add_argument("--min15-dir",  default="predictions_min15")
    parser.add_argument("--out",        default="results/comparison.md")
    parser.add_argument("--json-out",   default="results/comparison.json")
    args = parser.parse_args()

    blocks: List[Dict] = []
    hourly = load_twins(Path(args.hourly_dir))
    if hourly:
        blocks.append(compute_block("Hourly", 60, hourly))
    else:
        print(f"WARN: no hourly predictions at {args.hourly_dir}")

    min15 = load_twins(Path(args.min15_dir))
    if min15:
        blocks.append(compute_block("15-min", 15, min15))
    else:
        print(f"WARN: no 15-min predictions at {args.min15_dir}")

    if not blocks:
        raise SystemExit("No predictions found — train first.")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    Path(args.out).write_text(to_markdown(blocks))
    Path(args.json_out).write_text(json.dumps({"runs": blocks, "paper": PAPER}, indent=2))
    print(f"Wrote {args.out}  +  {args.json_out}\n")
    # Brief stdout summary
    for b in blocks:
        s = b["best_single"]; m2 = b["method2_best"]
        print(f"  {b['label']:7s} ({b['granularity_min']:>2d}min):  "
              f"best single {b['best_single_name']} MAE={s['mae']:.2f} R²={s['r2']:.4f}  |  "
              f"best M2 ({m2['combo']}) MAE={m2['mae']:.2f} R²={m2['r2']:.4f}")
    p1, p2 = PAPER["single_best"], PAPER["method2_best"]
    print(f"  Paper   ({PAPER['granularity_min']:>2d}min):  "
          f"best single {p1['name']} MAE={p1['MAE']:.2f} R²={p1['R2']:.4f}  |  "
          f"best M2 ({p2['combo']}) MAE={p2['MAE']:.2f} R²={p2['R2']:.4f}")


if __name__ == "__main__":
    main()
