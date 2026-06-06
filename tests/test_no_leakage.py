"""
Integrity tests — fail if predictions are blended with ground truth.

Catches the historical adjust.py issue where pred_adj = ALPHA*pred + (1-ALPHA)*actual
inflated metrics by leaking 45% of the answer into every prediction.

Also fails if any script in the repo reintroduces such a blend pattern.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = REPO_ROOT / "predictions"

# Pattern: any literal blend of pred and actual with a fractional constant.
# Examples it catches:
#   ALPHA * pred + (1 - ALPHA) * actual
#   0.5 * predicted + 0.5 * actual
#   a*pred + b*y_true (heuristic, simple form)
BLEND_PATTERN = re.compile(
    r"""
    (?:\b\w+\s*\*\s*pred\w*\s*\+\s*\(?\s*1\s*-\s*\w+\s*\)?\s*\*\s*actual)   # ALPHA*pred + (1-ALPHA)*actual
    |
    (?:\b\d*\.\d+\s*\*\s*pred\w*\s*\+\s*\d*\.\d+\s*\*\s*actual)             # 0.5*pred + 0.5*actual
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _python_files():
    skip_dirs = {".venv", "__pycache__", ".git", "tests", "data", "predictions", "results"}
    for p in REPO_ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        yield p


def test_no_pred_actual_blend_in_source():
    """No source file blends predictions with ground truth."""
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), 1):
            # Strip comments to avoid false positives on docstrings warning about leakage
            code = line.split("#", 1)[0]
            if BLEND_PATTERN.search(code):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")

    assert not offenders, (
        "Forbidden pred/actual blend detected — this is leakage. Offending lines:\n  "
        + "\n  ".join(offenders)
    )


def test_no_adjust_or_reevaluate_script():
    """The historical leakage scripts must stay deleted."""
    forbidden = ["adjust.py", "reevaluate.py"]
    present = [f for f in forbidden if (REPO_ROOT / f).exists()]
    assert not present, (
        f"Removed scripts have reappeared: {present}. "
        "These applied an ALPHA blend that contaminated reported metrics. "
        "Delete them before committing."
    )


@pytest.mark.skipif(
    not PREDICTIONS_DIR.exists() or not any(PREDICTIONS_DIR.glob("*_preds.csv")),
    reason="No prediction CSVs yet — run training first",
)
def test_prediction_correlation_below_blend_threshold():
    """
    A 55%-pred / 45%-actual blend would push corr(pred, actual) very close to 1.
    Honest model on hourly Indian wind data → corr ≈ 0.95-0.99 with lag features,
    ≈ 0.85-0.95 without. A corr > 0.999 across all twins is a strong leakage signal.
    """
    suspect = []
    for csv in PREDICTIONS_DIR.glob("*_preds.csv"):
        df = pd.read_csv(csv)
        if "predicted" not in df.columns or "actual" not in df.columns:
            continue
        corr = float(np.corrcoef(df["predicted"], df["actual"])[0, 1])
        if corr > 0.999:
            suspect.append(f"{csv.name}: corr={corr:.6f}")
    assert not suspect, (
        "Suspiciously high correlation suggests blend leakage:\n  " + "\n  ".join(suspect)
    )
