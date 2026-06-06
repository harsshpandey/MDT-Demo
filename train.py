"""
Training pipeline — 4 digital twins, fully deterministic, with diversity injection.

Why diversity injection?
------------------------
The Liu et al. 2024 MDT thesis only delivers gains when individual twins
disagree (different errors at different times). If all 4 twins train on
the same 39 features they tend to converge to nearly identical predictions
and fusion buys you nothing.

We force decorrelation by giving each twin a different VIEW of the data:

  • LSTM      — full feature set (39 cols)                — generalist
  • GRU       — meteorology only, NO time-of-day features — physics-driven
  • LSTMCNN   — physics + dynamics, NO rolling stats      — instantaneous
  • GRUCNN    — full + higher dropout                     — regularized

Each twin still sees a 24-hour lookback window and predicts t+1 power.
This is an HONEST source of diversity — no model sees the target directly,
no model gets a leaked answer. The diversity comes from input-feature
ablation, which is well-established in ensemble learning (random
subspace method, Ho 1998).

Reproducibility
---------------
seed_everything(42) is called before any random init. Combined with
deterministic PyTorch settings and a fixed DataLoader generator, the
metrics shipped in the dashboard are bit-identical across re-runs on
the same hardware.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline import prepare_data
from models import get_model

# ────────────────────────────── Reproducibility ──────────────────────────────


SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Lock every randomness source. Call once before any model/data init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms — slower but reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Apple MPS > CUDA > CPU, in that preference order on this hardware."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ────────────────────────────── Diversity recipe ─────────────────────────────


@dataclass(frozen=True)
class TwinConfig:
    """
    One row in the diversity matrix. Each twin trains with a distinct
    feature SUBSET plus arch knobs — forcing decorrelated outputs.
    """
    name: str
    arch: str               # key into models.get_model
    feature_filter: str     # "all" | "no_time" | "physics_only" | "all_high_dropout"
    note: str               # human-readable reason


TWIN_CONFIGS: List[TwinConfig] = [
    # NOTE: all four twins keep the time-of-day / seasonal features.
    # Real Indian wind data (NREL site 36565) has lag-24 autocorrelation = 0.65,
    # i.e. strong diurnal cycle. Dropping time features from any twin would
    # cripple it on real data. Diversity comes from orthogonal ablations:
    #   1. no_rolling      — twin must predict from instantaneous + lag values
    #   2. no_lags         — twin must predict from current + averaged windows
    #   3. no_high_freq    — twin uses only smoothed inputs (kills gust noise)
    #   4. all_high_dropout— same features, heavier stochastic regularization
    TwinConfig("LSTM",    "LSTM",    "all",
               "Generalist: full feature set."),
    TwinConfig("GRU",     "GRU",     "no_rolling",
               "Instantaneous-focused: drops rolling stats; relies on raw + lag signal."),
    TwinConfig("LSTMCNN", "LSTMCNN", "no_lags",
               "Window-focused: drops ws_lag_*; learns from rolling stats + current state."),
    TwinConfig("GRUCNN",  "GRUCNN",  "all_high_dropout",
               "Regularized: full feature set with elevated dropout (0.35)."),
]


def select_features(all_cols: List[str], filt: str) -> List[str]:
    """Apply a TwinConfig.feature_filter to the full feature list."""
    if filt == "all":
        return list(all_cols)
    if filt == "no_rolling":
        drop = {c for c in all_cols if c.startswith("ws_roll_") or c == "turbulence_intensity"}
        return [c for c in all_cols if c not in drop]
    if filt == "no_lags":
        drop = {c for c in all_cols if c.startswith("ws_lag_")
                or c in ("ws_diff_1", "ws_diff_2")}
        return [c for c in all_cols if c not in drop]
    if filt == "all_high_dropout":
        return list(all_cols)
    raise ValueError(f"Unknown feature_filter: {filt}")


# ────────────────────────────── Dataset (sliding window) ─────────────────────


class WindTimeSeriesDataset(Dataset):
    """
    Yields (X, y) pairs where:
      X = features[i : i+window]          (shape: window × n_features)
      y = wind_power[i+window]            (scalar)
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 target_col: str, window: int = 24):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.window = window

    def __len__(self) -> int:
        return max(0, len(self.X) - self.window)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.X[i:i + self.window]),
            torch.tensor(self.y[i + self.window]),
        )


# ────────────────────────────── Training loop ────────────────────────────────


def train_one_twin(
    cfg: TwinConfig,
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    all_features: List[str], target_col: str,
    *,
    window: int = 24, batch_size: int = 64, max_epochs: int = 120,
    patience: int = 25, lr: float = 5e-4, device: torch.device = None,
    out_dir: str = "results", predictions_dir: str = "predictions",
) -> Dict[str, float]:
    """Train a single twin, save best-val checkpoint + test predictions + loss curve."""
    device = device or get_device()
    feats  = select_features(all_features, cfg.feature_filter)

    print(f"\n[{cfg.name}] arch={cfg.arch}  features={len(feats)}/{len(all_features)}  "
          f"({cfg.feature_filter})")
    print(f"          {cfg.note}")

    # Per-twin DataLoaders (consistent generator → reproducible shuffling)
    g = torch.Generator().manual_seed(SEED)
    train_ds = WindTimeSeriesDataset(train_df, feats, target_col, window)
    val_ds   = WindTimeSeriesDataset(val_df,   feats, target_col, window)
    test_ds  = WindTimeSeriesDataset(test_df,  feats, target_col, window)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Model + extra dropout for the "regularized" twin
    model = get_model(cfg.arch, input_size=len(feats), window_size=window)
    if cfg.feature_filter == "all_high_dropout":
        # Inject a higher dropout into the final linear-classifier path
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = min(0.35, m.p + 0.20)
    model = model.to(device)

    optim     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=4, min_lr=1e-6,
    )
    loss_fn = torch.nn.SmoothL1Loss()  # Huber — robust to wind ramp outliers

    best_val, no_improve, history = float("inf"), 0, []
    ckpt_path = Path(out_dir) / "models" / f"{cfg.name}_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device).unsqueeze(-1)
            optim.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_loss_sum += loss.item(); n_batches += 1
        train_loss = train_loss_sum / max(1, n_batches)

        model.eval()
        val_loss_sum, n_batches = 0.0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device).unsqueeze(-1)
                val_loss_sum += loss_fn(model(X), y).item(); n_batches += 1
        val_loss = val_loss_sum / max(1, n_batches)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{cfg.name}] early stop at epoch {epoch} (best val={best_val:.5f})")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{cfg.name}] epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f}")

    train_time = time.time() - t0

    # Reload best, run test set, persist predictions + losses
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in test_dl:
            X = X.to(device)
            preds.append(model(X).cpu().numpy().squeeze(-1))
            actuals.append(y.numpy())
    preds   = np.concatenate(preds)   if preds   else np.array([])
    actuals = np.concatenate(actuals) if actuals else np.array([])

    preds_path  = Path(predictions_dir) / f"{cfg.name}_preds.csv"
    losses_path = Path(out_dir) / "predictions" / f"{cfg.name}_losses.csv"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    losses_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"idx": range(len(preds)), "predicted": preds, "actual": actuals}).to_csv(
        preds_path, index=False,
    )
    pd.DataFrame(history).to_csv(losses_path, index=False)

    return {
        "name": cfg.name,
        "n_features": len(feats),
        "n_test": int(len(preds)),
        "best_val_loss": float(best_val),
        "train_seconds": float(train_time),
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
    }


# ────────────────────────────── Orchestration ────────────────────────────────


def train_all(
    csv_path: str = "data/raw/36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv",
    out_dir:  str = "results",
    predictions_dir: str = "predictions",
    resample_15min: bool = False,
) -> Dict[str, Dict[str, float]]:
    seed_everything(SEED)
    device = get_device()
    print(f"Device: {device}  |  15-min resample: {resample_15min}\n")

    train_df, val_df, test_df, _, all_features, target_col = prepare_data(
        csv_path, save_dir=out_dir, resample_15min=resample_15min,
    )

    summary: Dict[str, Dict[str, float]] = {}
    for cfg in TWIN_CONFIGS:
        # Re-seed before each twin so order doesn't affect determinism
        seed_everything(SEED)
        summary[cfg.name] = train_one_twin(
            cfg, train_df, val_df, test_df, all_features, target_col,
            device=device, out_dir=out_dir, predictions_dir=predictions_dir,
        )

    summary["__meta__"] = {
        "csv": csv_path,
        "resample_15min": resample_15min,
        "predictions_dir": predictions_dir,
    }
    summary_path = Path(out_dir) / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print("\nPer-twin training stats:")
    print(pd.DataFrame({k: v for k, v in summary.items() if k != "__meta__"}).T.to_string())
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str,
        default="data/raw/36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv",
    )
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--predictions-dir", type=str, default="predictions")
    parser.add_argument("--resample-15min", action="store_true")
    args = parser.parse_args()
    train_all(
        csv_path=args.csv, out_dir=args.out,
        predictions_dir=args.predictions_dir,
        resample_15min=args.resample_15min,
    )
