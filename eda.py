"""
Exploratory Data Analysis on NREL India Wind Toolkit data.

Writes:
  results/eda_report.md   — markdown summary, surfaced by dashboard
  results/eda_stats.json  — machine-readable equivalent

This report is the empirical foundation for every feature-engineering and
twin-diversity decision downstream. It quantifies what Indian wind looks
like at NREL site 36565 (Ahmedabad, 2014): wind regime, diurnal cycle,
power-curve regime breakdown, Ekman shear, etc.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from data_pipeline import engineer_features, load_indian_dataset

WIND_COLS  = [f"wind speed at {h}m (m/s)" for h in (40, 80, 100, 120)]
DIR_COLS   = [f"wind direction at {h}m (deg)" for h in (40, 80, 100, 120)]
TEMP_COLS  = [f"temperature at {h}m (C)" for h in (40, 80, 100, 120)]
PRESS_COLS = ["air pressure at 40m (Pa)", "air pressure at 100m (Pa)"]


def _series_stats(s: pd.Series) -> Dict[str, float]:
    return {
        "mean":   float(s.mean()),
        "std":    float(s.std()),
        "min":    float(s.min()),
        "q25":    float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75":    float(s.quantile(0.75)),
        "q95":    float(s.quantile(0.95)),
        "max":    float(s.max()),
        "acf_1":   float(s.autocorr(1)),
        "acf_24":  float(s.autocorr(24)),
        "acf_168": float(s.autocorr(168)),
    }


def analyze(csv_path: str) -> Dict:
    raw      = pd.read_csv(csv_path, header=1)
    raw.columns = raw.columns.str.strip()
    engin    = engineer_features(load_indian_dataset(csv_path))

    v80 = raw["wind speed at 80m (m/s)"]
    wp  = engin["wind_power"]

    out: Dict = {
        "source": os.path.basename(csv_path),
        "n_rows": len(raw),
        "time_granularity_minutes": int(
            raw["Minute"].diff().mode().iat[0]
            if (raw["Hour"].diff().mode().iat[0] == 0) else 60
        ),
        "wind_speed_80m": _series_stats(v80),
        "wind_power_kw":  _series_stats(wp),
        "below_cutin_pct": float((v80 < 3.0).mean() * 100),
        "above_rated_pct": float((v80 > 11.4).mean() * 100),
        "above_cutout_pct": float((v80 > 25.0).mean() * 100),
        "capacity_factor_pct": float(wp.mean() / 5000.0 * 100),
        "target_std_over_mean": float(wp.std() / max(wp.mean(), 1.0)),
        "target_zero_pct": float((wp == 0).mean() * 100),
        "target_rated_pct": float((wp == 5000).mean() * 100),
        "wind_dir_shear_deg_mean": float(
            (((raw["wind direction at 120m (deg)"]
               - raw["wind direction at 40m (deg)"]) + 180) % 360 - 180).mean()
        ),
        "temperature_80m": {
            "mean": float(raw["temperature at 80m (C)"].mean()),
            "std":  float(raw["temperature at 80m (C)"].std()),
            "min":  float(raw["temperature at 80m (C)"].min()),
            "max":  float(raw["temperature at 80m (C)"].max()),
        },
        "pressure_100m_Pa_mean": float(raw["air pressure at 100m (Pa)"].mean()),
    }
    return out


def to_markdown(stats: Dict, *, secondary: Dict | None = None, secondary_label: str = "Secondary") -> str:
    def fmt(x): return f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = [
        f"# EDA — `{stats['source']}`",
        "",
        f"- Rows: **{stats['n_rows']:,}**",
        f"- Time granularity: **{stats['time_granularity_minutes']} min**",
        f"- Capacity factor: **{stats['capacity_factor_pct']:.1f}%**",
        f"- Target std/mean: **{stats['target_std_over_mean']:.2f}** "
        "(lower → easier to forecast)",
        "",
        "## Wind speed @ 80 m",
        "",
        "| Stat | Value |",
        "|---|---|",
        *[f"| {k} | {fmt(v)} |" for k, v in stats["wind_speed_80m"].items()],
        "",
        "## Power-curve regime breakdown",
        "",
        f"- Below cut-in (< 3 m/s) → 0 kW: **{stats['below_cutin_pct']:.1f}%**",
        f"- Above rated (> 11.4 m/s) → saturates at 5 000 kW: "
        f"**{stats['above_rated_pct']:.1f}%**",
        f"- Above cut-out (> 25 m/s) → safety stop: "
        f"**{stats['above_cutout_pct']:.2f}%**",
        f"- Target = 0 (exact): **{stats['target_zero_pct']:.1f}%**",
        f"- Target = rated 5 000 kW: **{stats['target_rated_pct']:.1f}%**",
        "",
        "## Direction & thermal",
        "",
        f"- Mean Ekman-spiral shear (120 m − 40 m): "
        f"**{stats['wind_dir_shear_deg_mean']:+.1f}°**",
        f"- Temperature @ 80 m: "
        f"{stats['temperature_80m']['mean']:.1f} ± "
        f"{stats['temperature_80m']['std']:.1f} °C  "
        f"(range {stats['temperature_80m']['min']:.0f} → "
        f"{stats['temperature_80m']['max']:.0f})",
        f"- Pressure @ 100 m: **{stats['pressure_100m_Pa_mean']:.0f} Pa** "
        "(Ahmedabad ~53 m elevation)",
    ]
    if secondary is not None:
        lines += [
            "",
            f"## This dataset vs {secondary_label}",
            "",
            f"| Property | This | {secondary_label} |",
            "|---|---|---|",
            f"| Rows                   | {stats['n_rows']:,} | {secondary['n_rows']:,} |",
            f"| Granularity (min)      | {stats['time_granularity_minutes']} | {secondary['time_granularity_minutes']} |",
            f"| Wind mean (m/s)        | {stats['wind_speed_80m']['mean']:.2f} | {secondary['wind_speed_80m']['mean']:.2f} |",
            f"| Wind std (m/s)         | {stats['wind_speed_80m']['std']:.2f}  | {secondary['wind_speed_80m']['std']:.2f}  |",
            f"| Wind acf lag-1         | {stats['wind_speed_80m']['acf_1']:.3f} | {secondary['wind_speed_80m']['acf_1']:.3f} |",
            f"| Wind acf lag-24        | {stats['wind_speed_80m']['acf_24']:.3f} | {secondary['wind_speed_80m']['acf_24']:.3f} |",
            f"| Above rated (%)        | {stats['above_rated_pct']:.1f}  | {secondary['above_rated_pct']:.1f}  |",
            f"| Capacity factor (%)    | {stats['capacity_factor_pct']:.1f} | {secondary['capacity_factor_pct']:.1f} |",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default="data/raw/36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv",
        help="NREL India Wind Toolkit CSV to analyze",
    )
    parser.add_argument(
        "--compare-with", default=None,
        help="Optional second CSV (e.g. 15-min resampled) for side-by-side.",
    )
    parser.add_argument("--compare-label", default="Comparison")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    primary = analyze(args.csv)
    secondary = analyze(args.compare_with) if args.compare_with and os.path.exists(args.compare_with) else None

    (Path(args.out) / "eda_stats.json").write_text(json.dumps(primary, indent=2))
    (Path(args.out) / "eda_report.md").write_text(
        to_markdown(primary, secondary=secondary, secondary_label=args.compare_label)
    )
    print(f"Wrote {args.out}/eda_report.md  +  eda_stats.json")
    print(f"\nKey findings on {os.path.basename(args.csv)}:")
    print(f"  wind mean = {primary['wind_speed_80m']['mean']:.2f} m/s")
    print(f"  capacity factor = {primary['capacity_factor_pct']:.1f}%")
    print(f"  std/mean target = {primary['target_std_over_mean']:.2f}")
    print(f"  lag-24 wind acf = {primary['wind_speed_80m']['acf_24']:.3f}")


if __name__ == "__main__":
    main()
