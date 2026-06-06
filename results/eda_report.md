# EDA — `36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv`

- Rows: **8,760**
- Time granularity: **60 min**
- Capacity factor: **17.9%**
- Target std/mean: **1.21** (lower → easier to forecast)

## Wind speed @ 80 m

| Stat | Value |
|---|---|
| mean | 5.452 |
| std | 2.658 |
| min | 0.040 |
| q25 | 3.370 |
| median | 5.315 |
| q75 | 7.340 |
| q95 | 10.080 |
| max | 14.960 |
| acf_1 | 0.900 |
| acf_24 | 0.664 |
| acf_168 | 0.463 |

## Power-curve regime breakdown

- Below cut-in (< 3 m/s) → 0 kW: **20.7%**
- Above rated (> 11.4 m/s) → saturates at 5 000 kW: **1.0%**
- Above cut-out (> 25 m/s) → safety stop: **0.00%**
- Target = 0 (exact): **20.7%**
- Target = rated 5 000 kW: **1.0%**

## Direction & thermal

- Mean Ekman-spiral shear (120 m − 40 m): **+2.2°**
- Temperature @ 80 m: 27.5 ± 5.4 °C  (range 11 → 43)
- Pressure @ 100 m: **98869 Pa** (Ahmedabad ~53 m elevation)

## Real vs synthetic — quick deltas

| Property | Real | Synthetic | Implication |
|---|---|---|---|
| Wind mean (m/s)        | 5.45 | 6.83 | mismatch tells us how far the simulator is from this site |
| Wind std (m/s)         | 2.66  | 4.52  | spread affects ramp prediction difficulty |
| Wind acf lag-24        | 0.664 | 0.312 | higher acf-24 → time features dominate prediction |
| Above rated (%)        | 1.0  | 15.3  | high → bimodal target → harder forecasting |
| Capacity factor (%)    | 17.9 | 31.4 | lower CF → smoother target distribution |
