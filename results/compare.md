# Model comparison

## 1. Unified comparison protocol

- Benchmark: the same fixed-discretization CoFeB-inspired **synthetic benchmark** ($0\,\mathrm{K}$, noiseless, uniform effective medium), not a reproduction of a specific device.
- Data and split: 1024 scrambled Sobol points, $\alpha = 0.004\cdot 5^u$ (logarithmic), $K_u = 2000 + 28000\,v$ (linear); by parameter_set_id, split seed 42, fixed 717 / 154 / 153 (train/val/test). **val 154 and test 153 are reported separately; the two splits are never mixed or averaged across splits.**
- This file is a unified evaluation of 30 frozen `best.pt` checkpoints: three models (MLP / CNN1D / Transformer) × `identity` / `logalpha` × seeds $42$–$46$. The 20 MLP/CNN checkpoints carry over from the previous freeze batch (results unchanged); the 10 Transformer checkpoints are the dropout=0 version of this release; all were locked before evaluation, and no checkpoint was swapped or seed selected based on test.
- Version scope: this version uses dropout=0 throughout; the adjustment was based on train/val diagnostics and single-factor experiments, and no test metric was used to choose parameters. The same test split was previously used for evaluating the old version.
- Randomness: 5 runs per (model, label) combination, seeds $42$–$46$; cross-model comparisons and label-transform comparisons are both paired by seed.
- Label conditions: `identity` uses the raw $\alpha$/$K_u$; `logalpha` takes $\log_{10}$ of $\alpha$ only and keeps $K_u$ as-is; each is then z-scored per output, with standardization statistics fit on the training split only.
- Metrics: physical-unit MAE, RMSE, and mean absolute relative error, summarized as $\mathrm{mean}\pm\mathrm{sample\ SD}$ ($\mathrm{ddof}=1$, $n=5$). The six metrics of all 12 model–label–split combinations are in the val/test tables of Section 2.
- Cross-model comparisons under the same label and label-transform comparisons within the same model are both paired by seed; standardized losses are not compared directly across labels.
- History: this file previously contained only the val 154 comparison ([mlp.md](mlp.md) and [cnn.md](cnn.md) are also historical val-only reports, unchanged in this version); the frozen evaluation later added test 153; this version replaces the old dropout=0.1 Transformer runs with the dropout=0 ones (the old artifacts are kept as historical reference and do not contribute to the numbers here), MLP/CNN results are unchanged, and only the current version is shown.

## 2. Model summary

- Parameter counts: MLP $80258$, CNN1D $4930$, Transformer $69474$ (dropout=0 does not change the parameter count); counts are recorded only and are not used to infer speed or efficiency. Structures and training configurations are in the per-model reports ([mlp.md](mlp.md), [cnn.md](cnn.md), [transformer.md](transformer.md)).

### 2.1 val 154

| Model | Label | $\alpha$ MAE | $\alpha$ RMSE | $K_u$ MAE ($\mathrm{J/m^3}$) | $K_u$ RMSE ($\mathrm{J/m^3}$) | $\alpha$ MAPE% | $K_u$ MAPE% |
|---|---|---|---|---|---|---|---|
| MLP | identity | $6.456\times10^{-5}\pm1.18\times10^{-5}$ | $9.921\times10^{-5}\pm2.00\times10^{-5}$ | $81.23\pm19.49$ | $140.71\pm23.02$ | $0.759\%\pm0.149\%$ | $1.119\%\pm0.259\%$ |
| MLP | logalpha | $5.121\times10^{-5}\pm1.10\times10^{-5}$ | $8.715\times10^{-5}\pm1.45\times10^{-5}$ | $94.05\pm24.34$ | $146.01\pm32.43$ | $0.570\%\pm0.135\%$ | $1.183\%\pm0.251\%$ |
| CNN1D | identity | $4.134\times10^{-5}\pm1.08\times10^{-5}$ | $5.867\times10^{-5}\pm1.33\times10^{-5}$ | $66.12\pm13.74$ | $102.68\pm11.19$ | $0.478\%\pm0.113\%$ | $0.752\%\pm0.080\%$ |
| CNN1D | logalpha | $3.024\times10^{-5}\pm8.43\times10^{-6}$ | $5.726\times10^{-5}\pm1.25\times10^{-5}$ | $60.06\pm20.71$ | $98.65\pm29.62$ | $0.288\%\pm0.070\%$ | $0.708\%\pm0.174\%$ |
| Transformer | identity | $2.292\times10^{-5}\pm3.59\times10^{-6}$ | $5.453\times10^{-5}\pm1.42\times10^{-5}$ | $49.04\pm8.04$ | $89.57\pm18.13$ | $0.246\%\pm0.040\%$ | $0.725\%\pm0.202\%$ |
| Transformer | logalpha | $2.121\times10^{-5}\pm4.50\times10^{-6}$ | $5.536\times10^{-5}\pm8.75\times10^{-6}$ | $38.96\pm11.54$ | $64.82\pm8.99$ | $0.196\%\pm0.042\%$ | $0.506\%\pm0.069\%$ |

### 2.2 test 153

| Model | Label | $\alpha$ MAE | $\alpha$ RMSE | $K_u$ MAE ($\mathrm{J/m^3}$) | $K_u$ RMSE ($\mathrm{J/m^3}$) | $\alpha$ MAPE% | $K_u$ MAPE% |
|---|---|---|---|---|---|---|---|
| MLP | identity | $6.404\times10^{-5}\pm1.29\times10^{-5}$ | $8.626\times10^{-5}\pm1.65\times10^{-5}$ | $75.54\pm15.26$ | $103.73\pm23.06$ | $0.762\%\pm0.165\%$ | $0.939\%\pm0.242\%$ |
| MLP | logalpha | $5.143\times10^{-5}\pm1.32\times10^{-5}$ | $7.386\times10^{-5}\pm1.37\times10^{-5}$ | $90.98\pm22.09$ | $120.37\pm22.14$ | $0.587\%\pm0.131\%$ | $1.149\%\pm0.199\%$ |
| CNN1D | identity | $4.108\times10^{-5}\pm1.18\times10^{-5}$ | $5.690\times10^{-5}\pm1.61\times10^{-5}$ | $65.69\pm21.13$ | $92.69\pm30.08$ | $0.490\%\pm0.114\%$ | $0.776\%\pm0.213\%$ |
| CNN1D | logalpha | $2.588\times10^{-5}\pm9.05\times10^{-6}$ | $3.862\times10^{-5}\pm1.39\times10^{-5}$ | $57.78\pm18.12$ | $80.74\pm26.27$ | $0.268\%\pm0.079\%$ | $0.694\%\pm0.217\%$ |
| Transformer | identity | $2.109\times10^{-5}\pm3.83\times10^{-6}$ | $3.233\times10^{-5}\pm9.68\times10^{-6}$ | $45.25\pm9.22$ | $62.34\pm15.95$ | $0.251\%\pm0.036\%$ | $0.616\%\pm0.196\%$ |
| Transformer | logalpha | $2.074\times10^{-5}\pm2.52\times10^{-6}$ | $3.214\times10^{-5}\pm5.61\times10^{-6}$ | $35.80\pm9.59$ | $48.39\pm9.51$ | $0.209\%\pm0.026\%$ | $0.448\%\pm0.085\%$ |

## 3. Seed-paired same-label between-architecture comparison (val 154)

- Improvement rate = $100\%\times\frac{\mathrm{baseline\ error}-\mathrm{candidate\ error}}{\mathrm{baseline\ error}}$; a positive value means the candidate has lower error, a negative value means it is worse. The expression uses the per-seed relative difference, not a ratio of group means; non-error metrics such as $R^2$ are not fed into it.
- The table below lists paired improvement rates only; group-mean RMSEs are in Section 2.

| baseline → candidate | Label | Output | Per-seed improvement rate (mean±SD) | $n$ improved/5 |
|---|---|---|---|---|
| MLP → CNN1D | identity | $\alpha$ | $+37.17\%\pm25.13\%$ | 5/5 |
| MLP → CNN1D | identity | $K_u$ | $+26.09\%\pm10.44\%$ | 5/5 |
| MLP → CNN1D | logalpha | $\alpha$ | $+33.33\%\pm16.12\%$ | 5/5 |
| MLP → CNN1D | logalpha | $K_u$ | $+28.73\%\pm31.75\%$ | 4/5 |
| MLP → Transformer | identity | $\alpha$ | $+43.96\%\pm14.68\%$ | 5/5 |
| MLP → Transformer | identity | $K_u$ | $+33.66\%\pm21.75\%$ | 5/5 |
| MLP → Transformer | logalpha | $\alpha$ | $+35.62\%\pm12.06\%$ | 5/5 |
| MLP → Transformer | logalpha | $K_u$ | $+54.41\%\pm8.65\%$ | 5/5 |
| CNN1D → Transformer | identity | $\alpha$ | $+2.44\%\pm36.18\%$ | 3/5 |
| CNN1D → Transformer | identity | $K_u$ | $+11.59\%\pm20.76\%$ | 3/5 |
| CNN1D → Transformer | logalpha | $\alpha$ | $+1.47\%\pm16.97\%$ | 4/5 |
| CNN1D → Transformer | logalpha | $K_u$ | $+30.33\%\pm19.02\%$ | 5/5 |

## 4. Seed-paired same-label between-architecture comparison (test 153)

- Same definition as Section 3; this version did not use test metrics to choose parameters, and the results are not fed back into tuning or seed selection.

| baseline → candidate | Label | Output | Per-seed improvement rate (mean±SD) | $n$ improved/5 |
|---|---|---|---|---|
| MLP → CNN1D | identity | $\alpha$ | $+30.57\%\pm31.40\%$ | 4/5 |
| MLP → CNN1D | identity | $K_u$ | $+7.97\%\pm37.70\%$ | 4/5 |
| MLP → CNN1D | logalpha | $\alpha$ | $+47.47\%\pm15.12\%$ | 5/5 |
| MLP → CNN1D | logalpha | $K_u$ | $+31.77\%\pm21.73\%$ | 5/5 |
| MLP → Transformer | identity | $\alpha$ | $+61.50\%\pm11.64\%$ | 5/5 |
| MLP → Transformer | identity | $K_u$ | $+36.03\%\pm26.34\%$ | 4/5 |
| MLP → Transformer | logalpha | $\alpha$ | $+56.27\%\pm4.31\%$ | 5/5 |
| MLP → Transformer | logalpha | $K_u$ | $+59.69\%\pm3.92\%$ | 5/5 |
| CNN1D → Transformer | identity | $\alpha$ | $+39.19\%\pm25.65\%$ | 5/5 |
| CNN1D → Transformer | identity | $K_u$ | $+28.64\%\pm21.44\%$ | 5/5 |
| CNN1D → Transformer | logalpha | $\alpha$ | $+10.40\%\pm28.51\%$ | 3/5 |
| CNN1D → Transformer | logalpha | $K_u$ | $+35.73\%\pm21.59\%$ | 4/5 |

## 5. Label effect and boundaries

- Same-architecture identity → logalpha per-seed label effect: Transformer val/test in [transformer.md](transformer.md); the historical val analyses of MLP/CNN are in [mlp.md](mlp.md) and [cnn.md](cnn.md); those two historical reports do not contain test label effects, and their val conclusions must not be applied to test.
- Two kinds of pairing must be distinguished: Sections 3/4 are "between architectures under the same label" pairs, while the label transform is a "between labels within the same architecture" pair; they must not be mixed.
- Descriptive comparison: in this version the **mean paired improvement rate** from MLP → Transformer is positive on val/test and under both labels (per-seed improved counts $n$ = 4/5–5/5 for each combination); CNN1D → Transformer is also positive on average but less consistent per seed ($n$ = 3/5–5/5). **A positive mean does not mean that every seed improves**: read it together with the "$n$ improved/5" column of Sections 3/4; some combinations have individual seeds where CNN1D has lower error, and no claim can be made that either model is universally better. By group means, all six test metrics are lowest for Transformer/logalpha (see Section 2.2).
- Scope limits: a single fixed frozen split; the 5 seeds share the same split (not 5 independent data splits), and a matching seed number does not mean identical random streams. The differences do not constitute a statistical-significance conclusion and cannot be generalized into "some model is universally better" or a real-device conclusion; test only supports in-distribution held-out performance under this fixed synthetic protocol and does not prove noise conditions, extrapolation ranges, multi-excitation transfer, or physical forward re-validation. The freeze rules, artifacts, and fingerprints of this version are in Section 6.
- The training cap and early-stopping rules are the same (at most 500 epochs, patience 50, min_delta 0), while the actual stopping epochs differ (MLP 115–428, CNN 208–500: 9 early stops, 1 reaching the cap; Transformer 125–412: all 10 stopped early); it cannot be concluded that all runs fully converged.

## 6. Traceability

- Freeze batch (relative to the repository root; located in the ignored `artifacts/` tree, raw artifacts are not committed to Git): `artifacts/experiments/three_model_dropout0_eval_20260928T082953Z/`.
- `freeze_manifest.json` (SHA256 `31f7578eac84c3d3411454f69e924fceb614eae93de73c5d70eeea2b6a89c87d`) locks the SHA of the 30 `best.pt` files together with fingerprints such as `config_resolved`, `preprocessing`, and split copies.
- Evaluation code HEAD `7ea7d1b679720f956786fa644b890313a67e2d7e`; dataset_meta `b1d99161a7ef1cc72ff802b7bef2419d84428998967d3d2aaa5a2bbf49165659`; split `b49c9dd32ab6cd7c48caf15ebbbadd749d115da35dc9a29c5a80ad8c3e223ad1` (train/val/test = 717/154/153).
- The numbers follow the batch's `per_run_metrics.csv` (60 rows = 30 runs × val/test) and `aggregate_metrics.csv` (72 rows = 12 groups × 6 metrics); the current SHAs and reuse lineage of existing artifacts are in `artifact_lineage.json` and `verified_artifacts.json`.
- The old dropout=0.1 Transformer runs and their frozen evaluation batch `artifacts/experiments/three_model_frozen_eval_20260928T063042Z/` are kept as superseded historical reference and do not contribute to this version's numbers.
