# Model comparison

## 1. Unified comparison protocol

- Benchmark: the same fixed-discretization CoFeB-inspired **synthetic benchmark** ($0\,\mathrm{K}$, noiseless, uniform effective medium), not a reproduction of a specific device.
- Data and split: 1024 scrambled Sobol points, $\alpha = 0.004\cdot 5^u$ (logarithmic), $K_u = 2000 + 28000\,v$ (linear); by parameter_set_id, split seed 42, fixed 717 / 154 / 153 (train/val/test). The comparison uses **val 154** only.
- Randomness: 5 runs per (model, label) combination, seeds $42$–$46$.
- Label conditions: `identity` uses the raw $\alpha$/$K_u$; `logalpha` takes $\log_{10}$ of $\alpha$ only and keeps $K_u$ as-is; each is then z-scored per output, with standardization statistics fit on the training split only.
- Metrics: physical-unit MAE, RMSE, and mean absolute relative error, summarized as $\mathrm{mean}\pm\mathrm{sample\ SD}$ ($\mathrm{ddof}=1$, $n=5$).
- Cross-model comparisons under the same label and label-transform comparisons within the same model are both paired by seed; standardized losses are not compared directly across labels.

## 2. Model summary

- MLP has $80258$ parameters ([MLP baseline](mlp.md)); CNN has $4930$ parameters ([CNN1D baseline](cnn.md)). Parameter counts are recorded only and are not used to infer speed or efficiency.

| Model | Label | $\alpha$ RMSE | $K_u$ RMSE ($\mathrm{J/m^3}$) | $\alpha$ mean absolute relative error | $K_u$ mean absolute relative error |
|---|---|---|---|---|---|
| MLP | identity | $9.921\times10^{-5}\pm2.00\times10^{-5}$ | $140.71\pm23.02$ | $0.759\%\pm0.149\%$ | $1.119\%\pm0.259\%$ |
| MLP | logalpha | $8.715\times10^{-5}\pm1.45\times10^{-5}$ | $146.01\pm32.43$ | $0.570\%\pm0.135\%$ | $1.183\%\pm0.251\%$ |
| CNN | identity | $5.867\times10^{-5}\pm1.33\times10^{-5}$ | $102.68\pm11.19$ | $0.478\%\pm0.113\%$ | $0.752\%\pm0.080\%$ |
| CNN | logalpha | $5.726\times10^{-5}\pm1.25\times10^{-5}$ | $98.65\pm29.62$ | $0.288\%\pm0.070\%$ | $0.708\%\pm0.174\%$ |

## 3. Seed-paired comparison with the MLP baseline

- For a candidate model $m$ and the MLP, the per-seed paired improvement rate is defined as $100\%\times\frac{\mathrm{RMSE}_{\mathrm{MLP},s}-\mathrm{RMSE}_{m,s}}{\mathrm{RMSE}_{\mathrm{MLP},s}}$; a positive value means the candidate has lower error.
- This expression uses per-seed relative differences, not a ratio of group means; non-error metrics such as $R^2$ are not fed into it. The table below lists paired improvement rates only; group-mean RMSEs are in Section 2.

| Model | Label | Output | Per-seed improvement rate (mean±SD) | $n$ improved |
|---|---|---|---|---|
| CNN | identity | $\alpha$ | $+37.17\%\pm25.13\%$ | 5/5 |
| CNN | identity | $K_u$ | $+26.09\%\pm10.44\%$ | 5/5 |
| CNN | logalpha | $\alpha$ | $+33.33\%\pm16.12\%$ | 5/5 |
| CNN | logalpha | $K_u$ | $+28.73\%\pm31.75\%$ | 4/5 |

## 4. Label effect and boundaries

- Across architectures, the same label effect (identity → logalpha, positive = logalpha better, seed-paired):
  - $\alpha$ RMSE: MLP 4/5, mean $+10.99\%$; CNN 2/5, mean $-2.14\%$, with a different direction and stability.
  - $K_u$ RMSE: MLP 2/5, mean $-4.55\%$; CNN 3/5, mean $+4.78\%$, both benefits varying with the seed.
- CNN's own label-effect details (`logalpha` improves the $\alpha$ mean absolute relative error in 5/5 pairs and the MAE in 4/5 pairs) are in Section 5 of the [CNN1D baseline](cnn.md).
- Two kinds of conclusions must be distinguished: Section 3 is paired "between architectures under the same label", this section is paired "between labels within the same architecture"; they must not be mixed.
- Scope limits: a single fixed split, 5 seeds, val 154; test was not used for this comparison, and there is no MuMax3 forward re-validation. The above differences are not statistical-significance or independent-test generalization conclusions and cannot be generalized to "some model is universally better".
- The training cap and early-stopping rules are the same (at most 500 epochs, patience 50, min_delta 0), although the actual stopping epochs differ. CNN `logalpha` seed 42 reached the 500-epoch cap while the rest stopped early, so it cannot be concluded that all runs fully converged.
