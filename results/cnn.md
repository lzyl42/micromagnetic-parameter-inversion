# CNN1D baseline

## 1. Model and training

- Benchmark, sampling, and the frozen train/val/test split: see [Model comparison](README.md). Both label conditions share the same frozen split and correspond seed by seed.
- Model: a 1D CNN regressor taking the averaged trajectory as input.
- Input $1\times401\times3$ ($m_x,m_y,m_z$); pulses and components are merged into 3 channels of length 401.
- Structure: two Conv1d layers (channels $3\to8\to16$, kernels $5/5$, stride 1, dilation 1, padding 2, each followed by ReLU) → AdaptiveAvgPool1d(16) → Flatten(256) → Linear(16) + ReLU → Linear(2); $4930$ parameters in total.
- Label conditions: `identity` uses the raw $\alpha$/$K_u$; `logalpha` takes $\log_{10}$ of $\alpha$ only and keeps $K_u$ as-is; each is then z-scored per output.
- Standardization statistics are fit on the training split only.
- Optimization: Adam, lr $1\times10^{-3}$, batch 32, weight_decay 0; 5 runs per label condition (seeds $42$–$46$).
- Cap of 500 epochs, early stopping patience 50, min_delta 0; best selected by the equal-weight validation MSE over the two standardized outputs.
- Actual training epochs $208$–$500$: 9 early stops, 1 reaching the 500-epoch cap. This range describes the results and does not imply that every run has fully converged.
- Within a label condition, the 5 runs share the same frozen split; structure and hyperparameters are identical except for the seed.

## 2. Validation-set results (val 154; $\mathrm{mean}\pm\mathrm{sample\ SD}$, $\mathrm{ddof}=1$, $n=5$)

| Output | Metric | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $4.134\times10^{-5}\pm1.08\times10^{-5}$ | $3.024\times10^{-5}\pm8.43\times10^{-6}$ |
| $\alpha$ | RMSE | $5.867\times10^{-5}\pm1.33\times10^{-5}$ | $5.726\times10^{-5}\pm1.25\times10^{-5}$ |
| $\alpha$ | mean absolute relative error | $0.478\%\pm0.113\%$ | $0.288\%\pm0.070\%$ |
| $K_u$ ($\mathrm{J/m^3}$) | MAE | $66.12\pm13.74$ | $60.06\pm20.71$ |
| $K_u$ ($\mathrm{J/m^3}$) | RMSE | $102.68\pm11.19$ | $98.65\pm29.62$ |
| $K_u$ ($\mathrm{J/m^3}$) | mean absolute relative error | $0.752\%\pm0.080\%$ | $0.708\%\pm0.174\%$ |

## 3. Test-set results (test 153; $\mathrm{mean}\pm\mathrm{sample\ SD}$, $\mathrm{ddof}=1$, $n=5$)

| Output | Metric | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $4.108\times10^{-5}\pm1.18\times10^{-5}$ | $2.588\times10^{-5}\pm9.05\times10^{-6}$ |
| $\alpha$ | RMSE | $5.690\times10^{-5}\pm1.61\times10^{-5}$ | $3.862\times10^{-5}\pm1.39\times10^{-5}$ |
| $\alpha$ | mean absolute relative error | $0.490\%\pm0.114\%$ | $0.268\%\pm0.079\%$ |
| $K_u$ ($\mathrm{J/m^3}$) | MAE | $65.69\pm21.13$ | $57.78\pm18.12$ |
| $K_u$ ($\mathrm{J/m^3}$) | RMSE | $92.69\pm30.08$ | $80.74\pm26.27$ |
| $K_u$ ($\mathrm{J/m^3}$) | mean absolute relative error | $0.776\%\pm0.213\%$ | $0.694\%\pm0.217\%$ |

## 4. Label-transform comparison

- The paired improvement rate of a label transform is defined as $100\%\times(E_{\mathrm{identity},s}-E_{\mathrm{logalpha},s})/E_{\mathrm{identity},s}$, where $E$ is the corresponding error metric; a positive value means logalpha is better, and the values below are the mean ± sample SD of the per-seed improvement rates.
- The label effect must be read per metric (val 154). The $\alpha$ MAE improves in 4/5 pairs; the mean absolute relative error improves in 5/5 pairs ($+36.11\%\pm24.48\%$). However, the $\alpha$ RMSE improves in only 2/5 pairs, at $-2.14\%\pm32.03\%$.
- The $K_u$ RMSE improves in 3/5 pairs, mean $+4.78\%\pm21.54\%$, with similarly pronounced seed-to-seed variation.
- Therefore `logalpha` cannot simply be called better overall: the $\alpha$ MAE mostly improves and the relative error improves on every seed, while the RMSE is unstable; the $K_u$ benefit varies with the seed.
- The group-mean $\alpha$ RMSE drops slightly from identity to logalpha, which is not contradicted by the negative mean per-seed paired relative difference for $\alpha$: the former is a difference of group-mean RMSEs, the latter is an average of per-seed relative differences — different conventions (this note applies to $\alpha$ only).
