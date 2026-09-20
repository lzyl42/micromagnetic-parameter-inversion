# CNN1D baseline

## 1. Scope

- Goal: invert the Gilbert damping coefficient $\alpha$ and the uniaxial anisotropy constant $K_u$ from the spatially averaged magnetization trajectory under a single excitation.
- Data: a **synthetic benchmark** at $0\,\mathrm{K}$, noiseless, uniform effective medium, CoFeB-inspired; it describes a fixed discretization protocol, not a reproduction of a specific device.
- Dynamics: LLG, with exchange $E_{\mathrm{ex}}$, uniaxial anisotropy $E_{\mathrm{ani}}$, demagnetizing $E_{\mathrm{demag}}$, and Zeeman $E_{\mathrm{Z}}$ energy terms.
- Model: a 1D CNN regressor taking the averaged trajectory as input; results are limited to the **validation set** of this synthetic benchmark, do not represent real-device inversion performance, and are not a generalization conclusion.
- Reported quantities: validation-set MAE, RMSE, and mean absolute relative error in physical units; no forward re-validation or device-level conclusion is involved.

## 2. Physical model

Fixed synthetic benchmark:

| Item | Value |
|---|---|
| $M_s$ | $1.25\times10^6\,\mathrm{A/m}$ |
| $A_{\mathrm{ex}}$ | $15\,\mathrm{pJ/m}$ |
| Geometry | true triaxial ellipsoid with full diameters $100\times50\times2\,\mathrm{nm}$, cells $40\times20\times4$ ($2.5\times2.5\times0.5\,\mathrm{nm}$) |
| Initial magnetization / easy axis | $+x$ / uniaxial $+x$ |
| Excitation | pulse_A2: $2\,\mathrm{mT}$ along $y$, $50\,\mathrm{ps}$ |
| Recording | $0\text{--}4\,\mathrm{ns}$ after field switch-off, $10\,\mathrm{ps}\times401$ points |
| Numerics | ES12, solver 5, MaxErr $1\times10^{-5}$, MaxDt $10\,\mathrm{ps}$, GammaLL $1.7595\times10^{11}\,\mathrm{rad/(T\cdot s)}$, RelaxTorqueThreshold $-1$ |

## 3. Sampling and split

- Sampling: 1024 scrambled Sobol points (`rng=42`, `random_base2(10)`); for a point $(u,v)$ in the unit square, $\alpha = 0.004\cdot 5^u$ (logarithmic) and $K_u = 2000 + 28000\,v$ (linear).
- Ranges: $\alpha \in [0.004, 0.020]$; $K_u \in [2000, 30000]\,\mathrm{J/m^3}$. The main domain excludes the $K_u=0$ control point.
- Split: by parameter_set_id, split seed 42, fixed 717 / 154 / 153 (train/val/test); the two label conditions of this model share this frozen split.
- The two label conditions use the same frozen split and correspond seed by seed.

## 4. Model and training

- Input $1\times401\times3$ ($m_x,m_y,m_z$); pulses and components are merged into 3 channels of length 401.
- Structure: two Conv1d layers (channels $3\to8\to16$, kernels $5/5$, stride 1, dilation 1, padding 2, each followed by ReLU) → AdaptiveAvgPool1d(16) → Flatten(256) → Linear(16) + ReLU → Linear(2); $4930$ parameters in total.
- Label conditions: `identity` uses the raw $\alpha$/$K_u$; `logalpha` takes $\log_{10}$ of $\alpha$ only and keeps $K_u$ as-is; each is then z-scored per output.
- Standardization statistics are fit on the training split only.
- Optimization: Adam, lr $1\times10^{-3}$, batch 32, weight_decay 0; 5 runs per label condition (seeds $42$–$46$).
- Cap of 500 epochs, early stopping patience 50, min_delta 0; best selected by the equal-weight validation MSE over the two standardized outputs.
- Actual training epochs $208$–$500$: 9 early stops, 1 reaching the 500-epoch cap. This range describes the results and does not imply that every run has fully converged.
- Within a label condition, the 5 runs share the same frozen split; structure and hyperparameters are identical except for the seed.

## 5. Validation-set results (val 154; $\mathrm{mean}\pm\mathrm{sample\ SD}$, $\mathrm{ddof}=1$, $n=5$)

| Output | Metric | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $4.134\times10^{-5}\pm1.08\times10^{-5}$ | $3.024\times10^{-5}\pm8.43\times10^{-6}$ |
| $\alpha$ | RMSE | $5.867\times10^{-5}\pm1.33\times10^{-5}$ | $5.726\times10^{-5}\pm1.25\times10^{-5}$ |
| $\alpha$ | mean absolute relative error | $0.478\%\pm0.113\%$ | $0.288\%\pm0.070\%$ |
| $K_u$ ($\mathrm{J/m^3}$) | MAE | $66.12\pm13.74$ | $60.06\pm20.71$ |
| $K_u$ ($\mathrm{J/m^3}$) | RMSE | $102.68\pm11.19$ | $98.65\pm29.62$ |
| $K_u$ ($\mathrm{J/m^3}$) | mean absolute relative error | $0.752\%\pm0.080\%$ | $0.708\%\pm0.174\%$ |

- The paired improvement rate of a label transform is defined as $100\%\times(E_{\mathrm{identity},s}-E_{\mathrm{logalpha},s})/E_{\mathrm{identity},s}$, where $E$ is the corresponding error metric; a positive value means logalpha is better, and the values below are the mean ± sample SD of the per-seed improvement rates.
- The label effect must be read per metric. The $\alpha$ MAE improves in 4/5 pairs; the mean absolute relative error improves in 5/5 pairs ($+36.11\%\pm24.48\%$). However, the $\alpha$ RMSE improves in only 2/5 pairs, at $-2.14\%\pm32.03\%$.
- The $K_u$ RMSE improves in 3/5 pairs, mean $+4.78\%\pm21.54\%$, with similarly pronounced seed-to-seed variation.
- Therefore `logalpha` cannot simply be called better overall: the $\alpha$ MAE mostly improves and the relative error improves on every seed, while the RMSE is unstable; the $K_u$ benefit varies with the seed.
- The group-mean $\alpha$ RMSE drops slightly from identity to logalpha, which is not contradicted by the negative mean per-seed paired relative difference for $\alpha$: the former is a difference of group-mean RMSEs, the latter is an average of per-seed relative differences — different conventions (this note applies to $\alpha$ only).

## 6. Scope and extensions

- Scope limits: a single fixed split, 5 seeds, val 154; test was not used for these results, and there is no MuMax3 forward re-validation. The above is not a statistical-significance or generalization conclusion.
- Cross-model comparison (with the MLP baseline and later models) is in [model comparison](compare.md).
