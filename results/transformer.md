# Temporal Transformer baseline

## 1. Scope

- Goal: invert the Gilbert damping coefficient $\alpha$ and the uniaxial anisotropy constant $K_u$ from the spatially averaged magnetization trajectory under a single excitation.
- Data: a $0\,\mathrm{K}$, noiseless, uniform effective medium, CoFeB-inspired **synthetic benchmark**; it describes a fixed discretization protocol, not a reproduction of a specific device.
- Model: a Temporal Transformer regressor taking the time–pulse–component sequence as input.
- Reported quantities: physical-unit MAE, RMSE, and mean absolute relative error on val 154 and test 153 of the frozen split.
- Version scope: this version uses dropout=0 throughout; the adjustment was based on train/val diagnostics and single-factor experiments, and no test metric was used to choose parameters. The same test split was previously used for evaluating the old version.
- Data, the freeze rules, and the complete three-model summary are in [Model comparison](compare.md) (unified protocol in Section 1, complete val/test six-metric tables in Section 2); the two baselines are in [mlp.md](mlp.md) and [cnn.md](cnn.md).

## 2. Physical model

Shared with the two baselines (facts reused from [mlp.md](mlp.md) and [cnn.md](cnn.md)):

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

- Sampling: 1024 scrambled Sobol points (`rng=42`, `random_base2(10)`); $\alpha = 0.004\cdot 5^u$ (logarithmic), $K_u = 2000 + 28000\,v$ (linear).
- Ranges: $\alpha \in [0.004, 0.020]$; $K_u \in [2000, 30000]\,\mathrm{J/m^3}$. The main domain excludes the $K_u=0$ control point.
- Split: by parameter_set_id, split seed 42, fixed 717 / 154 / 153 (train/val/test); both label conditions share this frozen split, paired one-to-one per seed.

## 4. Model and training

- Input layout: $[B, P, T, 3] \to [B, T, 3P]$ (token semantics time–pulse–component; for P1, T401 it is $[B, 401, 3]$).
- Structure: $d_{\mathrm{model}}=64$, 4 heads, 2 layers, FFN 128, dropout 0, head 32 (GELU activation); fixed sinusoidal positional encoding (base 10000); pre-LN + final-LN; mean pooling; $69474$ parameters in total (the dropout adjustment changes neither the parameter count nor the rest of the structure).
- Label conditions: `identity` uses the raw $\alpha$/$K_u$; `logalpha` takes $\log_{10}$ of $\alpha$ only and keeps $K_u$ as-is; both conditions then apply a **train-only** z-score to both outputs.
- Optimization: Adam, lr $1\times10^{-3}$, batch 32, weight_decay 0; 5 runs per label condition (seeds $42$–$46$).
- Run provenance: the seed-42 identity/logalpha runs are the dropout=0 runs of the dropout single-factor experiment batch (reused); seeds 43–46 were newly run in this multi-seed batch; this frozen evaluation did not retrain. The dropout=0 adjustment was based on train/val diagnostics and single-factor experiments, and no test metric was used.
- Cap of 500 epochs, early stopping patience 50, min_delta 0; best selected by the equal-weight validation MSE over the two standardized outputs.
- Actual training epochs $125$–$412$; all 10 runs triggered early stopping; the loss is an equal-weight MSE over the two standardized outputs.

## 5. Validation-set results (val 154)

$\mathrm{mean}\pm\mathrm{sample\ SD}$ ($\mathrm{ddof}=1$, $n=5$); $\alpha$ is dimensionless, $K_u$ is in $\mathrm{J/m^3}$, and MAPE is a percentage.

| Output | Metric | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $2.292\times10^{-5}\pm3.59\times10^{-6}$ | $2.121\times10^{-5}\pm4.50\times10^{-6}$ |
| $\alpha$ | RMSE | $5.453\times10^{-5}\pm1.42\times10^{-5}$ | $5.536\times10^{-5}\pm8.75\times10^{-6}$ |
| $\alpha$ | MAPE% | $0.246\%\pm0.040\%$ | $0.196\%\pm0.042\%$ |
| $K_u$ | MAE | $49.04\pm8.04$ | $38.96\pm11.54$ |
| $K_u$ | RMSE | $89.57\pm18.13$ | $64.82\pm8.99$ |
| $K_u$ | MAPE% | $0.725\%\pm0.202\%$ | $0.506\%\pm0.069\%$ |

Label effect (same architecture, identity → logalpha, paired per seed, improvement = $100\%\times(\mathrm{identity}-\mathrm{logalpha})/\mathrm{identity}$; a positive value means logalpha is lower):

| Output | Per-seed improvement rate (mean±SD) | $n$ improved/5 |
|---|---|---|
| $\alpha$ RMSE | $-5.75\%\pm28.34\%$ | 3/5 |
| $K_u$ RMSE | $+23.66\%\pm27.89\%$ | 4/5 |

- val participates in best selection and carries best-checkpoint selection bias.

## 6. Test results (153)

| Output | Metric | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $2.109\times10^{-5}\pm3.83\times10^{-6}$ | $2.074\times10^{-5}\pm2.52\times10^{-6}$ |
| $\alpha$ | RMSE | $3.233\times10^{-5}\pm9.68\times10^{-6}$ | $3.214\times10^{-5}\pm5.61\times10^{-6}$ |
| $\alpha$ | MAPE% | $0.251\%\pm0.036\%$ | $0.209\%\pm0.026\%$ |
| $K_u$ | MAE | $45.25\pm9.22$ | $35.80\pm9.59$ |
| $K_u$ | RMSE | $62.34\pm15.95$ | $48.39\pm9.51$ |
| $K_u$ | MAPE% | $0.616\%\pm0.196\%$ | $0.448\%\pm0.085\%$ |

Label effect (same definition as in Section 5):

| Output | Per-seed improvement rate (mean±SD) | $n$ improved/5 |
|---|---|---|
| $\alpha$ RMSE | $-4.99\%\pm31.77\%$ | 4/5 |
| $K_u$ RMSE | $+16.94\%\pm33.54\%$ | 4/5 |

- This version did not use test metrics for selection; test is used only for the evaluation report after the frozen split, not for tuning or seed selection.

## 7. Label effect and boundaries

- The per-seed mean improvement for $K_u$ RMSE is positive (val $+23.66\%\pm27.89\%$, 4/5; test $+16.94\%\pm33.54\%$, 4/5), while the mean for $\alpha$ RMSE is slightly negative (val $-5.75\%\pm28.34\%$, 3/5; test $-4.99\%\pm31.77\%$, 4/5). Note that the **group means** of `logalpha` and identity on $\alpha$ are close and cross over (the test group means are slightly lower on all three metrics, e.g. RMSE $3.214\times10^{-5}$ vs $3.233\times10^{-5}$; val MAE/MAPE are slightly lower while the $\alpha$ RMSE is slightly higher): the group-mean difference and the mean per-seed relative improvement are two different quantities and do not contradict each other (the negative mean comes mainly from large negative outliers in individual seeds). This report keeps that unstable convention, does not mix the two aggregations, and does not claim that `logalpha` is stably or significantly better.
- Scope limits: a single fixed frozen split; the 5 seeds share the same split (not 5 independent data splits), and a matching seed number does not mean identical random streams. The conclusions only support in-distribution held-out performance under the current fixed synthetic protocol; they do not prove real-device validity, noise conditions, extrapolation ranges, multi-excitation transfer, physical forward re-validation, or statistical significance (full boundary discussion in Section 5 of [Model comparison](compare.md)).
- The old dropout=0.1 Transformer artifacts are kept as historical reference (superseded); their numbers are not merged into this version's conclusions.
- Cross-links: [MLP baseline](mlp.md), [CNN1D baseline](cnn.md), [Model comparison](compare.md).
