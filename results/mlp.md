# MLP baseline

## 1. Scope

- Goal: invert the Gilbert damping coefficient $\alpha$ and the uniaxial anisotropy constant $K_u$ from the spatially averaged magnetization trajectory under a single excitation.
- Data: a **synthetic benchmark** at $0\,\mathrm{K}$, noiseless, uniform effective medium, CoFeB-inspired; it describes a fixed discretization protocol, not a reproduction of a specific device.
- Dynamics: LLG, with exchange $E_{\mathrm{ex}}$, uniaxial anisotropy $E_{\mathrm{ani}}$, demagnetizing $E_{\mathrm{demag}}$, and Zeeman $E_{\mathrm{Z}}$ energy terms.
- Model: an MLP regressor taking the averaged trajectory as input; results are limited to the validation set of this synthetic benchmark and do not represent real-device inversion performance.

## 2. Physical model

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
- Ranges: $\alpha \in [0.004, 0.020]$; $K_u \in [2000, 30000]\,\mathrm{J/m^3}$. The main domain excludes the $K_u=0$ control point; independent four-corner diagnostic points are not mixed into this dataset's members.
- Split: by parameter_set_id, split seed 42, fixed 717 / 154 / 153 (train/val/test); the 10 training runs share this frozen split.

## 4. Model and training

- Input $1\times401\times3$ ($m_x,m_y,m_z$), flattened $1203 \to 64 \to 32 \to 32 \to 2$; ReLU hidden layers; $80258$ parameters.
- Label conditions: `identity` uses the raw $\alpha$/$K_u$; `logalpha` takes $\log_{10}$ of $\alpha$ and keeps $K_u$ as-is; each is then z-scored per output.
- Standardization statistics are fit on the training split only; labels in the npz files are not pre-transformed.
- Optimization: Adam, lr $1\times10^{-3}$, batch 32, weight_decay 0; 5 runs per label condition (seeds $42\text{--}46$).
- Cap of 500 epochs, early stopping patience 50, min_delta 0; best selected by validation loss. The loss is an equal-weight MSE over the two standardized outputs.
- All 10 runs completed normally; actual training epochs $115\text{--}428$.

## 5. Validation-set results (val 154; $\mathrm{mean}\pm\mathrm{sample\ SD}$, $\mathrm{ddof}=1$, $n=5$)

| Output | Metric | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $6.456\times10^{-5}\pm1.18\times10^{-5}$ | $5.121\times10^{-5}\pm1.10\times10^{-5}$ |
| $\alpha$ | RMSE | $9.921\times10^{-5}\pm2.00\times10^{-5}$ | $8.715\times10^{-5}\pm1.45\times10^{-5}$ |
| $\alpha$ | mean absolute relative error | $0.759\%\pm0.149\%$ | $0.570\%\pm0.135\%$ |
| $K_u$ ($\mathrm{J/m^3}$) | MAE | $81.23\pm19.49$ | $94.05\pm24.34$ |
| $K_u$ ($\mathrm{J/m^3}$) | RMSE | $140.71\pm23.02$ | $146.01\pm32.43$ |
| $K_u$ ($\mathrm{J/m^3}$) | mean absolute relative error | $1.119\%\pm0.259\%$ | $1.183\%\pm0.251\%$ |

- The paired improvement rate is defined as $100\%\times\frac{\mathrm{RMSE}_{\mathrm{identity}}-\mathrm{RMSE}_{\mathrm{logalpha}}}{\mathrm{RMSE}_{\mathrm{identity}}}$, where a positive value means logalpha is better. $\alpha$ improves in 4/5 pairs, with a mean per-seed improvement rate of $+10.99\%$; $K_u$ improves in 2/5 pairs, with a mean of $-4.55\%$.
- $\alpha$ log transform: logalpha shows an improving trend in the mean error for $\alpha$, but there is variation across seeds; the benefit for $K_u$ is inconsistent. This is not sufficient to claim that it is overall better or that the differences are statistically significant.
