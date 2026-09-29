# MLP baseline

## 1. Model and training

- Benchmark, sampling, and the frozen train/val/test split: see [Model comparison](README.md). Both label conditions share the same frozen split; the 10 training runs differ only by seed.
- Model: an MLP regressor taking the averaged trajectory as input.
- Input $1\times401\times3$ ($m_x,m_y,m_z$), flattened $1203 \to 64 \to 32 \to 32 \to 2$; ReLU hidden layers; $80258$ parameters.
- Label conditions: `identity` uses the raw $\alpha$/$K_u$; `logalpha` takes $\log_{10}$ of $\alpha$ and keeps $K_u$ as-is; each is then z-scored per output.
- Standardization statistics are fit on the training split only; labels in the npz files are not pre-transformed.
- Optimization: Adam, lr $1\times10^{-3}$, batch 32, weight_decay 0; 5 runs per label condition (seeds $42\text{--}46$).
- Cap of 500 epochs, early stopping patience 50, min_delta 0; best selected by validation loss. The loss is an equal-weight MSE over the two standardized outputs.
- All 10 runs completed normally; actual training epochs $115\text{--}428$.

## 2. Validation-set results (val 154; $\mathrm{mean}\pm\mathrm{sample\ SD}$, $\mathrm{ddof}=1$, $n=5$)

| Output | Metric | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $6.456\times10^{-5}\pm1.18\times10^{-5}$ | $5.121\times10^{-5}\pm1.10\times10^{-5}$ |
| $\alpha$ | RMSE | $9.921\times10^{-5}\pm2.00\times10^{-5}$ | $8.715\times10^{-5}\pm1.45\times10^{-5}$ |
| $\alpha$ | mean absolute relative error | $0.759\%\pm0.149\%$ | $0.570\%\pm0.135\%$ |
| $K_u$ ($\mathrm{J/m^3}$) | MAE | $81.23\pm19.49$ | $94.05\pm24.34$ |
| $K_u$ ($\mathrm{J/m^3}$) | RMSE | $140.71\pm23.02$ | $146.01\pm32.43$ |
| $K_u$ ($\mathrm{J/m^3}$) | mean absolute relative error | $1.119\%\pm0.259\%$ | $1.183\%\pm0.251\%$ |

## 3. Test-set results (test 153; $\mathrm{mean}\pm\mathrm{sample\ SD}$, $\mathrm{ddof}=1$, $n=5$)

| Output | Metric | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $6.404\times10^{-5}\pm1.29\times10^{-5}$ | $5.143\times10^{-5}\pm1.32\times10^{-5}$ |
| $\alpha$ | RMSE | $8.626\times10^{-5}\pm1.65\times10^{-5}$ | $7.386\times10^{-5}\pm1.37\times10^{-5}$ |
| $\alpha$ | mean absolute relative error | $0.762\%\pm0.165\%$ | $0.587\%\pm0.131\%$ |
| $K_u$ ($\mathrm{J/m^3}$) | MAE | $75.54\pm15.26$ | $90.98\pm22.09$ |
| $K_u$ ($\mathrm{J/m^3}$) | RMSE | $103.73\pm23.06$ | $120.37\pm22.14$ |
| $K_u$ ($\mathrm{J/m^3}$) | mean absolute relative error | $0.939\%\pm0.242\%$ | $1.149\%\pm0.199\%$ |

## 4. Label-transform comparison

- (val 154) The paired improvement rate is defined as $100\%\times\frac{\mathrm{RMSE}_{\mathrm{identity}}-\mathrm{RMSE}_{\mathrm{logalpha}}}{\mathrm{RMSE}_{\mathrm{identity}}}$, where a positive value means logalpha is better. $\alpha$ improves in 4/5 pairs, with a mean per-seed improvement rate of $+10.99\%$; $K_u$ improves in 2/5 pairs, with a mean of $-4.55\%$.
- $\alpha$ log transform: logalpha shows an improving trend in the mean error for $\alpha$, but there is variation across seeds; the benefit for $K_u$ is inconsistent. This is not sufficient to claim that it is overall better or that the differences are statistically significant.
