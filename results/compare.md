# 模型比较

## 1. 统一比较口径

- 基准：同一固定离散的 CoFeB 启发**合成基准**（$0\,\mathrm{K}$、无噪声、均匀有效介质），不是具体器件复现。
- 数据与划分：1024 个 scrambled Sobol 点，$\alpha = 0.004\cdot 5^u$（对数）、$K_u = 2000 + 28000\,v$（线性）；按 parameter_set_id、split seed 42 固定 717 / 154 / 153（train/val/test）。比较只使用 **val 154**。
- 随机性：每个（模型，标签）组合 5 次运行，seed $42$–$46$。
- 标签条件：`identity` 用原始 $\alpha$/$K_u$；`logalpha` 仅对 $\alpha$ 取 $\log_{10}$、$K_u$ 保持原值；随后各自逐输出 z-score，标准化统计仅由训练组拟合。
- 指标：物理单位的 MAE、RMSE 与平均绝对相对误差；汇总为 $\mathrm{mean}\pm\mathrm{sample\ SD}$（$\mathrm{ddof}=1$，$n=5$）。
- 同标签下的跨模型比较、同模型内的标签变换比较均按 seed 配对；不跨标签直接比较标准化 loss。

## 2. 模型汇总

- MLP 参数 $80258$（[MLP 基线](mlp.md)）；CNN 参数 $4930$（[CNN1D 基线](cnn.md)）。参数量仅记录，不用于推断速度或效率。

| 模型 | 标签 | $\alpha$ RMSE | $K_u$ RMSE ($\mathrm{J/m^3}$) | $\alpha$ 平均绝对相对误差 | $K_u$ 平均绝对相对误差 |
|---|---|---|---|---|---|
| MLP | identity | $9.921\times10^{-5}\pm2.00\times10^{-5}$ | $140.71\pm23.02$ | $0.759\%\pm0.149\%$ | $1.119\%\pm0.259\%$ |
| MLP | logalpha | $8.715\times10^{-5}\pm1.45\times10^{-5}$ | $146.01\pm32.43$ | $0.570\%\pm0.135\%$ | $1.183\%\pm0.251\%$ |
| CNN | identity | $5.867\times10^{-5}\pm1.33\times10^{-5}$ | $102.68\pm11.19$ | $0.478\%\pm0.113\%$ | $0.752\%\pm0.080\%$ |
| CNN | logalpha | $5.726\times10^{-5}\pm1.25\times10^{-5}$ | $98.65\pm29.62$ | $0.288\%\pm0.070\%$ | $0.708\%\pm0.174\%$ |

## 3. 与 MLP 基线逐 seed 配对

- 对候选模型 $m$ 与 MLP，逐 seed 配对改善率定义为 $100\%\times\frac{\mathrm{RMSE}_{\mathrm{MLP},s}-\mathrm{RMSE}_{m,s}}{\mathrm{RMSE}_{\mathrm{MLP},s}}$，正值表示候选模型误差更低。
- 该式使用逐 seed 的相对差，不是组均值之比；$R^2$ 等非误差指标不套用此式。下表只列配对改善率，RMSE 组均值见第 2 节。

| 模型 | 标签 | 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善 |
|---|---|---|---|---|
| CNN | identity | $\alpha$ | $+37.17\%\pm25.13\%$ | 5/5 |
| CNN | identity | $K_u$ | $+26.09\%\pm10.44\%$ | 5/5 |
| CNN | logalpha | $\alpha$ | $+33.33\%\pm16.12\%$ | 5/5 |
| CNN | logalpha | $K_u$ | $+28.73\%\pm31.75\%$ | 4/5 |

## 4. 标签效应与边界

- 跨架构看同一标签效应（identity → logalpha，正值表示 logalpha 更好，逐 seed 配对）：
  - $\alpha$ RMSE：MLP 4/5、均值 $+10.99\%$；CNN 2/5、均值 $-2.14\%$，方向与稳定性不同。
  - $K_u$ RMSE：MLP 2/5、均值 $-4.55\%$；CNN 3/5、均值 $+4.78\%$，两者都属随 seed 波动的收益。
- CNN 自身的标签效应细节（`logalpha` 的 $\alpha$ 平均绝对相对误差 5/5 改善、MAE 4/5 改善）见 [CNN1D 基线](cnn.md) 第 5 节。
- 须区分两类结论：第 3 节是「同标签下架构间」配对，本节是「同架构下标签间」配对，不可混用。
- 范围限定：固定单一 split、5 个 seed、val 154；test 未用于本比较，也无 MuMax3 正向回代。以上差异不构成统计显著性或独立测试集泛化结论，不能推广为某模型普遍更优。
- 训练上限与早停规则一致（最多 500 epoch、patience 50、min_delta 0），实际停止轮数不同。CNN 的 `logalpha` seed42 达到 500 epoch 上限，其余运行触发早停，不能据此认定所有运行均已充分收敛。
