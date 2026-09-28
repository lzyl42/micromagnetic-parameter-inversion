# 模型比较

## 1. 统一比较口径

- 基准：同一固定离散的 CoFeB 启发**合成基准**（$0\,\mathrm{K}$、无噪声、均匀有效介质），不是具体器件复现。
- 数据与划分：1024 个 scrambled Sobol 点，$\alpha = 0.004\cdot 5^u$（对数）、$K_u = 2000 + 28000\,v$（线性）；按 parameter_set_id、split seed 42 固定 717 / 154 / 153（train/val/test）。**val 154 与 test 153 分开报告，两个 split 不混合、不跨 split 平均。**
- 本文件为三模型（MLP / CNN1D / Transformer）× `identity` / `logalpha` × seeds $42$–$46$ 的 **30 个 `best.pt` 冻结**后统一评估；评估前一次性锁定，未依 test 更换 checkpoint 或挑选 seed。test 为本轮首次评估（已揭盲），此后不得据 test 回流调参或挑 seed。
- 随机性：每个（模型，标签）组合 5 次运行，seed $42$–$46$；跨模型比较与标签变换比较均按 seed 配对。
- 标签条件：`identity` 用原始 $\alpha$/$K_u$；`logalpha` 仅对 $\alpha$ 取 $\log_{10}$、$K_u$ 保持原值；随后各自逐输出 z-score，标准化统计仅由训练组拟合。
- 指标：物理单位的 MAE、RMSE 与平均绝对相对误差；汇总为 $\mathrm{mean}\pm\mathrm{sample\ SD}$（$\mathrm{ddof}=1$，$n=5$）。全部 12 个模型–标签–split 组合的六项指标见 [final_evaluation.md](final_evaluation.md)。
- 同标签下的跨模型比较、同模型内的标签变换比较均按 seed 配对；不跨标签直接比较标准化 loss。
- 历史说明：本文件此前仅含 val 154 比较（[mlp.md](mlp.md)、[cnn.md](cnn.md) 亦为历史 val-only 报告，正文未改动）；本次冻结评估新增 test 153，以下按 split 分表。

## 2. 模型汇总

- 参数量：MLP $80258$、CNN1D $4930$、Transformer $69474$；参数量仅记录，不用于推断速度或效率。结构与训练配置见 [final_evaluation.md](final_evaluation.md) 与各模型报告（[mlp.md](mlp.md)、[cnn.md](cnn.md)、[transformer.md](transformer.md)）。

### 2.1 val 154

| 模型 | 标签 | $\alpha$ RMSE | $K_u$ RMSE ($\mathrm{J/m^3}$) | $\alpha$ MAPE% | $K_u$ MAPE% |
|---|---|---|---|---|---|
| MLP | identity | $9.921\times10^{-5}\pm2.00\times10^{-5}$ | $140.71\pm23.02$ | $0.759\%\pm0.149\%$ | $1.119\%\pm0.259\%$ |
| MLP | logalpha | $8.715\times10^{-5}\pm1.45\times10^{-5}$ | $146.01\pm32.43$ | $0.570\%\pm0.135\%$ | $1.183\%\pm0.251\%$ |
| CNN1D | identity | $5.867\times10^{-5}\pm1.33\times10^{-5}$ | $102.68\pm11.19$ | $0.478\%\pm0.113\%$ | $0.752\%\pm0.080\%$ |
| CNN1D | logalpha | $5.726\times10^{-5}\pm1.25\times10^{-5}$ | $98.65\pm29.62$ | $0.288\%\pm0.070\%$ | $0.708\%\pm0.174\%$ |
| Transformer | identity | $2.833\times10^{-4}\pm7.68\times10^{-5}$ | $577.15\pm85.89$ | $2.905\%\pm1.133\%$ | $5.005\%\pm1.048\%$ |
| Transformer | logalpha | $2.621\times10^{-4}\pm9.08\times10^{-5}$ | $556.05\pm121.97$ | $2.323\%\pm0.552\%$ | $4.741\%\pm0.980\%$ |

### 2.2 test 153

| 模型 | 标签 | $\alpha$ RMSE | $K_u$ RMSE ($\mathrm{J/m^3}$) | $\alpha$ MAPE% | $K_u$ MAPE% |
|---|---|---|---|---|---|
| MLP | identity | $8.626\times10^{-5}\pm1.65\times10^{-5}$ | $103.73\pm23.06$ | $0.762\%\pm0.165\%$ | $0.939\%\pm0.242\%$ |
| MLP | logalpha | $7.386\times10^{-5}\pm1.37\times10^{-5}$ | $120.37\pm22.14$ | $0.587\%\pm0.131\%$ | $1.149\%\pm0.199\%$ |
| CNN1D | identity | $5.690\times10^{-5}\pm1.61\times10^{-5}$ | $92.69\pm30.08$ | $0.490\%\pm0.114\%$ | $0.776\%\pm0.213\%$ |
| CNN1D | logalpha | $3.862\times10^{-5}\pm1.39\times10^{-5}$ | $80.74\pm26.27$ | $0.268\%\pm0.079\%$ | $0.694\%\pm0.217\%$ |
| Transformer | identity | $2.762\times10^{-4}\pm6.46\times10^{-5}$ | $600.36\pm89.44$ | $2.826\%\pm1.145\%$ | $5.887\%\pm1.263\%$ |
| Transformer | logalpha | $2.602\times10^{-4}\pm9.23\times10^{-5}$ | $584.70\pm118.38$ | $2.247\%\pm0.579\%$ | $5.615\%\pm1.236\%$ |

## 3. 同标签架构间逐 seed 配对（val 154）

- 改善率 = $100\%\times\frac{\mathrm{baseline\ error}-\mathrm{candidate\ error}}{\mathrm{baseline\ error}}$；正值表示候选误差更低，负值表示候选更差。该式使用逐 seed 的相对差，不是组均值之比；$R^2$ 等非误差指标不套用此式。
- 下表只列配对改善率；RMSE 组均值见第 2 节。

| baseline → candidate | 标签 | 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善/5 |
|---|---|---|---|---|
| MLP → CNN1D | identity | $\alpha$ | $+37.17\%\pm25.13\%$ | 5/5 |
| MLP → CNN1D | identity | $K_u$ | $+26.09\%\pm10.44\%$ | 5/5 |
| MLP → CNN1D | logalpha | $\alpha$ | $+33.33\%\pm16.12\%$ | 5/5 |
| MLP → CNN1D | logalpha | $K_u$ | $+28.73\%\pm31.75\%$ | 4/5 |
| MLP → Transformer | identity | $\alpha$ | $-191.75\%\pm86.80\%$ | 0/5 |
| MLP → Transformer | identity | $K_u$ | $-313.83\%\pm60.49\%$ | 0/5 |
| MLP → Transformer | logalpha | $\alpha$ | $-210.64\%\pm130.94\%$ | 0/5 |
| MLP → Transformer | logalpha | $K_u$ | $-291.98\%\pm100.51\%$ | 0/5 |
| CNN1D → Transformer | identity | $\alpha$ | $-407.32\%\pm176.62\%$ | 0/5 |
| CNN1D → Transformer | identity | $K_u$ | $-460.48\%\pm38.23\%$ | 0/5 |
| CNN1D → Transformer | logalpha | $\alpha$ | $-377.92\%\pm185.95\%$ | 0/5 |
| CNN1D → Transformer | logalpha | $K_u$ | $-521.37\%\pm286.37\%$ | 0/5 |

## 4. 同标签架构间逐 seed 配对（test 153）

- 定义与第 3 节相同；test 为本轮首次评估（已揭盲），结果不得回流用于调参或挑选 seed。

| baseline → candidate | 标签 | 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善/5 |
|---|---|---|---|---|
| MLP → CNN1D | identity | $\alpha$ | $+30.57\%\pm31.40\%$ | 4/5 |
| MLP → CNN1D | identity | $K_u$ | $+7.97\%\pm37.70\%$ | 4/5 |
| MLP → CNN1D | logalpha | $\alpha$ | $+47.47\%\pm15.12\%$ | 5/5 |
| MLP → CNN1D | logalpha | $K_u$ | $+31.77\%\pm21.73\%$ | 5/5 |
| MLP → Transformer | identity | $\alpha$ | $-224.22\%\pm76.82\%$ | 0/5 |
| MLP → Transformer | identity | $K_u$ | $-491.67\%\pm111.16\%$ | 0/5 |
| MLP → Transformer | logalpha | $\alpha$ | $-271.65\%\pm183.33\%$ | 0/5 |
| MLP → Transformer | logalpha | $K_u$ | $-399.19\%\pm145.69\%$ | 0/5 |
| CNN1D → Transformer | identity | $\alpha$ | $-426.53\%\pm196.18\%$ | 0/5 |
| CNN1D → Transformer | identity | $K_u$ | $-587.25\%\pm176.47\%$ | 0/5 |
| CNN1D → Transformer | logalpha | $\alpha$ | $-630.59\%\pm306.32\%$ | 0/5 |
| CNN1D → Transformer | logalpha | $K_u$ | $-693.27\%\pm294.02\%$ | 0/5 |

## 5. 标签效应与边界

- 同架构 identity → logalpha 的逐 seed 标签效应：Transformer 的 val/test 分别见 [transformer.md](transformer.md)；MLP/CNN 的历史 val 分析见 [mlp.md](mlp.md)、[cnn.md](cnn.md)，这两份历史报告不含 test 标签效应，不能将其 val 结论用于 test。
- 须区分两类配对：第 3/4 节是「同标签下架构间」配对，标签变换属于「同架构下标签间」配对，不可混用。
- 范围限定：固定单一冻结 split；5 个 seed 共用同一 split（非 5 个独立数据划分），seed 编号配对不代表随机流完全相同。差异不构成统计显著性结论，不能推广为某模型普遍更优或真实器件结论；test 仅支持该固定合成协议的同分布 held-out 表现，不证明噪声工况、外推区间、多激励迁移或物理正向回代。本次冻结评估的总体结论与执行说明见 [final_evaluation.md](final_evaluation.md)。
- 训练上限与早停规则一致（最多 500 epoch、patience 50、min_delta 0），实际停止轮数不同（MLP 115–428、CNN 208–500、Transformer 84–242）；不能据此认定所有运行均已充分收敛。
