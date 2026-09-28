# 模型比较

## 1. 统一比较口径

- 基准：同一固定离散的 CoFeB 启发**合成基准**（$0\,\mathrm{K}$、无噪声、均匀有效介质），不是具体器件复现。
- 数据与划分：1024 个 scrambled Sobol 点，$\alpha = 0.004\cdot 5^u$（对数）、$K_u = 2000 + 28000\,v$（线性）；按 parameter_set_id、split seed 42 固定 717 / 154 / 153（train/val/test）。**val 154 与 test 153 分开报告，两个 split 不混合、不跨 split 平均。**
- 本文件为三模型（MLP / CNN1D / Transformer）× `identity` / `logalpha` × seeds $42$–$46$ 的 **30 个 `best.pt` 冻结**后统一评估：MLP/CNN 的 20 个沿用上一轮冻结批次（结果不变），Transformer 的 10 个为本版本 dropout=0 版本；评估前一次性锁定，未依 test 更换 checkpoint 或挑选 seed。
- 版本口径：本版本统一采用 dropout=0；调整依据为 train/val 诊断与单因素实验，未使用 test 指标选参。相同测试划分此前曾用于旧版本评估。
- 随机性：每个（模型，标签）组合 5 次运行，seed $42$–$46$；跨模型比较与标签变换比较均按 seed 配对。
- 标签条件：`identity` 用原始 $\alpha$/$K_u$；`logalpha` 仅对 $\alpha$ 取 $\log_{10}$、$K_u$ 保持原值；随后各自逐输出 z-score，标准化统计仅由训练组拟合。
- 指标：物理单位的 MAE、RMSE 与平均绝对相对误差；汇总为 $\mathrm{mean}\pm\mathrm{sample\ SD}$（$\mathrm{ddof}=1$，$n=5$）。全部 12 个模型–标签–split 组合的六项指标见第 2 节 val/test 表。
- 同标签下的跨模型比较、同模型内的标签变换比较均按 seed 配对；不跨标签直接比较标准化 loss。
- 历史说明：本文件此前仅含 val 154 比较（[mlp.md](mlp.md)、[cnn.md](cnn.md) 亦为历史 val-only 报告，正文未改动）；其后冻结评估加入 test 153；本版本以 dropout=0 的 Transformer 替换旧 dropout=0.1 的 run（旧产物保留为历史参考、不参与正文数值），MLP/CNN 结果不变，正文只展示当前版本。

## 2. 模型汇总

- 参数量：MLP $80258$、CNN1D $4930$、Transformer $69474$（dropout=0 不改变参数量）；参数量仅记录，不用于推断速度或效率。结构与训练配置见各模型报告（[mlp.md](mlp.md)、[cnn.md](cnn.md)、[transformer.md](transformer.md)）。

### 2.1 val 154

| 模型 | 标签 | $\alpha$ MAE | $\alpha$ RMSE | $K_u$ MAE ($\mathrm{J/m^3}$) | $K_u$ RMSE ($\mathrm{J/m^3}$) | $\alpha$ MAPE% | $K_u$ MAPE% |
|---|---|---|---|---|---|---|---|
| MLP | identity | $6.456\times10^{-5}\pm1.18\times10^{-5}$ | $9.921\times10^{-5}\pm2.00\times10^{-5}$ | $81.23\pm19.49$ | $140.71\pm23.02$ | $0.759\%\pm0.149\%$ | $1.119\%\pm0.259\%$ |
| MLP | logalpha | $5.121\times10^{-5}\pm1.10\times10^{-5}$ | $8.715\times10^{-5}\pm1.45\times10^{-5}$ | $94.05\pm24.34$ | $146.01\pm32.43$ | $0.570\%\pm0.135\%$ | $1.183\%\pm0.251\%$ |
| CNN1D | identity | $4.134\times10^{-5}\pm1.08\times10^{-5}$ | $5.867\times10^{-5}\pm1.33\times10^{-5}$ | $66.12\pm13.74$ | $102.68\pm11.19$ | $0.478\%\pm0.113\%$ | $0.752\%\pm0.080\%$ |
| CNN1D | logalpha | $3.024\times10^{-5}\pm8.43\times10^{-6}$ | $5.726\times10^{-5}\pm1.25\times10^{-5}$ | $60.06\pm20.71$ | $98.65\pm29.62$ | $0.288\%\pm0.070\%$ | $0.708\%\pm0.174\%$ |
| Transformer | identity | $2.292\times10^{-5}\pm3.59\times10^{-6}$ | $5.453\times10^{-5}\pm1.42\times10^{-5}$ | $49.04\pm8.04$ | $89.57\pm18.13$ | $0.246\%\pm0.040\%$ | $0.725\%\pm0.202\%$ |
| Transformer | logalpha | $2.121\times10^{-5}\pm4.50\times10^{-6}$ | $5.536\times10^{-5}\pm8.75\times10^{-6}$ | $38.96\pm11.54$ | $64.82\pm8.99$ | $0.196\%\pm0.042\%$ | $0.506\%\pm0.069\%$ |

### 2.2 test 153

| 模型 | 标签 | $\alpha$ MAE | $\alpha$ RMSE | $K_u$ MAE ($\mathrm{J/m^3}$) | $K_u$ RMSE ($\mathrm{J/m^3}$) | $\alpha$ MAPE% | $K_u$ MAPE% |
|---|---|---|---|---|---|---|---|
| MLP | identity | $6.404\times10^{-5}\pm1.29\times10^{-5}$ | $8.626\times10^{-5}\pm1.65\times10^{-5}$ | $75.54\pm15.26$ | $103.73\pm23.06$ | $0.762\%\pm0.165\%$ | $0.939\%\pm0.242\%$ |
| MLP | logalpha | $5.143\times10^{-5}\pm1.32\times10^{-5}$ | $7.386\times10^{-5}\pm1.37\times10^{-5}$ | $90.98\pm22.09$ | $120.37\pm22.14$ | $0.587\%\pm0.131\%$ | $1.149\%\pm0.199\%$ |
| CNN1D | identity | $4.108\times10^{-5}\pm1.18\times10^{-5}$ | $5.690\times10^{-5}\pm1.61\times10^{-5}$ | $65.69\pm21.13$ | $92.69\pm30.08$ | $0.490\%\pm0.114\%$ | $0.776\%\pm0.213\%$ |
| CNN1D | logalpha | $2.588\times10^{-5}\pm9.05\times10^{-6}$ | $3.862\times10^{-5}\pm1.39\times10^{-5}$ | $57.78\pm18.12$ | $80.74\pm26.27$ | $0.268\%\pm0.079\%$ | $0.694\%\pm0.217\%$ |
| Transformer | identity | $2.109\times10^{-5}\pm3.83\times10^{-6}$ | $3.233\times10^{-5}\pm9.68\times10^{-6}$ | $45.25\pm9.22$ | $62.34\pm15.95$ | $0.251\%\pm0.036\%$ | $0.616\%\pm0.196\%$ |
| Transformer | logalpha | $2.074\times10^{-5}\pm2.52\times10^{-6}$ | $3.214\times10^{-5}\pm5.61\times10^{-6}$ | $35.80\pm9.59$ | $48.39\pm9.51$ | $0.209\%\pm0.026\%$ | $0.448\%\pm0.085\%$ |

## 3. 同标签架构间逐 seed 配对（val 154）

- 改善率 = $100\%\times\frac{\mathrm{baseline\ error}-\mathrm{candidate\ error}}{\mathrm{baseline\ error}}$；正值表示候选误差更低，负值表示候选更差。该式使用逐 seed 的相对差，不是组均值之比；$R^2$ 等非误差指标不套用此式。
- 下表只列配对改善率；RMSE 组均值见第 2 节。

| baseline → candidate | 标签 | 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善/5 |
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

## 4. 同标签架构间逐 seed 配对（test 153）

- 定义与第 3 节相同；本版本未使用 test 指标选参，结果不用于回流调参或挑选 seed。

| baseline → candidate | 标签 | 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善/5 |
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

## 5. 标签效应与边界

- 同架构 identity → logalpha 的逐 seed 标签效应：Transformer 的 val/test 见 [transformer.md](transformer.md)；MLP/CNN 的历史 val 分析见 [mlp.md](mlp.md)、[cnn.md](cnn.md)，这两份历史报告不含 test 标签效应，不能将其 val 结论用于 test。
- 须区分两类配对：第 3/4 节是「同标签下架构间」配对，标签变换属于「同架构下标签间」配对，不可混用。
- 描述性比较：本版本 MLP → Transformer 的**配对改善率均值**在 val/test、两标签上均为正（各组合逐 seed 改善数 $n$ 4/5–5/5）；CNN1D → Transformer 的配对改善率均值同样为正但逐 seed 一致性更弱（$n$ 3/5–5/5）。**均值为正不等于每个 seed 都改善**：须结合第 3/4 节「$n$ 改善/5」列阅读，存在单个 seed 上 CNN1D 误差更低的组合，不能据此宣称任一模型普遍更优。按组均值，test 六项指标以 Transformer/logalpha 最低（见第 2.2 节）。
- 范围限定：固定单一冻结 split；5 个 seed 共用同一 split（非 5 个独立数据划分），seed 编号配对不代表随机流完全相同。差异不构成统计显著性结论，不能推广为某模型普遍更优或真实器件结论；test 仅支持该固定合成协议的同分布 held-out 表现，不证明噪声工况、外推区间、多激励迁移或物理正向回代。本版本冻结规则、产物与指纹见第 6 节。
- 训练上限与早停规则一致（最多 500 epoch、patience 50、min_delta 0），实际停止轮数不同（MLP 115–428、CNN 208–500：9 次早停、1 次达上限；Transformer 125–412：10 次全部早停）；不能据此认定所有运行均已充分收敛。

## 6. 可追溯性

- 冻结批次（相对仓库根；位于 ignored 的 `artifacts/`，原始产物不入 Git）：`artifacts/experiments/three_model_dropout0_eval_20260928T082953Z/`。
- `freeze_manifest.json`（SHA256 `31f7578eac84c3d3411454f69e924fceb614eae93de73c5d70eeea2b6a89c87d`）锁定 30 个 `best.pt` 的 SHA 及 `config_resolved`、`preprocessing`、split 副本等指纹。
- 评估代码 HEAD `7ea7d1b679720f956786fa644b890313a67e2d7e`；dataset_meta `b1d99161a7ef1cc72ff802b7bef2419d84428998967d3d2aaa5a2bbf49165659`；split `b49c9dd32ab6cd7c48caf15ebbbadd749d115da35dc9a29c5a80ad8c3e223ad1`（train/val/test = 717/154/153）。
- 数值以批次内 `per_run_metrics.csv`（60 行 = 30 run × val/test）与 `aggregate_metrics.csv`（72 行 = 12 组 × 6 指标）为准；既有产物的当前 SHA 与复用血缘见 `artifact_lineage.json` 与 `verified_artifacts.json`。
- 旧 dropout=0.1 的 Transformer runs 与其冻结评估批次 `artifacts/experiments/three_model_frozen_eval_20260928T063042Z/` 保留为 superseded 历史参考，不参与本版本数值。
