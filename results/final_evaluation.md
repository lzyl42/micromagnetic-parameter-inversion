# 冻结评估最终报告：三模型 × identity/logalpha × seeds 42–46（val 154 / test 153）

- **日期**：2026-09-28
- **范围**：本报告只针对「当前固定离散 CoFeB 启发合成 benchmark（`cofeb_protocol_b_a2_sobol1024_v1`）的冻结 val/test 评估」，不是整个项目的最终科研结论，也不外推到真实器件、其他协议或外推区间。
- **评估对象**：30 个已训练 run = 3 个模型族（MLP / CNN1D / Transformer）× 2 个标签条件（identity / logalpha）× 5 个 seed（42–46），每个 run 取 `best.pt`。
- **数据与协议**（摘自 [mlp.md](mlp.md)、[cnn.md](cnn.md)，本轮未修改这些历史报告）：单激励 pulse_A2（y 方向 2 mT、50 ps），记录关场后 0–4 ns、dt = 10 ps、401 点（P1×T401×3 分量）；几何 100×50×2 nm、40×20×4 cells；Ms = 1.25e6 A/m、Aex = 15 pJ/m；0 K、无噪声、均匀有效介质。参数采样为 1024 个 scrambled Sobol 点（rng 42）：α = 0.004·5^u ∈ [0.004, 0.020]（对数均匀），Ku = 2000 + 28000v ∈ [2000, 30000] J/m³（线性）；主域不含 Ku = 0 控制点。按 parameter_set_id、split seed 42 固定划分 train/val/test = 717/154/153。

## 1. 模型与训练配置

| 模型 | 代码文件 / 类 | 结构 | 参数量 |
|---|---|---|---|
| MLP | `src/micromagnetic_parameter_inversion/models/mlp.py` / `MLPRegressor` | 输入 1×401×3 展平 1203 → 64 → 32 → 32 → 2，隐层 ReLU | 80258 |
| CNN1D | `src/micromagnetic_parameter_inversion/models/cnn1d.py` / `CNN1DRegressor` | Conv1d 3→8→16（kernel 5/5，ReLU）→ AdaptiveAvgPool1d(16) → Flatten(256) → Linear(16)+ReLU → Linear(2) | 4930 |
| Transformer | `src/micromagnetic_parameter_inversion/models/transformer.py` / `TemporalTransformerRegressor` | d64、h4、2 层、FFN128、dropout 0.1、sinusoidal PE、pre-LN + final-LN、mean pooling、head 32、GELU | 69474 |

- 标签条件：`identity` 保留 α/Ku 原始数值，`logalpha` 仅先对 α 取 log10（Ku 不变）；两种条件随后都对两个输出分别进行 train-only z-score。
- 标准化统计量仅由训练组拟合；损失为标准化两输出等权 MSE。
- 共同训练超参：Adam，lr = 1e-3，batch 32，weight_decay = 0；max 500 epoch，early stopping patience 50、min_delta 0；按 val loss 选择 best。实际 epoch：MLP 115–428、CNN 208–500（9 次早停、1 次达上限）、Transformer 84–242（逐 run 值见冻结批次 `per_run_metrics.csv`）。
- 差异与归因边界：三个模型结构、参数规模不同，也未做等参数量或充分调参比较；Transformer 另有 dropout/GELU 等与两基线不同，其差异不能全部归因于 attention。历史训练代码版本不同（各 run checkpoint 记录的 git SHA 不同）；本冻结评估使用的代码 HEAD 为 `5204ed793c50c187a3f4c05f1556be934d530351`，不能称三个模型均在该 SHA 下训练。

## 2. 冻结与评估规则

- 评估前一次性锁定全部 30 个 `best.pt`（按各 run 的 val loss 选出）；未使用 `final.pt`，未依据 test 更换 checkpoint，也未挑选 seed。
- val 参与了所有 run 的模型选择，故 val 指标带 best-checkpoint 选择偏差；test 为**本轮首次评估**（30 个 run 的 test 标准产物均在本轮新增）。val 标准产物：20 个新增（MLP/CNN）、10 个复用（Transformer 既有），复用前均核验 provenance、成员集合与指标重算一致。
- 冻结清单及全部输入（best.pt、config_resolved、preprocessing、split 副本、metrics、dataset_meta、split.yaml）的 SHA 在汇总后复核未变；60/60 份 val/test 产物通过 ckpt SHA / split SHA provenance、成员集合、有限性与指标重算核验。
- test 自此揭盲：若后续再依据 test 调整模型或超参，则不得再声称其为未见过的独立 test；本报告不据此推荐调参或追最佳 seed。

## 3. 结果

- 表示：mean ± sample SD（ddof = 1，n = 5）；α 无量纲，Ku 单位 J/m³，MAPE 为百分数；MAPE = 100·mean(|pred − true|/|true|)。
- 每格是 5 个 seed 的组均值与组内样本 SD，不是置信区间；5 个 seed 共用同一冻结 split，不能视为 5 个独立数据划分。control（Ku = 0）n = 0，不填 0。未跨标签比较标准化 loss。

### 3.1 val（n = 154）

α（MAE / RMSE / MAPE%）：

| 模型 | 标签 | MAE α | RMSE α | MAPE% α |
|---|---|---|---|---|
| mlp | identity | 6.456e-05 ± 1.183e-05 | 9.921e-05 ± 2.002e-05 | 0.759 ± 0.149 |
| mlp | logalpha | 5.121e-05 ± 1.096e-05 | 8.715e-05 ± 1.451e-05 | 0.570 ± 0.135 |
| cnn | identity | 4.134e-05 ± 1.080e-05 | 5.867e-05 ± 1.329e-05 | 0.478 ± 0.113 |
| cnn | logalpha | 3.024e-05 ± 8.432e-06 | 5.726e-05 ± 1.253e-05 | 0.288 ± 0.070 |
| transformer | identity | 2.381e-04 ± 7.278e-05 | 2.833e-04 ± 7.677e-05 | 2.905 ± 1.133 |
| transformer | logalpha | 2.107e-04 ± 6.024e-05 | 2.621e-04 ± 9.075e-05 | 2.323 ± 0.552 |

Ku（MAE / RMSE / MAPE%，Ku 单位 J/m³）：

| 模型 | 标签 | MAE Ku | RMSE Ku | MAPE% Ku |
|---|---|---|---|---|
| mlp | identity | 81.23 ± 19.49 | 140.71 ± 23.02 | 1.119 ± 0.259 |
| mlp | logalpha | 94.05 ± 24.34 | 146.01 ± 32.43 | 1.183 ± 0.251 |
| cnn | identity | 66.12 ± 13.74 | 102.68 ± 11.19 | 0.752 ± 0.080 |
| cnn | logalpha | 60.06 ± 20.71 | 98.65 ± 29.62 | 0.708 ± 0.174 |
| transformer | identity | 481.93 ± 91.78 | 577.15 ± 85.89 | 5.005 ± 1.048 |
| transformer | logalpha | 459.00 ± 97.23 | 556.05 ± 121.97 | 4.741 ± 0.980 |

### 3.2 test（n = 153）

α（MAE / RMSE / MAPE%）：

| 模型 | 标签 | MAE α | RMSE α | MAPE% α |
|---|---|---|---|---|
| mlp | identity | 6.404e-05 ± 1.286e-05 | 8.626e-05 ± 1.646e-05 | 0.762 ± 0.165 |
| mlp | logalpha | 5.143e-05 ± 1.317e-05 | 7.386e-05 ± 1.372e-05 | 0.587 ± 0.131 |
| cnn | identity | 4.108e-05 ± 1.176e-05 | 5.690e-05 ± 1.605e-05 | 0.490 ± 0.114 |
| cnn | logalpha | 2.588e-05 ± 9.051e-06 | 3.862e-05 ± 1.386e-05 | 0.268 ± 0.079 |
| transformer | identity | 2.315e-04 ± 6.950e-05 | 2.762e-04 ± 6.458e-05 | 2.826 ± 1.145 |
| transformer | logalpha | 2.060e-04 ± 5.982e-05 | 2.602e-04 ± 9.226e-05 | 2.247 ± 0.579 |

Ku（MAE / RMSE / MAPE%，Ku 单位 J/m³）：

| 模型 | 标签 | MAE Ku | RMSE Ku | MAPE% Ku |
|---|---|---|---|---|
| mlp | identity | 75.54 ± 15.26 | 103.73 ± 23.06 | 0.939 ± 0.242 |
| mlp | logalpha | 90.98 ± 22.09 | 120.37 ± 22.14 | 1.149 ± 0.199 |
| cnn | identity | 65.69 ± 21.13 | 92.69 ± 30.08 | 0.776 ± 0.213 |
| cnn | logalpha | 57.78 ± 18.12 | 80.74 ± 26.27 | 0.694 ± 0.217 |
| transformer | identity | 501.23 ± 97.69 | 600.36 ± 89.44 | 5.887 ± 1.263 |
| transformer | logalpha | 475.79 ± 96.85 | 584.70 ± 118.38 | 5.615 ± 1.236 |

## 4. 结论（限定于本固定合成协议的同分布 held-out 表现）

- 在本次六个冻结（模型，标签）条件中，**CNN/logalpha 的 test 六项指标的组均值均最低**（α MAE 2.588e-05、α RMSE 3.862e-05、α MAPE 0.268%、Ku MAE 57.78、Ku RMSE 80.74、Ku MAPE 0.694%）。
- **Transformer 两个标签条件的六项误差组均值均高于同标签 MLP/CNN**（val 与 test 方向一致）。
- 标签效应随输出与模型而异：MLP 的 logalpha 在 α 上组均值较低、在 Ku 上较高；CNN 的 logalpha 在 test 的两组均值上均较低；Transformer 的 logalpha 略低，但不宣称稳定或显著改善。
- 以上结论仅为当前固定合成协议下的描述性比较：val 用于模型选择，test 提供该协议同分布留出样本的评估证据。它们**不证明**真实器件、外推区间、噪声工况、多激励迁移、物理正向回代或统计显著性；这不是三个模型的普适排名。

## 5. 可追溯性

- 冻结批次（相对仓库根；位于 ignored 的 `artifacts/`，不入 Git）：`artifacts/experiments/three_model_frozen_eval_20260928T063042Z/`
  - `freeze_manifest.json`（SHA256 `07a6035862ea1a893654801533a69143c7113c11251c46f5b331f5ffb1615da0`）
  - `summary.json`、`per_run_metrics.csv`（60 行 = 30 run × val/test）、`aggregate_metrics.csv`（72 行 = 12 组 × 6 指标）、`paired_val_same_seed.csv`、`SUMMARY.md`、`logs/`
- 评估代码 HEAD：`5204ed793c50c187a3f4c05f1556be934d530351`；数据指纹：dataset_meta `b1d99161a7ef1cc72ff802b7bef2419d84428998967d3d2aaa5a2bbf49165659`，split `b49c9dd32ab6cd7c48caf15ebbbadd749d115da35dc9a29c5a80ad8c3e223ad1`（train/val/test = 717/154/153）。
- 历史 val-only 报告（本轮未修改）：[mlp.md](mlp.md)、[cnn.md](cnn.md)、[compare.md](compare.md)。
- 本报告是小型摘要；完整复现需保留上述原始产物（数值以冻结批次 CSV 为准）。

## 6. 执行事件说明

- 首轮执行 shell 的计划文件曾使用 CRLF 行尾，导致 30 个 test 评估被跳过且汇总中断；修正为 LF 行尾后，在同一 1800 s 总预算内补执行 30 个 test 评估并完成汇总。过程中未重训、未更换 checkpoint、未覆盖或重评已有产物；冻结清单与输入 SHA 未变，该事件不影响任何数值结果。
