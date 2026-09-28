# Temporal Transformer 基线

## 1. 定位

- 目标：由单激励磁化平均轨迹反演 Gilbert 阻尼系数 $\alpha$ 与单轴各向异性常数 $K_u$。
- 数据：$0\,\mathrm{K}$、无噪声、均匀有效介质、CoFeB 启发的**合成基准**；描述的是固定离散协议，不是具体器件复现。
- 模型：以时间–脉冲–分量序列为输入的 Temporal Transformer 回归器。
- 报告量：冻结划分上的 val 154 与 test 153 物理单位 MAE、RMSE 与平均绝对相对误差；test 为本轮首次评估（已揭盲）。
- 数据、冻结规则与三模型完整汇总见 [final_evaluation.md](final_evaluation.md)；两基线见 [mlp.md](mlp.md)、[cnn.md](cnn.md)。

## 2. 物理模型

与两基线共用同一固定合成基准（事实复用 [mlp.md](mlp.md)、[cnn.md](cnn.md)）：

| 项 | 值 |
|---|---|
| $M_s$ | $1.25\times10^6\,\mathrm{A/m}$ |
| $A_{\mathrm{ex}}$ | $15\,\mathrm{pJ/m}$ |
| 几何 | 真三轴椭球全直径 $100\times50\times2\,\mathrm{nm}$，cells $40\times20\times4$（$2.5\times2.5\times0.5\,\mathrm{nm}$） |
| 初始磁化 / 易轴 | $+x$ / 单轴 $+x$ |
| 激励 | pulse_A2：$y$ 方向 $2\,\mathrm{mT}$、$50\,\mathrm{ps}$ |
| 记录 | 关场后 $0\text{--}4\,\mathrm{ns}$，$10\,\mathrm{ps}\times401$ 点 |
| 数值 | ES12、solver 5、MaxErr $1\times10^{-5}$、MaxDt $10\,\mathrm{ps}$、GammaLL $1.7595\times10^{11}\,\mathrm{rad/(T\cdot s)}$、RelaxTorqueThreshold $-1$ |

## 3. 采样与划分

- 采样：1024 个 scrambled Sobol 点（`rng=42`，`random_base2(10)`）；$\alpha = 0.004\cdot 5^u$（对数），$K_u = 2000 + 28000\,v$（线性）。
- 范围：$\alpha \in [0.004, 0.020]$；$K_u \in [2000, 30000]\,\mathrm{J/m^3}$。主域不含 $K_u=0$ 控制点。
- 划分：按 parameter_set_id、split seed 42，固定 717 / 154 / 153（train/val/test）；两个标签条件共用该冻结划分，逐 seed 一一对应。

## 4. 模型与训练

- 输入布局：$[B, P, T, 3] \to [B, T, 3P]$（token 语义 time-pulse-component；P1、T401 时为 $[B, 401, 3]$）。
- 结构：$d_{\mathrm{model}}=64$、4 头、2 层、FFN 128、dropout 0.1、head 32（GELU 激活）；固定 sinusoidal 位置编码（base 10000）；pre-LN + final-LN；mean pooling；共 $69474$ 参数。
- 标签条件：`identity` 用原始 $\alpha$/$K_u$；`logalpha` 仅对 $\alpha$ 取 $\log_{10}$、$K_u$ 保持原值；两种条件随后都对两个输出做 **train-only** z-score。
- 优化：Adam，lr $1\times10^{-3}$、batch 32、weight_decay 0；每个标签 5 次运行（seed $42$–$46$）；identity seed42 为复用 run，其余为本轮多 seed 批次新跑，本轮冻结评估未再训练。
- 上限 500 epoch，早停 patience 50、min_delta 0；按标准化两输出等权验证 MSE 选择 best。
- 实际训练轮数 $84$–$242$，10 次全部触发早停；损失为标准化两输出等权 MSE。

## 5. val 结果（154）

$\mathrm{mean}\pm\mathrm{sample\ SD}$（$\mathrm{ddof}=1$，$n=5$）；$\alpha$ 无量纲、$K_u$ 单位 $\mathrm{J/m^3}$、MAPE 为百分数。

| 输出 | 指标 | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $2.381\times10^{-4}\pm7.28\times10^{-5}$ | $2.107\times10^{-4}\pm6.02\times10^{-5}$ |
| $\alpha$ | RMSE | $2.833\times10^{-4}\pm7.68\times10^{-5}$ | $2.621\times10^{-4}\pm9.08\times10^{-5}$ |
| $\alpha$ | MAPE% | $2.905\%\pm1.133\%$ | $2.323\%\pm0.552\%$ |
| $K_u$ | MAE | $481.93\pm91.78$ | $459.00\pm97.23$ |
| $K_u$ | RMSE | $577.15\pm85.89$ | $556.05\pm121.97$ |
| $K_u$ | MAPE% | $5.005\%\pm1.048\%$ | $4.741\%\pm0.980\%$ |

标签效应（同架构 identity → logalpha，逐 seed 配对，改善 = $100\%\times(\mathrm{identity}-\mathrm{logalpha})/\mathrm{identity}$；正值表示 logalpha 更低）：

| 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善/5 |
|---|---|---|
| $\alpha$ RMSE | $+2.06\%\pm37.82\%$ | 2/5 |
| $K_u$ RMSE | $+2.80\%\pm20.43\%$ | 3/5 |

- val 参与 best 选择，带 best-checkpoint 选择偏差。

## 6. test 结果（153）

| 输出 | 指标 | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $2.315\times10^{-4}\pm6.95\times10^{-5}$ | $2.060\times10^{-4}\pm5.98\times10^{-5}$ |
| $\alpha$ | RMSE | $2.762\times10^{-4}\pm6.46\times10^{-5}$ | $2.602\times10^{-4}\pm9.23\times10^{-5}$ |
| $\alpha$ | MAPE% | $2.826\%\pm1.145\%$ | $2.247\%\pm0.579\%$ |
| $K_u$ | MAE | $501.23\pm97.69$ | $475.79\pm96.85$ |
| $K_u$ | RMSE | $600.36\pm89.44$ | $584.70\pm118.38$ |
| $K_u$ | MAPE% | $5.887\%\pm1.263\%$ | $5.615\%\pm1.236\%$ |

标签效应（同上定义）：

| 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善/5 |
|---|---|---|
| $\alpha$ RMSE | $+1.08\%\pm39.36\%$ | 3/5 |
| $K_u$ RMSE | $+1.41\%\pm20.70\%$ | 2/5 |

- test 为本轮首次评估；此后依据 test 调整模型或超参，则不得再声称其为未见过的独立 test。

## 7. 标签效应与边界

- 标签效应随输出与 seed 波动：$\alpha$、$K_u$ 的 RMSE 逐 seed 改善均值在 val/test 都接近 0，改善数 2/5–3/5，不足以宣称 `logalpha` 稳定或显著更优；不得沿用历史 val 标签效应作为 test 结论。
- 范围限定：固定单一冻结 split；5 个 seed 共用同一 split（非 5 个独立数据划分），seed 编号配对不代表随机流完全相同。结论仅支持当前固定合成协议的同分布 held-out 表现，不证明真实器件、噪声工况、外推区间、多激励迁移、物理正向回代或统计显著性（完整边界说明见 [final_evaluation.md](final_evaluation.md)）。
- 交叉链接：[MLP 基线](mlp.md)、[CNN1D 基线](cnn.md)、[模型比较](compare.md)。
