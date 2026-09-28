# Temporal Transformer 基线

## 1. 定位

- 目标：由单激励磁化平均轨迹反演 Gilbert 阻尼系数 $\alpha$ 与单轴各向异性常数 $K_u$。
- 数据：$0\,\mathrm{K}$、无噪声、均匀有效介质、CoFeB 启发的**合成基准**；描述的是固定离散协议，不是具体器件复现。
- 模型：以时间–脉冲–分量序列为输入的 Temporal Transformer 回归器。
- 报告量：冻结划分上的 val 154 与 test 153 物理单位 MAE、RMSE 与平均绝对相对误差。
- 版本口径：本版本统一采用 dropout=0；调整依据为 train/val 诊断与单因素实验，未使用 test 指标选参。相同测试划分此前曾用于旧版本评估。
- 数据、冻结规则与三模型完整汇总见 [模型比较](compare.md)（统一口径见第 1 节、val/test 完整六指标见第 2 节）；两基线见 [mlp.md](mlp.md)、[cnn.md](cnn.md)。

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
- 结构：$d_{\mathrm{model}}=64$、4 头、2 层、FFN 128、dropout 0、head 32（GELU 激活）；固定 sinusoidal 位置编码（base 10000）；pre-LN + final-LN；mean pooling；共 $69474$ 参数（dropout 调整不改变参数量与其余结构）。
- 标签条件：`identity` 用原始 $\alpha$/$K_u$；`logalpha` 仅对 $\alpha$ 取 $\log_{10}$、$K_u$ 保持原值；两种条件随后都对两个输出做 **train-only** z-score。
- 优化：Adam，lr $1\times10^{-3}$、batch 32、weight_decay 0；每个标签 5 次运行（seed $42$–$46$）。
- run 来源：seed 42 的 identity/logalpha 两次 run 为 dropout 单因素实验批次的 dropout=0 run（复用）；seeds 43–46 为本轮多 seed 批次新跑；本轮冻结评估未再训练。dropout=0 的调整依据为 train/val 诊断与单因素实验，未使用 test 指标。
- 上限 500 epoch，早停 patience 50、min_delta 0；按标准化两输出等权验证 MSE 选择 best。
- 实际训练轮数 $125$–$412$，10 次全部触发早停；损失为标准化两输出等权 MSE。

## 5. val 结果（154）

$\mathrm{mean}\pm\mathrm{sample\ SD}$（$\mathrm{ddof}=1$，$n=5$）；$\alpha$ 无量纲、$K_u$ 单位 $\mathrm{J/m^3}$、MAPE 为百分数。

| 输出 | 指标 | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $2.292\times10^{-5}\pm3.59\times10^{-6}$ | $2.121\times10^{-5}\pm4.50\times10^{-6}$ |
| $\alpha$ | RMSE | $5.453\times10^{-5}\pm1.42\times10^{-5}$ | $5.536\times10^{-5}\pm8.75\times10^{-6}$ |
| $\alpha$ | MAPE% | $0.246\%\pm0.040\%$ | $0.196\%\pm0.042\%$ |
| $K_u$ | MAE | $49.04\pm8.04$ | $38.96\pm11.54$ |
| $K_u$ | RMSE | $89.57\pm18.13$ | $64.82\pm8.99$ |
| $K_u$ | MAPE% | $0.725\%\pm0.202\%$ | $0.506\%\pm0.069\%$ |

标签效应（同架构 identity → logalpha，逐 seed 配对，改善 = $100\%\times(\mathrm{identity}-\mathrm{logalpha})/\mathrm{identity}$；正值表示 logalpha 更低）：

| 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善/5 |
|---|---|---|
| $\alpha$ RMSE | $-5.75\%\pm28.34\%$ | 3/5 |
| $K_u$ RMSE | $+23.66\%\pm27.89\%$ | 4/5 |

- val 参与 best 选择，带 best-checkpoint 选择偏差。

## 6. test 结果（153）

| 输出 | 指标 | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $2.109\times10^{-5}\pm3.83\times10^{-6}$ | $2.074\times10^{-5}\pm2.52\times10^{-6}$ |
| $\alpha$ | RMSE | $3.233\times10^{-5}\pm9.68\times10^{-6}$ | $3.214\times10^{-5}\pm5.61\times10^{-6}$ |
| $\alpha$ | MAPE% | $0.251\%\pm0.036\%$ | $0.209\%\pm0.026\%$ |
| $K_u$ | MAE | $45.25\pm9.22$ | $35.80\pm9.59$ |
| $K_u$ | RMSE | $62.34\pm15.95$ | $48.39\pm9.51$ |
| $K_u$ | MAPE% | $0.616\%\pm0.196\%$ | $0.448\%\pm0.085\%$ |

标签效应（同上定义）：

| 输出 | 逐 seed 改善率 (mean±SD) | $n$ 改善/5 |
|---|---|---|
| $\alpha$ RMSE | $-4.99\%\pm31.77\%$ | 4/5 |
| $K_u$ RMSE | $+16.94\%\pm33.54\%$ | 4/5 |

- 本版本未使用 test 指标进行选择；test 仅用于固定冻结后的评估报告，不用于调参或挑选 seed。

## 7. 标签效应与边界

- $K_u$ RMSE 的逐 seed 改善均值为正（val $+23.66\%\pm27.89\%$、4/5；test $+16.94\%\pm33.54\%$、4/5），而 $\alpha$ RMSE 均值略负（val $-5.75\%\pm28.34\%$、3/5；test $-4.99\%\pm31.77\%$、4/5）。注意 $\alpha$ 上 `logalpha` 的**组均值**与 identity 接近且互有高低（test 三项组均值略低，如 RMSE $3.214\times10^{-5}$ vs $3.233\times10^{-5}$；val MAE/MAPE 略低、α RMSE 略高）：组均值差与逐 seed 相对改善平均是两种不同口径，二者并不矛盾（负向均值主要来自个别 seed 的大幅负向离群）；本报告保留该不稳定口径、不混用两种聚合，也不宣称 `logalpha` 稳定或显著更优。
- 范围限定：固定单一冻结 split；5 个 seed 共用同一 split（非 5 个独立数据划分），seed 编号配对不代表随机流完全相同。结论仅支持当前固定合成协议的同分布 held-out 表现，不证明真实器件、噪声工况、外推区间、多激励迁移、物理正向回代或统计显著性（完整边界说明见 [模型比较](compare.md) 第 5 节）。
- 旧 dropout=0.1 版本的 Transformer 产物保留为历史参考（superseded），其数值不并入本版本结论。
- 交叉链接：[MLP 基线](mlp.md)、[CNN1D 基线](cnn.md)、[模型比较](compare.md)。
