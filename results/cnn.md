# CNN1D 基线

## 1. 定位

- 目标：由单激励磁化平均轨迹反演 Gilbert 阻尼系数 $\alpha$ 与单轴各向异性常数 $K_u$。
- 数据：$0\,\mathrm{K}$、无噪声、均匀有效介质、CoFeB 启发的**合成基准**；描述的是固定离散协议，不是具体器件复现。
- 动力学：LLG，能量含交换 $E_{\mathrm{ex}}$、单轴各向异性 $E_{\mathrm{ani}}$、退磁 $E_{\mathrm{demag}}$ 与 Zeeman $E_{\mathrm{Z}}$。
- 模型：以平均轨迹为输入的 1D CNN 回归器；结果限于该合成基准的**验证集**，不代表真实器件反演性能，也不构成泛化结论。
- 报告量：验证集物理单位的 MAE、RMSE 与平均绝对相对误差；不涉及正向回代或器件级结论。

## 2. 物理模型

固定合成基准：

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

- 采样：1024 个 scrambled Sobol 点（`rng=42`，`random_base2(10)`）；对单位方形内的点 $(u,v)$，$\alpha = 0.004\cdot 5^u$（对数），$K_u = 2000 + 28000\,v$（线性）。
- 范围：$\alpha \in [0.004, 0.020]$；$K_u \in [2000, 30000]\,\mathrm{J/m^3}$。主域不含 $K_u=0$ 控制点。
- 划分：按 parameter_set_id、split seed 42，固定 717 / 154 / 153（train/val/test）；本模型两个标签条件共用该冻结划分。
- 两个标签条件使用同一冻结划分，逐 seed 一一对应。

## 4. 模型与训练

- 输入 $1\times401\times3$（$m_x,m_y,m_z$）；按 pulse 与分量合并为 3 通道、长度 401。
- 结构：两层 Conv1d（通道 $3\to8\to16$，kernel $5/5$，stride 1、dilation 1、padding 2，每层接 ReLU）→ AdaptiveAvgPool1d(16) → Flatten(256) → Linear(16) + ReLU → Linear(2)；共 $4930$ 参数。
- 标签条件：`identity` 用原始 $\alpha$/$K_u$；`logalpha` 仅对 $\alpha$ 取 $\log_{10}$、$K_u$ 保持原值；随后各自逐输出 z-score。
- 标准化统计量仅由训练组拟合。
- 优化：Adam，lr $1\times10^{-3}$、batch 32、weight_decay 0；每个标签 5 次运行（seed $42$–$46$）。
- 上限 500 epoch，早停 patience 50、min_delta 0；按标准化两输出等权验证 MSE 选择 best。
- 实际训练轮数 $208$–$500$：9 次早停、1 次达到 500 上限。该范围是对结果的描述，不表示各次均已充分收敛。
- 同一标签内的 5 次运行共用同一冻结 split，除 seed 外的结构与超参一致。

## 5. 验证集结果（val 154；$\mathrm{mean}\pm\mathrm{sample\ SD}$，$\mathrm{ddof}=1$，$n=5$）

| 输出 | 指标 | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $4.134\times10^{-5}\pm1.08\times10^{-5}$ | $3.024\times10^{-5}\pm8.43\times10^{-6}$ |
| $\alpha$ | RMSE | $5.867\times10^{-5}\pm1.33\times10^{-5}$ | $5.726\times10^{-5}\pm1.25\times10^{-5}$ |
| $\alpha$ | 平均绝对相对误差 | $0.478\%\pm0.113\%$ | $0.288\%\pm0.070\%$ |
| $K_u$ ($\mathrm{J/m^3}$) | MAE | $66.12\pm13.74$ | $60.06\pm20.71$ |
| $K_u$ ($\mathrm{J/m^3}$) | RMSE | $102.68\pm11.19$ | $98.65\pm29.62$ |
| $K_u$ ($\mathrm{J/m^3}$) | 平均绝对相对误差 | $0.752\%\pm0.080\%$ | $0.708\%\pm0.174\%$ |

- 标签变换的配对改善率定义为 $100\%\times(E_{\mathrm{identity},s}-E_{\mathrm{logalpha},s})/E_{\mathrm{identity},s}$，其中 $E$ 为相应误差指标；正值表示 logalpha 更好，以下均为逐 seed 改善率的均值 ± 样本 SD。
- 标签效应须按指标分开看。$\alpha$ 的 MAE 在 4/5 对中改善；平均绝对相对误差 5/5 对改善（$+36.11\%\pm24.48\%$）。但 $\alpha$ 的 RMSE 仅 2/5 对改善，为 $-2.14\%\pm32.03\%$。
- $K_u$ 的 RMSE 3/5 对改善，均值 $+4.78\%\pm21.54\%$，种子间波动同样明显。
- 因此不能简单地说 `logalpha` 整体更好：$\alpha$ 的 MAE 多数改善、相对误差在全部 seed 上改善，而 RMSE 不稳定；$K_u$ 的收益随 seed 波动。
- $\alpha$ 的组均值 RMSE 由 identity 到 logalpha 略降，与 $\alpha$ 逐 seed 配对相对差均值为负并不矛盾：前者是 RMSE 组均值之差，后者是逐 seed 相对差再平均，口径不同（该说明仅针对 $\alpha$）。

## 6. 范围与延伸

- 范围限定：固定单一 split、5 个 seed、val 154；test 未用于本结果，也无 MuMax3 正向回代。以上不构成统计显著性或泛化结论。
- 跨模型比较（与 MLP 基线及后续模型）见 [模型比较](compare.md)。
