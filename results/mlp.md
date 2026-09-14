# MLP 基线

## 1. 定位

- 目标：由单激励磁化平均轨迹反演 Gilbert 阻尼系数 $\alpha$ 与单轴各向异性常数 $K_u$。
- 数据：$0\,\mathrm{K}$、无噪声、均匀有效介质、CoFeB 启发的**合成基准**；描述的是固定离散协议，不是具体器件复现。
- 动力学：LLG，能量含交换 $E_{\mathrm{ex}}$、单轴各向异性 $E_{\mathrm{ani}}$、退磁 $E_{\mathrm{demag}}$ 与 Zeeman $E_{\mathrm{Z}}$。
- 模型：以平均轨迹为输入的 MLP 回归器；结果限于该合成基准的验证集，不代表真实器件反演性能。

## 2. 物理模型

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
- 范围：$\alpha \in [0.004, 0.020]$；$K_u \in [2000, 30000]\,\mathrm{J/m^3}$。主域不含 $K_u=0$ 控制点；独立四角诊断点不混入本数据集成员。
- 划分：按 parameter_set_id、split seed 42，固定 717 / 154 / 153（train/val/test）；10 次训练共用该冻结划分。

## 4. 模型与训练

- 输入 $1\times401\times3$（$m_x,m_y,m_z$），展平 $1203 \to 64 \to 32 \to 32 \to 2$；隐层 ReLU；$80258$ 参数。
- 标签条件：`identity` 用原始 $\alpha$/$K_u$；`logalpha` 对 $\alpha$ 取 $\log_{10}$、$K_u$ 保持原值；随后各自逐输出 z-score。
- 标准化统计量仅由训练组拟合；npz 中的标签不做提前变换。
- 优化：Adam，lr $1\times10^{-3}$、batch 32、weight_decay 0；每个标签 5 次运行（seed $42\text{--}46$）。
- 上限 500 epoch，早停 patience 50、min_delta 0；按验证 loss 选择 best。损失为标准化两输出等权 MSE。
- 10 次运行均正常，实际训练轮数 $115\text{--}428$。

## 5. 验证集结果（val 154；$\mathrm{mean}\pm\mathrm{sample\ SD}$，$\mathrm{ddof}=1$，$n=5$）

| 输出 | 指标 | identity | logalpha |
|---|---|---|---|
| $\alpha$ | MAE | $6.456\times10^{-5}\pm1.18\times10^{-5}$ | $5.121\times10^{-5}\pm1.10\times10^{-5}$ |
| $\alpha$ | RMSE | $9.921\times10^{-5}\pm2.00\times10^{-5}$ | $8.715\times10^{-5}\pm1.45\times10^{-5}$ |
| $\alpha$ | 平均绝对相对误差 | $0.759\%\pm0.149\%$ | $0.570\%\pm0.135\%$ |
| $K_u$ ($\mathrm{J/m^3}$) | MAE | $81.23\pm19.49$ | $94.05\pm24.34$ |
| $K_u$ ($\mathrm{J/m^3}$) | RMSE | $140.71\pm23.02$ | $146.01\pm32.43$ |
| $K_u$ ($\mathrm{J/m^3}$) | 平均绝对相对误差 | $1.119\%\pm0.259\%$ | $1.183\%\pm0.251\%$ |

- 配对改善率定义为 $100\%\times\frac{\mathrm{RMSE}_{\mathrm{identity}}-\mathrm{RMSE}_{\mathrm{logalpha}}}{\mathrm{RMSE}_{\mathrm{identity}}}$，正值表示 logalpha 更好。$\alpha$ 在 4/5 对中改善，逐 seed 改善率均值为 $+10.99\%$；$K_u$ 在 2/5 对中改善，均值为 $-4.55\%$。
- $\alpha$ log取值：logalpha 对 $\alpha$ 的平均误差有改善趋势，但 seed 间存在波动；对 $K_u$ 的收益不一致。不足以宣称整体更优或差异具有统计显著性。
