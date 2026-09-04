# 基于多激励时序 Transformer 的微磁材料参数反演研究规划

## 1. 项目名称

**中文题目：**

基于多激励磁化动力学与时序 Transformer 的微磁材料参数反演

**英文题目：**

Inverse Identification of Micromagnetic Material Parameters from Multi-Excitation Magnetization Dynamics Using Temporal Transformers

---

## 2. 研究背景

微磁学使用连续磁化场描述纳米尺度磁性材料中的磁化结构和磁化动力学。

微磁学中的基本变量为归一化磁化方向：

$$
\mathbf{m}(\mathbf{r},t)
=
\frac{\mathbf{M}(\mathbf{r},t)}{M_s},
\qquad
|\mathbf{m}|=1.
$$

其中：

- $\mathbf{M}(\mathbf{r},t)$ 是磁化强度；
- $M_s$ 是饱和磁化强度；
- $\mathbf{m}(\mathbf{r},t)$ 表示磁化方向；
- $\mathbf{r}$ 是空间位置；
- $t$ 是时间。

在常规微磁模拟中，研究者已知材料参数、几何结构和外加磁场，然后通过求解 Landau–Lifshitz–Gilbert 方程得到磁化随时间的变化：

$$
\text{材料参数}
\longrightarrow
\text{LLG 方程}
\longrightarrow
\mathbf{m}(\mathbf{r},t).
$$

但是在实验中，研究者通常只能测量材料受到外场激励后的动态响应，而不能直接获得 Gilbert 阻尼系数、磁各向异性常数等材料参数。

因此，实际材料研究中经常需要解决相反的问题：

$$
\text{磁化动力学响应}
\longrightarrow
\text{材料参数}.
$$

这类问题称为**逆问题**或**参数反演问题**。

本项目计划使用 MuMax3 生成不同材料参数下的磁化动力学数据，再训练时序 Transformer，根据磁化时间序列快速反演材料的 Gilbert 阻尼系数 $\alpha$ 和单轴磁各向异性常数 $K_u$。

---

## 3. 研究目标

本项目的核心目标是：

> 给定一个纳米磁体受到外场脉冲后的平均磁化轨迹，使用机器学习模型预测产生该轨迹的材料参数 $\alpha$ 和 $K_u$。

具体研究目标包括：

1. 建立一个结构简单、计算量较小的纳米磁体微磁模型；
2. 使用 MuMax3 模拟不同 $\alpha$ 和 $K_u$ 下的磁化动力学；
3. 研究不同磁场激励对材料参数可辨识性的影响；
4. 使用人工特征、MLP 和一维 CNN 建立基线模型；
5. 构建时序 Transformer 反演 $\alpha$ 和 $K_u$；
6. 比较单激励与多激励反演的性能；
7. 测试模型在噪声、未见参数组合和部分外推条件下的表现；
8. 将机器学习预测的参数重新输入 MuMax3，检验其能否重建原始轨迹。

---

## 4. 核心研究问题

### 4.1 参数能否从磁化轨迹中被识别

不同材料参数会改变磁化的振荡频率、振幅和衰减速度。

本项目首先研究：

> 磁化时间序列中是否包含足够的信息，使模型能够同时区分 $\alpha$ 和 $K_u$？

一般来说：

- $\alpha$ 主要影响振荡的衰减速度；
- $K_u$ 主要影响回复力矩和振荡频率。

但是，二者对轨迹的影响并不完全独立，因此可能存在不同参数组合产生相似轨迹的情况。

---

### 4.2 多种外场激励能否提高参数可辨识性

只施加一种磁场脉冲时，某些参数可能难以区分。

因此，本项目计划对同一组材料参数使用两种或三种不同方向的磁场脉冲。

研究问题为：

> 多种激励产生的响应是否比单一激励包含更多材料参数信息？

---

### 4.3 Transformer 是否优于简单模型

Transformer 是一种适合处理序列数据的神经网络。

它能够比较时间序列中相距较远的时刻，例如比较多个振荡峰值的高度和时间间隔。

需要研究：

> Transformer 是否比人工提取频率和衰减率、MLP 或一维 CNN 更准确、更抗噪？

---

### 4.4 预测参数是否具有物理一致性

模型输出参数接近真实参数，并不一定意味着该参数能够正确解释观测到的物理轨迹。

因此需要进行正向回代：

$$
\left(
\widehat{\alpha},
\widehat{K}_u
\right)
\overset{\mathrm{MuMax3}}{\longrightarrow}
\widehat{\mathbf{m}}(t).
$$

然后比较：

$$
\widehat{\mathbf{m}}(t)
\quad \text{与} \quad
\mathbf{m}_{\mathrm{target}}(t).
$$

研究问题为：

> 模型预测的参数是否能够在 MuMax3 中重新产生原始磁化响应？

---

## 5. 物理系统

### 5.1 几何结构

初步计划使用一个扁平的三轴椭球薄纳米磁体，例如：

$$
L_x \times L_y \times L_z
=
100\,\mathrm{nm}
\times
50\,\mathrm{nm}
\times
2\,\mathrm{nm}.
$$

该尺寸对应椭球三轴全直径（即包围盒尺寸 $[d_x, d_y, d_z]$）；网格
$[n_x, n_y, n_z]$ 各分量可取任意正整数：$n_z=1$ 时单层体素离散自然表现为
恒厚椭圆截面薄片，$n_z>1$ 时才逐层解析 $z$ 方向椭球表面。该尺寸仅作为
初始方案，最终尺寸和网格大小需要通过试运行和网格收敛测试确定。

选择椭球结构的原因包括：

- 几何形状简单；
- 计算网格较小；
- 长轴提供明确的形状易轴；
- 磁化动力学比较直观；
- 可以接近单畴状态，同时保留一定的空间非均匀性；
- 适合使用普通桌面 GPU 进行批量模拟。

设椭球长轴沿 $x$ 方向。

初始磁化近似为：

$$
\mathbf{m}(\mathbf{r},0)
\approx
(1,0,0).
$$

---

### 5.2 固定参数与待反演参数

核心版本中固定以下参数：

- 饱和磁化强度 $M_s$；
- 交换常数 $A$；
- 磁体几何结构；
- 网格尺寸；
- 易轴方向；
- 外场脉冲的已知参数。

待反演参数为：

$$
\boldsymbol{\theta}
=
(\alpha,K_u).
$$

其中：

- $\alpha$ 是 Gilbert 阻尼系数；
- $K_u$ 是单轴磁各向异性常数。

第一阶段只反演两个参数，可以减小逆问题的难度，也便于分析不同参数对轨迹的影响。

---

## 6. 微磁物理模型

### 6.1 总能量

系统的微磁总能量可以写为：

$$
E
=
E_{\mathrm{ex}}
+
E_{\mathrm{ani}}
+
E_{\mathrm{demag}}
+
E_{\mathrm{Z}}.
$$

其中：

- $E_{\mathrm{ex}}$ 为交换能；
- $E_{\mathrm{ani}}$ 为磁各向异性能；
- $E_{\mathrm{demag}}$ 为退磁能；
- $E_{\mathrm{Z}}$ 为 Zeeman 能。

---

### 6.2 交换能

交换能为：

$$
E_{\mathrm{ex}}
=
A
\int_{\Omega}
|\nabla \mathbf{m}|^2\,dV.
$$

交换作用倾向于使相邻位置的磁化方向保持平行。

当磁化在空间中变化很快时，交换能会增加。因此，交换作用能够抑制过于剧烈的局部磁化变化。

---

### 6.3 单轴磁各向异性能

单轴磁各向异性能可以写为：

$$
E_{\mathrm{ani}}
=
K_u
\int_{\Omega}
\left[
1-(\mathbf{m}\cdot\mathbf{u})^2
\right]dV.
$$

其中 $\mathbf{u}$ 是易轴方向。

对应的各向异性有效场为：

$$
\mathbf{H}_{\mathrm{ani}}
=
\frac{2K_u}{\mu_0 M_s}
(\mathbf{m}\cdot\mathbf{u})\mathbf{u}.
$$

$K_u$ 越大，磁化偏离易轴所需要的能量越高。

可以将各向异性理解为一个将磁化拉回易轴方向的“回复作用”。

因此，$K_u$ 会影响：

- 磁化的进动频率；
- 磁化偏离易轴的幅度；
- 系统返回平衡状态的方式；
- 磁化翻转的难度。

---

### 6.4 Zeeman 能

外加磁场对应的 Zeeman 能为：

$$
E_{\mathrm{Z}}
=
-\mu_0 M_s
\int_{\Omega}
\mathbf{H}_{\mathrm{ext}}
\cdot
\mathbf{m}\,dV.
$$

Zeeman 能使磁化倾向于沿外加磁场方向排列。

本项目使用短时间磁场脉冲将磁化推离平衡位置，随后观察磁化的自由衰减过程。

---

### 6.5 退磁场

退磁场来源于磁体本身的形状和磁化分布。

其满足：

$$
\nabla \times \mathbf{H}_{\mathrm{demag}}=0,
$$

以及：

$$
\nabla \cdot
\left(
\mathbf{H}_{\mathrm{demag}}+\mathbf{M}
\right)
=0.
$$

退磁场是非局域的，即某个位置的退磁场会受到整个磁体磁化分布的影响。

对于薄椭球磁体，退磁场通常使磁化倾向于保持在薄膜平面内。

---

## 7. LLG 动力学方程

磁化动力学由 Landau–Lifshitz–Gilbert 方程描述：

$$
\frac{\partial \mathbf{m}}{\partial t}
=
-\frac{\gamma_0}{1+\alpha^2}
\left[
\mathbf{m}\times\mathbf{H}_{\mathrm{eff}}
+
\alpha
\mathbf{m}\times
\left(
\mathbf{m}\times\mathbf{H}_{\mathrm{eff}}
\right)
\right].
$$

有效场为：

$$
\mathbf{H}_{\mathrm{eff}}
=
\mathbf{H}_{\mathrm{ext}}
+
\mathbf{H}_{\mathrm{ani}}
+
\mathbf{H}_{\mathrm{demag}}
+
\mathbf{H}_{\mathrm{ex}}.
$$

LLG 方程右侧包含进动项和阻尼项。

### 7.1 进动项

进动项为：

$$
-\mathbf{m}\times\mathbf{H}_{\mathrm{eff}}.
$$

它使磁化围绕有效场旋转。

磁化不会立即指向有效场，而是像陀螺一样发生进动。

---

### 7.2 阻尼项

阻尼项为：

$$
-\alpha
\mathbf{m}\times
\left(
\mathbf{m}\times\mathbf{H}_{\mathrm{eff}}
\right).
$$

阻尼项使磁化逐渐靠近有效场方向，并使振荡逐渐衰减。

因此：

- 较小的 $\alpha$ 对应较慢的衰减；
- 较大的 $\alpha$ 对应较快的衰减。

在较简单的情况下，某个磁化分量可能近似表现为阻尼振荡：

$$
m_y(t)
\approx
C
e^{-t/\tau}
\cos(\omega t+\phi).
$$

其中：

- $\tau$ 主要与阻尼有关；
- $\omega$ 主要与有效场及 $K_u$ 有关；
- $C$ 和 $\phi$ 与激励方式和初始状态有关。

该表达式只是帮助理解的近似形式。实际 MuMax3 轨迹由完整 LLG 方程和空间磁化分布共同决定。

---

## 8. 外场激励设计

### 8.1 单激励方案

最简单的方案是施加一个沿 $y$ 方向的短脉冲：

$$
\mathbf{H}_A(t)
=
H_A(t)\hat{\mathbf{y}}.
$$

初始磁化沿 $x$ 方向，横向脉冲会产生明显力矩，使磁化偏离平衡状态。

脉冲结束后，记录磁化的自由衰减轨迹。

---

### 8.2 多激励方案

为了提高参数可辨识性，计划对同一组材料参数分别使用多种外场。

#### 激励 A：面内横向脉冲

$$
\mathbf{H}_A
=
H_A\hat{\mathbf{y}}.
$$

#### 激励 B：面内倾斜脉冲

$$
\mathbf{H}_B
=
H_B
\left(
\cos\theta_B,
\sin\theta_B,
0
\right).
$$

#### 激励 C：带有面外分量的脉冲

$$
\mathbf{H}_C
=
H_C
\left(
0,
\cos\theta_C,
\sin\theta_C
\right).
$$

同一组材料参数对应多条轨迹：

$$
X_A(t),
\qquad
X_B(t),
\qquad
X_C(t).
$$

模型利用这些轨迹的互补信息预测：

$$
\left[
X_A,
X_B,
X_C
\right]
\longrightarrow
(\alpha,K_u).
$$

核心研究假设是：

> 不同方向的磁场对不同材料参数具有不同的敏感性，因此多激励数据能够降低参数之间的混淆。

---

## 9. MuMax3 数值模拟方法

### 9.1 模拟流程

每组材料参数的模拟流程为：

1. 建立椭球纳米磁体；
2. 设置 $M_s$、$A$、$\alpha$ 和 $K_u$；
3. 设置易轴方向；
4. 初始化磁化；
5. 使用能量最小化或高阻尼弛豫获得平衡状态；
6. 施加短磁场脉冲；
7. 关闭磁场脉冲；
8. 使用真实 $\alpha$ 运行 LLG 动力学；
9. 记录平均磁化随时间的变化；
10. 对不同激励重复上述过程。

---

### 9.2 输出数据

每次模拟主要保存空间平均磁化：

$$
\overline{\mathbf{m}}(t)
=
\frac{1}{V}
\int_{\Omega}
\mathbf{m}(\mathbf{r},t)\,dV.
$$

数据表的形式为：

| 时间 | $\overline{m}_x$ | $\overline{m}_y$ | $\overline{m}_z$ |
|---:|---:|---:|---:|
| $t_0$ | $\overline{m}_x(t_0)$ | $\overline{m}_y(t_0)$ | $\overline{m}_z(t_0)$ |
| $t_1$ | $\overline{m}_x(t_1)$ | $\overline{m}_y(t_1)$ | $\overline{m}_z(t_1)$ |
| $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ |

必要时还可以保存：

- 总能量；
- 交换能；
- 退磁能；
- 各向异性能；
- 外场；
- 少量关键时刻的完整磁化场。

核心数据集不保存全部时间步的完整磁化场，以减少存储和计算压力。

---

### 9.3 参数采样

待采样参数为：

$$
\alpha\in[\alpha_{\min},\alpha_{\max}],
$$

$$
K_u\in[K_{u,\min},K_{u,\max}].
$$

初步先进行约 $100$ 到 $200$ 组试验模拟，用于确定：

- 轨迹是否有足够明显的差异；
- 外场脉冲是否过强或过弱；
- 模拟时间是否足够；
- 参数范围是否合理；
- 是否出现不希望出现的多畴或翻转状态。

确定参数范围后，再生成约 $1000$ 到 $3000$ 组参数组合。

参数采样可以使用：

- 均匀随机采样；
- Latin hypercube sampling；
- Sobol sequence。

相较于规则网格，这些方法能够更均匀地覆盖参数空间。

---

## 10. 数据预处理

### 10.1 时间插值

MuMax3 可能使用自适应时间步，因此不同模拟的原始输出时间点可能不同。

需要将所有轨迹插值到统一时间网格：

$$
t_0,t_1,\ldots,t_N.
$$

例如，每条轨迹统一为 $200$ 个时间点。

---

### 10.2 输入通道

单个时间点的输入可以表示为：

$$
\mathbf{x}_i
=
[
\overline{m}_x(t_i),
\overline{m}_y(t_i),
\overline{m}_z(t_i),
H_x(t_i),
H_y(t_i),
H_z(t_i)
].
$$

因此，一条轨迹可以写为矩阵：

$$
X\in\mathbb{R}^{N\times 6}.
$$

将外场同时作为输入，可以帮助模型区分：

- 轨迹变化是由材料参数引起的；
- 还是由激励方式引起的。

---

### 10.3 参数归一化

由于 $\alpha$ 和 $K_u$ 的数量级差别较大，需要进行归一化：

$$
\widetilde{\alpha}
=
\frac{\alpha-\alpha_{\min}}
{\alpha_{\max}-\alpha_{\min}},
$$

$$
\widetilde{K}_u
=
\frac{K_u-K_{u,\min}}
{K_{u,\max}-K_{u,\min}}.
$$

模型预测归一化后的参数，最后再转换回实际单位。

---

### 10.4 轨迹标准化

磁化分量本身位于 $[-1,1]$，通常不需要复杂缩放。

对于外场等其他输入，可以使用：

$$
\widetilde{H}
=
\frac{H-\mu_H}{\sigma_H}
$$

进行标准化。

---

## 11. 机器学习任务定义

每个样本的输入为一条或多条磁化轨迹：

$$
X_i
=
\left[
X_i^{(A)},
X_i^{(B)},
X_i^{(C)}
\right].
$$

对应标签为：

$$
\mathbf{y}_i
=
\left(
\alpha_i,
K_{u,i}
\right).
$$

模型学习映射：

$$
f_{\boldsymbol{\phi}}:
X_i
\longrightarrow
\widehat{\mathbf{y}}_i.
$$

其中 $\boldsymbol{\phi}$ 表示神经网络中需要训练的参数。

---

## 12. 基线模型

### 12.1 人工特征与 MLP

首先从磁化轨迹中人工提取一些具有明确物理意义的特征，例如：

- 主要振荡频率；
- 第一个和第二个峰值；
- 峰值衰减比；
- 首次过零时间；
- 最大振幅；
- 达到平衡所需时间；
- 曲线积分；
- 频谱峰宽。

形成特征向量：

$$
\mathbf{f}
=
[f_1,f_2,\ldots,f_p].
$$

再使用多层感知机预测参数：

$$
\mathbf{f}
\longrightarrow
(\widehat{\alpha},\widehat{K}_u).
$$

该模型用于建立容易理解的物理基线。

---

### 12.2 一维卷积神经网络

一维 CNN 在时间轴上使用滑动卷积核提取局部模式，例如：

- 峰值；
- 过零点；
- 局部振荡；
- 快速衰减；
- 脉冲开始和结束时的变化。

基本结构为：

$$
\text{轨迹}
\longrightarrow
\text{1D Convolution}
\longrightarrow
\text{Pooling}
\longrightarrow
\text{MLP}
\longrightarrow
(\alpha,K_u).
$$

一维 CNN 参数较少、训练稳定，可以作为 Transformer 的重要对照模型。

---

## 13. 主模型：时序 Transformer

Transformer 将轨迹中的每个时间点视为一个 token。

输入序列为：

$$
\mathbf{x}_1,
\mathbf{x}_2,
\ldots,
\mathbf{x}_N.
$$

首先通过线性映射将每个时间点转换为隐藏特征：

$$
\mathbf{e}_i
=
W_{\mathrm{emb}}\mathbf{x}_i
+
\mathbf{b}_{\mathrm{emb}}.
$$

例如，可以将原始 $6$ 维输入转换为 $64$ 或 $128$ 维内部表示。

---

### 13.1 时间位置编码

Transformer 本身不能天然判断时间顺序，因此需要加入时间编码：

$$
\mathbf{h}_i^{(0)}
=
\mathbf{e}_i
+
\mathbf{p}_i.
$$

其中 $\mathbf{p}_i$ 是第 $i$ 个时间点的位置编码。

也可以直接将归一化时间作为额外输入：

$$
\widetilde{t}_i
=
\frac{t_i}{T}.
$$

---

### 13.2 Self-Attention

Self-attention 允许模型比较不同时间点之间的关系：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^{\mathrm{T}}}{\sqrt{d}}
\right)V.
$$

在本项目中，attention 可能学习到：

- 比较多个振荡峰值，判断阻尼；
- 比较相邻峰值间隔，判断频率；
- 关注脉冲关闭后的自由衰减部分；
- 识别轨迹中最能区分 $\alpha$ 和 $K_u$ 的时间区域。

---

### 13.3 全局特征汇总

Transformer 输出每个时间点的隐藏表示：

$$
\mathbf{h}_1,
\mathbf{h}_2,
\ldots,
\mathbf{h}_N.
$$

使用平均池化得到整条轨迹的全局表示：

$$
\mathbf{h}_{\mathrm{global}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{h}_i.
$$

最后通过 MLP 输出：

$$
\left(
\widehat{\alpha},
\widehat{K}_u
\right)
=
\operatorname{MLP}
\left(
\mathbf{h}_{\mathrm{global}}
\right).
$$

---

## 14. 多激励 Transformer

对于多种激励，可以采用共享编码器结构：

$$
X_A
\overset{\mathrm{Transformer}}{\longrightarrow}
\mathbf{z}_A,
$$

$$
X_B
\overset{\mathrm{Transformer}}{\longrightarrow}
\mathbf{z}_B,
$$

$$
X_C
\overset{\mathrm{Transformer}}{\longrightarrow}
\mathbf{z}_C.
$$

三个分支共享同一个 Transformer 的模型参数。

随后将不同激励的特征拼接：

$$
\mathbf{z}_{\mathrm{fusion}}
=
[
\mathbf{z}_A,
\mathbf{z}_B,
\mathbf{z}_C
].
$$

最终预测为：

$$
(\widehat{\alpha},\widehat{K}_u)
=
\operatorname{MLP}
\left(
\mathbf{z}_{\mathrm{fusion}}
\right).
$$

该结构可以分别理解每种激励的动力学，同时利用不同实验之间的互补信息。

---

## 15. 损失函数

模型预测归一化参数：

$$
\widehat{\mathbf{y}}
=
\left(
\widehat{\widetilde{\alpha}},
\widehat{\widetilde{K}}_u
\right).
$$

最基本的损失函数为均方误差：

$$
\mathcal{L}_{\mathrm{param}}
=
\left(
\widehat{\widetilde{\alpha}}
-
\widetilde{\alpha}
\right)^2
+
\left(
\widehat{\widetilde{K}}_u
-
\widetilde{K}_u
\right)^2.
$$

如果两个参数的反演难度差别较大，可以使用加权损失：

$$
\mathcal{L}_{\mathrm{param}}
=
w_{\alpha}
\left(
\widehat{\widetilde{\alpha}}
-
\widetilde{\alpha}
\right)^2
+
w_K
\left(
\widehat{\widetilde{K}}_u
-
\widetilde{K}_u
\right)^2.
$$

模型优化器可以首先使用 AdamW。

---

## 16. 数据集划分

数据集应按照参数组合划分，而不是按照单条轨迹随机划分。

例如：

- 训练集：$70\%$ 的 $(\alpha,K_u)$ 参数组合；
- 验证集：$15\%$；
- 测试集：$15\%$。

同一参数组合下产生的所有不同激励轨迹必须放入同一个数据集。

否则可能出现同一材料参数的一条轨迹进入训练集，另一条轨迹进入测试集，从而产生数据泄漏。

---

## 17. 模型测试方案

### 17.1 插值测试

测试参数位于训练参数范围内，但具体数值未在训练中出现。

该测试主要评价模型在已知参数范围内的预测能力。

---

### 17.2 外推测试

测试参数位于训练范围边缘或略超出训练范围。

机器学习模型通常不擅长外推，因此应将外推性能与插值性能分开报告。

---

### 17.3 单激励与多激励比较

比较：

$$
X_A
\longrightarrow
(\alpha,K_u)
$$

与：

$$
[X_A,X_B,X_C]
\longrightarrow
(\alpha,K_u).
$$

重点观察：

- 参数误差是否降低；
- 噪声下是否更加稳定；
- 哪个参数从多激励中获益最大。

---

### 17.4 不同模型比较

计划比较以下模型：

| 模型 | 输入方式 | 主要作用 |
|---|---|---|
| 人工特征加 MLP | 频率、峰值和衰减率 | 物理基线 |
| 1D CNN | 原始时间序列 | 深度学习基线 |
| 单激励 Transformer | 单条原始轨迹 | 检验 attention |
| 多激励 Transformer | 多条轨迹 | 最终主模型 |

---

## 18. 噪声与鲁棒性研究

为了模拟实验误差，可以在磁化轨迹中加入高斯噪声：

$$
\widetilde{\mathbf{m}}(t)
=
\mathbf{m}(t)
+
\boldsymbol{\epsilon}(t).
$$

其中：

$$
\boldsymbol{\epsilon}(t)
\sim
\mathcal{N}(0,\sigma^2).
$$

可以测试不同噪声水平，例如：

$$
\sigma
=
0,
\quad
0.001,
\quad
0.005,
\quad
0.01.
$$

还可以加入：

- 外场幅值误差；
- 时间轴轻微偏移；
- 少量时间点缺失；
- 信号幅度缩放；
- 基线漂移。

研究目标为：

> 多激励 Transformer 是否比单激励模型和 CNN 更能抵抗测量噪声？

---

## 19. 评价指标

### 19.1 平均绝对误差

对于阻尼系数：

$$
\operatorname{MAE}_{\alpha}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
\widehat{\alpha}_i-\alpha_i
\right|.
$$

对于各向异性常数：

$$
\operatorname{MAE}_{K}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
\widehat{K}_{u,i}-K_{u,i}
\right|.
$$

---

### 19.2 相对误差

$$
\operatorname{RelativeError}_{\alpha}
=
\frac{
|\widehat{\alpha}-\alpha|
}{
|\alpha|
}.
$$

$$
\operatorname{RelativeError}_{K}
=
\frac{
|\widehat{K}_u-K_u|
}{
|K_u|
}.
$$

---

### 19.3 决定系数

$$
R^2
=
1
-
\frac{
\sum_i
(y_i-\widehat{y}_i)^2
}{
\sum_i
(y_i-\overline{y})^2
}.
$$

---

### 19.4 正向轨迹误差

将预测参数重新输入 MuMax3，获得轨迹：

$$
\widehat{\mathbf{m}}_{\mathrm{forward}}(t).
$$

定义正向轨迹误差：

$$
\mathcal{E}_{\mathrm{traj}}
=
\frac{1}{N_t}
\sum_{j=1}^{N_t}
\left|
\widehat{\mathbf{m}}_{\mathrm{forward}}(t_j)
-
\mathbf{m}_{\mathrm{target}}(t_j)
\right|^2.
$$

该指标可以检验预测参数是否在物理上能够解释原始观测。

---

## 20. 预期结果图表

项目最终计划产生以下主要图表：

1. 纳米磁体几何结构与外场方向示意图；
2. 不同 $\alpha$ 下的磁化衰减曲线；
3. 不同 $K_u$ 下的振荡频率变化；
4. 单激励与多激励轨迹示例；
5. $\alpha_{\mathrm{true}}$ 与 $\alpha_{\mathrm{pred}}$ 散点图；
6. $K_{u,\mathrm{true}}$ 与 $K_{u,\mathrm{pred}}$ 散点图；
7. 参数空间误差热图；
8. 不同模型误差对比；
9. 不同噪声强度下的误差变化；
10. 单激励与多激励性能比较；
11. 真实参数轨迹与预测参数回代轨迹对比；
12. MuMax3 传统搜索与 Transformer 推理时间比较。

---

## 21. 项目实现工具

### 21.1 物理模拟

- MuMax3；
- CUDA GPU；
- MuMax3 表格数据；
- 必要时使用 OVF 磁化场文件。

### 21.2 Python 科学计算

- NumPy：数组与数据处理；
- SciPy：插值、信号处理和曲线拟合；
- Pandas：表格数据管理；
- Matplotlib：曲线和误差图；
- scikit-learn：数据划分、基线模型和评价指标。

### 21.3 深度学习

- PyTorch；
- Transformer Encoder；
- 1D CNN；
- AdamW；
- mixed precision；
- TensorBoard 或其他训练记录工具。

### 21.4 自动化和高性能计算

- Python 自动生成 MuMax3 输入文件；
- Python 或 Shell 批量调用 MuMax3；
- 自动读取模拟结果；
- 自动检测失败任务；
- GPU 夜间批量模拟；
- 使用配置文件管理参数范围。

---

## 22. 计算资源规划

本项目主要使用平均磁化时间序列，因此机器学习部分的计算量较小。

建议的 Transformer 规模为：

- Transformer Encoder 层数：$2$ 到 $4$；
- hidden dimension：$64$ 或 $128$；
- attention heads：$4$；
- 时间序列长度：$100$ 到 $256$；
- 模型参数量：约 $10^6$ 到 $5\times10^6$；
- batch size：约 $32$ 到 $256$。

RTX 5070 Ti 足以训练该规模的模型。

主要计算开销来自 MuMax3 数据生成，而不是 Transformer 训练。

为了控制计算量：

- 使用二维或准二维薄膜；
- 不保存所有完整磁化场；
- 初始阶段只变化两个材料参数；
- 先进行少量 pilot simulation；
- 确定合理参数范围后再批量生成数据。

---

## 23. 项目时间安排

按照每天约 $5$ 小时、总计约十周进行规划。

| 时间 | 主要任务 | 预期产出 |
|---|---|---|
| 第 1 周 | 学习 LLG、$\alpha$、$K_u$ 和宏自旋模型 | Python 单磁矩动力学程序 |
| 第 2 周 | 建立 MuMax3 椭球纳米磁体 | 基本磁化响应曲线 |
| 第 3 周 | 测试参数范围与多种脉冲 | $100$ 到 $200$ 组试验数据 |
| 第 4 周 | 批量生成正式数据集 | 标准化时间序列数据 |
| 第 5 周 | 人工特征、MLP 和 1D CNN | 基线模型结果 |
| 第 6 周 | 实现单激励 Transformer | 单激励反演结果 |
| 第 7 周 | 实现多激励 Transformer | 主模型结果 |
| 第 8 周 | 噪声、插值与外推实验 | 鲁棒性和泛化分析 |
| 第 9 周 | MuMax3 正向回代验证 | 物理一致性结果 |
| 第 10 周 | 整理图表、报告和代码 | 完整项目成果 |

---

## 24. 风险与应对方案

### 24.1 不同参数产生相似轨迹

可能出现：

$$
(\alpha_1,K_{u,1})
\neq
(\alpha_2,K_{u,2}),
$$

但是：

$$
\mathbf{m}_1(t)
\approx
\mathbf{m}_2(t).
$$

应对方法包括：

- 使用多种外场激励；
- 延长自由衰减观测时间；
- 加入面外激励；
- 同时输入外场时间序列；
- 使用概率模型描述参数不确定性。

---

### 24.2 外场过弱

如果外场过弱，磁化轨迹几乎不发生变化：

$$
\mathbf{m}(t)
\approx
\mathbf{m}(0).
$$

此时轨迹中没有足够的信息用于参数反演。

应对方法：

- 在试验阶段扫描脉冲强度；
- 确保轨迹具有明显振荡；
- 同时避免外场过强导致复杂翻转。

---

### 24.3 外场过强导致多畴或翻转

过强激励可能产生不同的动力学模式，使参数反演难度显著增加。

应对方法：

- 核心项目限制在非翻转、小幅或中等振荡范围；
- 将翻转和多畴动力学作为后续扩展；
- 使用磁化快照检查系统是否仍然接近单畴。

---

### 24.4 系统过于接近宏自旋

如果所有网格中的磁化完全一致，MuMax3 可能没有体现微磁模型的必要性。

应对方法：

- 使用略大一些的椭球结构；
- 保留边缘处的小幅非均匀磁化；
- 比较宏自旋模型和完整 MuMax3 结果；
- 分析宏自旋近似何时开始失效。

---

### 24.5 模型过拟合

应对方法：

- 按参数组合划分数据集；
- 使用 dropout；
- 使用 early stopping；
- 控制 Transformer 规模；
- 增加噪声和数据增强；
- 同时报告训练、验证和测试误差。

---

## 25. 最低可交付成果

如果项目时间紧张，最低版本应完成：

1. 建立椭球纳米磁体 MuMax3 模型；
2. 生成不同 $\alpha$ 和 $K_u$ 下的平均磁化轨迹；
3. 建立 1D CNN 基线；
4. 建立单激励 Transformer；
5. 对比不同模型的参数反演误差；
6. 将预测参数回代 MuMax3；
7. 提交代码、数据处理流程和研究报告。

最低版本已经能够形成完整的研究流程：

$$
\text{模拟}
\longrightarrow
\text{数据}
\longrightarrow
\text{机器学习反演}
\longrightarrow
\text{物理验证}.
$$

---

## 26. 理想完成版本

在核心任务基础上进一步完成：

1. 三种不同外场激励；
2. 多激励共享 Transformer；
3. 噪声鲁棒性研究；
4. 插值与外推测试；
5. 参数空间误差热图；
6. Deep Ensemble 不确定性估计；
7. 多种激励的信息量比较；
8. Transformer 初值加传统优化精修。

---

## 27. 后续研究拓展

### 27.1 增加第三个材料参数

在完成 $\alpha$ 和 $K_u$ 后，可以加入 $M_s$：

$$
\mathbf{m}(t)
\longrightarrow
(\alpha,K_u,M_s).
$$

由于 $M_s$ 会影响退磁场，可能需要增加面外激励来提高其可辨识性。

---

### 27.2 概率参数反演

普通模型输出单个参数值：

$$
(\widehat{\alpha},\widehat{K}_u).
$$

更先进的模型可以输出参数的条件概率分布：

$$
p(\alpha,K_u\mid X).
$$

这样可以表示：

- 哪些参数组合都能解释观测；
- 模型对预测有多大把握；
- 逆问题是否存在多组可能解。

可选方法包括：

- Deep Ensemble；
- Mixture Density Network；
- Conditional Normalizing Flow；
- Neural Posterior Estimation。

在当前项目中，Deep Ensemble 是最容易实现的扩展。

---

### 27.3 激励脉冲优化

可以进一步研究：

> 什么方向、强度和持续时间的磁场最有利于区分材料参数？

将激励参数记为：

$$
\boldsymbol{\eta}
=
(H,\theta,t_{\mathrm{pulse}}).
$$

目标是寻找最优激励：

$$
\boldsymbol{\eta}^{*}
=
\operatorname*{arg\,min}_{\boldsymbol{\eta}}
\mathcal{E}_{\mathrm{inverse}}
(\boldsymbol{\eta}).
$$

这相当于使用模拟和机器学习设计最有信息量的“虚拟实验”。

---

### 27.4 加入空间磁化信息

核心模型只使用平均磁化：

$$
\overline{\mathbf{m}}(t).
$$

平均操作会丢失部分空间信息。

后续可以在少量时间点保存完整磁化场：

$$
\mathbf{m}(x,y,t).
$$

使用 CNN 提取空间特征：

$$
\mathbf{m}(x,y,t_i)
\longrightarrow
\mathbf{z}_i.
$$

再使用 Transformer 处理时间序列：

$$
\mathbf{z}_1,
\ldots,
\mathbf{z}_N
\longrightarrow
(\alpha,K_u).
$$

该扩展能够利用畴壁、边缘磁化和局部非均匀结构，但数据量和模型复杂度会增加。

---

### 27.5 Transformer 与传统优化结合

传统反演方法通过最小化轨迹误差寻找参数：

$$
J(\alpha,K_u)
=
\sum_j
\left|
\mathbf{m}_{\mathrm{sim}}
(t_j;\alpha,K_u)
-
\mathbf{m}_{\mathrm{obs}}(t_j)
\right|^2.
$$

最优参数为：

$$
(\alpha^*,K_u^*)
=
\operatorname*{arg\,min}_{\alpha,K_u}
J(\alpha,K_u).
$$

可以使用 Transformer 给出快速初始猜测：

$$
(\alpha_0,K_{u,0})
=
f_{\mathrm{Transformer}}(X).
$$

再在该初始值附近进行少量局部优化。

这种混合方法可能同时获得：

- Transformer 的快速推理；
- 传统物理优化的高精度；
- 更少的 MuMax3 调用次数。

---

## 28. 预期创新点

本项目的主要特点不应只表述为“使用 Transformer”，而应包括：

1. 从微磁动力学轨迹反演材料参数，而不是预测完整未来磁化场；
2. 使用多种外场激励提高材料参数的可辨识性；
3. 比较人工物理特征、CNN 和 Transformer；
4. 系统分析噪声、插值和外推条件；
5. 将预测参数重新输入 MuMax3 进行正向物理验证；
6. 探索激励方式与参数反演精度之间的关系。

---

## 29. 预期学习成果

### 29.1 物理学

- 微磁学连续模型；
- LLG 方程；
- Gilbert 阻尼；
- 磁各向异性；
- 退磁场；
- 外场驱动磁化动力学；
- 逆问题和参数可辨识性。

### 29.2 数值计算

- ODE 和 LLG 时间积分的基本思想；
- 有限差分网格；
- 时间插值；
- 网格收敛；
- 参数采样；
- 数值误差和物理验证。

### 29.3 Python 科学计算

- NumPy；
- SciPy；
- Pandas；
- Matplotlib；
- 批量数据处理；
- 自动化生成模拟任务。

### 29.4 机器学习

- 训练集、验证集和测试集；
- 回归任务；
- MLP；
- 一维 CNN；
- Transformer；
- attention；
- 损失函数；
- 过拟合；
- 噪声鲁棒性；
- 模型不确定性。

### 29.5 高性能计算

- MuMax3 GPU 模拟；
- CUDA 环境；
- 批量任务；
- 混合精度训练；
- GPU 显存管理；
- 模拟数据和日志管理。

---

## 30. 项目总结

本项目研究的核心映射为：

$$
\left[
\overline{\mathbf{m}}_A(t),
\overline{\mathbf{m}}_B(t),
\overline{\mathbf{m}}_C(t)
\right]
\longrightarrow
(\alpha,K_u).
$$

完整流程为：

$$
\text{选择材料参数}
\longrightarrow
\text{MuMax3 动力学模拟}
\longrightarrow
\text{生成多激励磁化轨迹},
$$

$$
\text{多激励磁化轨迹}
\longrightarrow
\text{Transformer 参数反演}
\longrightarrow
\text{预测参数回代 MuMax3}
\longrightarrow
\text{物理一致性验证}.
$$

该课题规模适合在约两个半月内完成，机器学习模型对算力要求较低，主要计算成本来自 MuMax3 数据生成。

项目同时包含：

- 微磁学；
- 数值模拟；
- Python 科学计算；
- Transformer；
- 逆问题；
- GPU 批量计算。

因此，该项目能够形成一条较为完整的计算物理与科学机器学习研究路线。