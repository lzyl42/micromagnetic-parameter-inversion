"""CNN1D 反演回归模型（**架构审核骨架，未实现**）。

本模块当前只用于评审接口与数据流设计（设计要点见本 docstring 与 ``TODO``）。
构造函数与 ``forward`` 均**未实现**（显式 ``raise NotImplementedError``），
不创建任何层、不持有权重、也不产生输出；必须在架构评审通过后按批准阶段再
实现。

与 MLP 的关系：CNN 与 MLP **只共用同一份数据集与冻结 split**；不共用 MLP
权重、checkpoint、已拟合的预处理统计量或训练产物。CNN 未来独立做 train-only
拟合、独立 checkpoint（``CNNCheckpoint``）与运行产物，**不读取** MLP 的任何
ckpt/preprocessing。

输入契约（与 MLP 共用同一份 npz 数据契约，见 ``train.md`` 第 4/5 节）：

- ``input_shape = (P, T, 3)``：单样本为某参数组全部 pulse 的原始时域轨迹
  ``x[P, T, 3]``（``P`` = pulse 数、``T`` = 时间步数、``3`` = (mx, my, mz)）。
- 一个 batch 形状 ``[N, P, T, 3]``（``N`` = 参数组级样本数）。
- ``T`` 由 ``InputContract`` 冻结；契约语义（``t_s``、多 ``P`` 顺序等）由
  数据/工厂层校验，本模块不自行重新定义。

拟议前向数据流（**待审核基线，不是已证明更优的方案**）：

1. ``permute(0, 1, 3, 2)``：``[N, P, T, 3] → [N, P, 3, T]``，把分量维移到
   时间维之前；**pulse 维保持分离**，不做任何时间拼接。
2. ``contiguous().reshape(N, 3 * P, T)``：``contiguous()`` 是**显式保证内存
   布局**（使后续按 pulse-major、component-minor 展开符合预期），**不是**
   ``reshape`` 的 API 要求；把 (pulse, component) 合并为通道维 ``[N, 3P, T]``，
   通道顺序固定为 pulse-major、component-minor
   （``p0_mx, p0_my, p0_mz, p1_mx, …``），与 ``InputContract.pulse_order`` /
   ``component_order`` 一一对应；时间轴 ``T`` 原样保留，**绝不把不同 pulse
   的时间序列首尾相接**。
3. ``Conv1d → ReLU`` 逐层堆叠：每层 ``stride=1``、kernel 取**奇数**、
   ``padding=(k-1)//2``，使长度维保持 ``T`` 不变。
4. ``AdaptiveAvgPool1d(pool_bins)``：沿时间轴下采样到 ``pool_bins`` 个 bin，
   **保留多段局部时间信息**，而非只取单一全局均值；这只是特征聚合，**不改变**
   输入契约对 ``T`` 的固定要求。
5. ``Flatten(start_dim=1)`` → 小规模 ``Linear + ReLU`` 回归头 → ``Linear``
   输出 2 维：``[N, 2]``。

``forward`` 需校验 ``ndim`` 与各维尺寸，并支持**非连续输入**（内部显式整理
布局）；``T`` 由契约固定，本模块不因时间池化而接受任意 ``T``。

感受野/时间跨度记录（仅备忘，不作性能断言）：单分支感受野
``RF = 1 + sum(k_i - 1)``，对应真实时间跨度约 ``(RF - 1) * dt``；池化 bin 之间
可能重叠、边界 padding 可能带来边缘效应，具体影响留待实现后按数据评估。

输出语义与 MLP 一致：形状 ``[N, 2]``，两列分别为 ``alpha`` 与 ``Ku`` 的
**标准化标签值（z-score 空间，非物理单位）**；物理单位还原依赖 CNN 自己的
预处理统计量，不属于本模块职责。

校验规则（**本骨架未实现**）：``channels`` 为非空整数序列、各维为正且拒绝
``bool``；``kernel_sizes`` 为非空整数序列、各维为**正奇数**且拒绝 ``bool``；
``len(kernel_sizes) == len(channels)``；``pool_bins`` 为正且 ``<= T``（项目
设计限制：只做时间聚合/下采样、不扩张时间 bin）；``head_hidden_dims`` 为整数
序列、各维为正、**允许为空**（= 直接线性输出）且拒绝 ``bool``。拟议结构仍用
ReLU 激活；网络结构保持此前设计，**不指定任何超参**。

本模块不读取数据、不做任何标准化/归一化，也不导入其他训练模块
（training_config / training_data / preprocessing / training 等），以避免
环形依赖。
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["CNN1DRegressor"]


class CNN1DRegressor(nn.Module):
    """从多激励磁化轨迹反演 (alpha, Ku) 的 1D CNN 回归器（**未实现骨架**）。

    Attributes（实现后拟持有；骨架阶段不创建任何子模块/参数）:
        input_shape: 输入样本形状 ``(P, T, 3)``，用于校验 batch 形状并推导
            首层输入通道 ``3P``；随实例保存（未来独立 CNN ckpt 结构字段来源）。
        channels: 各 Conv1d 层的输出通道数序列（``out_channels``）；首层
            输入通道固定为 ``3P``。
        kernel_sizes: 与 ``channels`` 等长的卷积核长度序列（均为正奇数）。
        pool_bins: ``AdaptiveAvgPool1d`` 的时间维输出 bin 数（``<= T``）。
        head_hidden_dims: 回归头隐层宽度序列（可为空 = 直接线性输出）。

    注意：默认超参**未**在本骨架中锁定；``channels``/``kernel_sizes``/
    ``pool_bins``/``head_hidden_dims`` 的具体取值留待评审与实验阶段决定。
    本类与 MLP 完全独立，不复用 MLP 权重/ckpt/预处理。
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        channels: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        pool_bins: int,
        head_hidden_dims: tuple[int, ...],
    ) -> None:
        """初始化网络层（**架构审核骨架：不构造任何层**）。

        Args:
            input_shape: ``(P, T, 3)``，见模块 docstring 输入契约。
            channels: 各 Conv1d 层输出通道数（keyword-only，非空）。
            kernel_sizes: 各 Conv1d 层核长度（正奇数，与 ``channels`` 等长）。
            pool_bins: ``AdaptiveAvgPool1d`` 的输出时间 bin 数（``<= T``）。
            head_hidden_dims: 回归头隐层宽度序列（允许为空）。

        Raises:
            NotImplementedError: 恒抛出——本类为评审骨架，尚未实现。
        """
        # 步骤 0：完成 nn.Module 基类初始化；随后显式声明骨架未实现。
        super().__init__()
        raise NotImplementedError(
            "CNN1DRegressor 为架构审核骨架，尚未实现；待架构评审通过后再按批准阶段实现。"
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向传播 ``[N, P, T, 3] → [N, 2]``（**未实现**）。

        Args:
            x: 输入 batch，形状 ``[N, P, T, 3]``——已按训练组统计标准化的
                原始时域轨迹（标准化由 CNN 自己的 preprocessing 负责，不在此
                处理）。实现时须校验 ``ndim`` 与各维尺寸、支持非连续输入；
                拟议流程见模块 docstring：permute → contiguous → reshape →
                Conv1d/ReLU → AdaptiveAvgPool1d → Flatten → Linear 回归头。

        Returns:
            形状 ``[N, 2]`` 的标准化标签预测值（z-score 空间，非物理单位）。

        Raises:
            NotImplementedError: 恒抛出——本类为评审骨架，尚未实现。
        """
        raise NotImplementedError(
            "CNN1DRegressor.forward 为架构审核骨架，尚未实现；请在骨架阶段不要调用本模型。"
        )
