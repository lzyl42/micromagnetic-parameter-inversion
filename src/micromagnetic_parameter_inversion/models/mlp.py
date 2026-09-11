"""MLP 反演回归模型。

输入契约 ``input_shape = (P, T, 3)``：单样本为某参数组全部 pulse 的原始
时域轨迹，batch 形状 ``[N, P, T, 3]``（P 个 pulse、T 个时间步、3 个磁化
分量 mx/my/mz）。不做每 pulse 切分、不做手工统计特征、不做降采样。

输出语义：形状 ``[N, 2]``，两列分别为 ``alpha`` 与 ``Ku`` 的**标准化标签
值（z-score 后）**，**不是物理单位**；物理单位还原依赖预处理模块保存的
标准化统计量。

本模块不读取数据、不做任何标准化/归一化，也不导入其他训练模块
（training_config / training_data / preprocessing / training 等），
以避免环形依赖。
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["MLPRegressor"]

_N_OUTPUTS = 2  # 输出列：(alpha, ku_j_per_m3) 的标准化标签
_N_CHANNELS = 3  # 磁化分量数 (mx, my, mz)：输入契约恒为 3


class MLPRegressor(nn.Module):
    """从多激励磁化轨迹反演 (alpha, Ku) 的 MLP 回归器。

    ``input_shape``/``hidden_dims`` 随实例保存，是 checkpoint 的结构字段
    来源；展平维度 ``D = P * T * 3``。
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dims: tuple[int, ...] = (64, 32, 32),
    ) -> None:
        """初始化网络层。

        Args:
            input_shape: ``(P, T, 3)``，各维必须为正，通道维恒为 3。
            hidden_dims: 隐层宽度序列（可为空 = 直接 ``Linear D→2``），
                各维必须为正；末层线性输出标准化标签（非物理单位）。

        Raises:
            ValueError: ``input_shape`` 非 3 元、P/T 非正、通道维不为 3，
                或 ``hidden_dims`` 含非正宽度。
        """
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"input_shape 须为 (P, T, 3) (got {tuple(input_shape)})")
        n_pulse, n_time, n_channels = input_shape
        if n_pulse < 1 or n_time < 1:
            raise ValueError(f"input_shape 的 P/T 必须为正 (got {tuple(input_shape)})")
        if n_channels != _N_CHANNELS:
            raise ValueError(
                f"input_shape 的通道维恒为 {_N_CHANNELS} (mx,my,mz) (got {n_channels})"
            )
        if any(width < 1 for width in hidden_dims):
            raise ValueError(f"hidden_dims 各维必须为正 (got {tuple(hidden_dims)})")

        self.input_shape: tuple[int, int, int] = (n_pulse, n_time, n_channels)
        self.hidden_dims: tuple[int, ...] = tuple(hidden_dims)

        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        prev_dim = n_pulse * n_time * n_channels
        for width in hidden_dims:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, _N_OUTPUTS))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：``[N, P, T, 3] → [N, 2]`` 标准化标签预测。

        Args:
            x: 输入 batch，形状 ``[N, P, T, 3]``——已按训练组统计标准化
                的原始时域轨迹（标准化由 preprocessing 模块负责，不在此
                处理）。

        Returns:
            形状 ``[N, 2]`` 的张量：``alpha`` 与 ``Ku`` 的标准化标签预测
            值（z-score 空间，非物理单位）。

        Raises:
            ValueError: ``x`` 非 4 维，或第 1–3 维与 ``input_shape`` 不一致。
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"输入形状须为 [N, P, T, 3] 且与 input_shape={self.input_shape} 一致 "
                f"(got {tuple(x.shape)})"
            )
        return self.network(x)
