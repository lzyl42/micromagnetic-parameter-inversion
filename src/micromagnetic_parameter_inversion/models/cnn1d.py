"""CNN1D 反演回归模型（已实现）。

从多激励磁化轨迹反演 ``(alpha, Ku)`` 的 1D 卷积回归器。输入契约与 MLP 共用
同一份 npz 数据（``[N, P, T, 3]``），但本模型**完全独立**：不与 MLP 共享
权重、checkpoint、已拟合的预处理统计量或训练产物；训练时自行做 train-only
拟合。

前向数据流：

1. ``permute(0, 1, 3, 2).contiguous()``：``[N, P, T, 3] → [N, P, 3, T]``，
   分量维移到时间维之前；**pulse 维保持分离，不拼接时间**。
2. ``reshape(N, 3P, T)``：把 ``(pulse, component)`` 合并为通道维，通道顺序固定
   pulse-major、component-minor（索引 ``= p*3 + c``，即
   ``p0_mx, p0_my, p0_mz, p1_mx, …``），与 ``InputContract.pulse_order`` /
   ``component_order`` 对应。
3. ``self.features``：逐层 ``Conv1d(stride=1, dilation=1, padding=(k-1)//2) +
   ReLU``（奇数 kernel 使长度维保持 ``T``）。
4. ``self.pool = AdaptiveAvgPool1d(pool_bins)``：沿时间轴聚合为固定 bin 数。
5. ``self.head``：``Flatten(start_dim=1)`` → 各 ``Linear + ReLU`` → 末层
   ``Linear`` 输出 2 维（无末激活）。

输出为 ``[N, 2]`` 的 ``alpha``、``Ku`` **标准化标签值（z-score 空间，非物理
单位）**；物理单位还原依赖训练侧保存的预处理统计量，不属于本模块职责。

``padding=(k-1)//2`` 使用默认 zero padding，序列边缘与池化分 bin 处可能有边界
效应。本模块只做特征提取：不做标准化、不读数据、不导入任何训练模块。
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["CNN1DRegressor"]

_N_OUTPUTS = 2  # 输出列：(alpha, Ku) 标准化标签
_N_CHANNELS = 3  # 磁化分量数 (mx, my, mz)


def _require_positive_int(value: object, field: str) -> int:
    """正整数：拒绝 bool/float/string，非正值报 ValueError。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} 必须为正整数 (got {value!r})")
    return value


def _require_int_sequence(value: object, field: str, *, allow_empty: bool) -> tuple[int, ...]:
    """整数序列：接受 tuple/list 并规范为 tuple；None/标量/字符串报 ValueError。"""
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field} 必须为整数序列 (got {value!r})")
    sequence = tuple(value)
    if not sequence and not allow_empty:
        raise ValueError(f"{field} 必须为非空整数序列 (got {value!r})")
    for index, item in enumerate(sequence):
        _require_positive_int(item, f"{field}[{index}]")
    return sequence


class CNN1DRegressor(nn.Module):
    """从多激励磁化轨迹反演 (alpha, Ku) 的 1D CNN 回归器。

    Attributes:
        input_shape: 规范化 ``(P, T, 3)``。
        channels: 各 Conv1d 层输出通道数（tuple）。
        kernel_sizes: 各 Conv1d 层核长度（tuple，奇数，与 channels 等长）。
        pool_bins: ``AdaptiveAvgPool1d`` 输出 bin 数（``1 <= pool_bins <= T``）。
        head_hidden_dims: 回归头隐层宽度序列（tuple，可为空）。
        features: ``nn.Sequential``：Conv1d/ReLU 堆叠。
        pool: ``nn.AdaptiveAvgPool1d``。
        head: ``nn.Sequential``：Flatten + Linear/ReLU + 末层 Linear。
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
        """初始化网络层（无默认超参，全部由调用方显式给出）。

        Args:
            input_shape: ``(P, T, 3)``，三维正整数且末维恒为 3。
            channels: 各 Conv1d 层输出通道数，非空正整数序列。
            kernel_sizes: 各 Conv1d 层核长度，非空正奇数序列，与 channels 等长。
            pool_bins: 正整数且 ``<= T``（允许 1 与 T）。
            head_hidden_dims: 回归头隐层宽度，可为空；各元素正整数。

        Raises:
            ValueError: 上述任一约束不满足（含 bool/float/string 冒充整数、
                层数不匹配、偶数 kernel、``pool_bins > T`` 等）。
        """
        super().__init__()
        shape = _require_int_sequence(input_shape, "input_shape", allow_empty=False)
        if len(shape) != 3:
            raise ValueError(f"input_shape 必须恰为 3 维 (P, T, 3) (got {input_shape!r})")
        if shape[2] != _N_CHANNELS:
            raise ValueError(f"input_shape 末维须为 {_N_CHANNELS} (mx,my,mz) (got {shape[2]})")
        n_pulse, n_time = shape[0], shape[1]

        normalized_channels = _require_int_sequence(channels, "channels", allow_empty=False)
        normalized_kernels = _require_int_sequence(kernel_sizes, "kernel_sizes", allow_empty=False)
        if len(normalized_kernels) != len(normalized_channels):
            raise ValueError(
                f"kernel_sizes 层数须与 channels 相同 "
                f"(got {len(normalized_kernels)} vs {len(normalized_channels)})"
            )
        if any(kernel % 2 == 0 for kernel in normalized_kernels):
            raise ValueError(f"kernel_sizes 须全为奇数 (got {kernel_sizes!r})")
        normalized_pool_bins = _require_positive_int(pool_bins, "pool_bins")
        if normalized_pool_bins > n_time:
            raise ValueError(f"pool_bins 须 <= T={n_time} (got {normalized_pool_bins})")
        normalized_head = _require_int_sequence(
            head_hidden_dims, "head_hidden_dims", allow_empty=True
        )

        self.input_shape: tuple[int, int, int] = (n_pulse, n_time, _N_CHANNELS)
        self.channels: tuple[int, ...] = normalized_channels
        self.kernel_sizes: tuple[int, ...] = normalized_kernels
        self.pool_bins: int = normalized_pool_bins
        self.head_hidden_dims: tuple[int, ...] = normalized_head

        features: list[nn.Module] = []
        in_channels = n_pulse * _N_CHANNELS
        for out_channels, kernel in zip(normalized_channels, normalized_kernels, strict=True):
            features.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=1,
                    dilation=1,
                    padding=(kernel - 1) // 2,
                )
            )
            features.append(nn.ReLU())
            in_channels = out_channels
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool1d(normalized_pool_bins)

        head_layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        in_features = normalized_channels[-1] * normalized_pool_bins
        for width in normalized_head:
            head_layers.append(nn.Linear(in_features, width))
            head_layers.append(nn.ReLU())
            in_features = width
        head_layers.append(nn.Linear(in_features, _N_OUTPUTS))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：``[N, P, T, 3] → [N, 2]`` 标准化标签预测。

        Args:
            x: 输入 batch，形状须与 ``input_shape`` 一致；允许非连续张量
                （内部经 ``permute + contiguous`` 整理）。

        Returns:
            形状 ``[N, 2]`` 的标准化标签预测值（z-score 空间，非物理单位）。

        Raises:
            ValueError: ``x`` 非 4 维，或 ``x.shape[1:]`` 与 ``input_shape``
                不一致。
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"输入形状须为 [N, {self.input_shape}] 且与 input_shape 一致 (got {tuple(x.shape)})"
            )
        batch = x.shape[0]
        n_pulse, n_time, _ = self.input_shape
        x = x.permute(0, 1, 3, 2).contiguous().reshape(batch, n_pulse * _N_CHANNELS, n_time)
        return self.head(self.pool(self.features(x)))
