"""固定完整时间序列的 Temporal Transformer 回归器。

输入 [B,P,T,3] 按 p*3+c 排列为时间 token，输出两列标准化标签。
位置编码采用 index=0..T-1、base=10000 的固定正弦算法，不保存到 checkpoint。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from micromagnetic_parameter_inversion.models.cnn1d import (
    _require_int_sequence,
    _require_positive_int,
)

__all__ = ["TemporalTransformerRegressor"]


class TemporalTransformerRegressor(nn.Module):
    """输入投影 → 位置编码 → pre-norm Encoder → 时间均值 → GELU 回归头。"""

    positional_encoding: Tensor

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        head_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        shape = _require_int_sequence(input_shape, "input_shape", allow_empty=False)
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"input_shape 必须为 (P, T, 3) (got {input_shape!r})")
        self.input_shape = (shape[0], shape[1], shape[2])
        self.d_model = _require_positive_int(d_model, "d_model")
        self.nhead = _require_positive_int(nhead, "nhead")
        self.num_layers = _require_positive_int(num_layers, "num_layers")
        self.dim_feedforward = _require_positive_int(dim_feedforward, "dim_feedforward")
        if d_model % 2 or d_model % nhead:
            raise ValueError("d_model 须为偶数且能被 nhead 整除")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout 须为有限实数且 0 <= dropout < 1")
        self.dropout = float(dropout)
        self.head_hidden_dims = _require_int_sequence(
            head_hidden_dims, "head_hidden_dims", allow_empty=True
        )

        self.input_projection = nn.Linear(shape[0] * 3, d_model)
        positions = torch.arange(shape[1], dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.empty(1, shape[1], d_model)
        pe[0, :, 0::2] = torch.sin(positions * frequencies)
        pe[0, :, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("positional_encoding", pe, persistent=False)
        self.input_dropout = nn.Dropout(self.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        # Encoder 深拷贝层默认初值相同；逐层独立抽样，包含 packed QKV 矩阵。
        for encoder_layer in self.encoder.layers:
            for name, parameter in encoder_layer.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif name.endswith("weight"):
                    nn.init.ones_(parameter)  # LayerNorm
                else:
                    nn.init.zeros_(parameter)

        head: list[nn.Module] = []
        width = d_model
        for hidden in self.head_hidden_dims:
            head.extend([nn.Linear(width, hidden), nn.GELU()])
            width = hidden
        head.append(nn.Linear(width, 2))
        self.head = nn.Sequential(*head)

    def forward(self, x: Tensor) -> Tensor:
        """完整 shape 固定，允许非连续输入；无 mask/patch/CLS。"""
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"输入形状须与 input_shape={self.input_shape} 一致 (got {tuple(x.shape)})"
            )
        batch = x.shape[0]
        n_pulse, n_time, _ = self.input_shape
        tokens = x.permute(0, 2, 1, 3).reshape(batch, n_time, n_pulse * 3)
        projected = self.input_projection(tokens)
        encoded = self.encoder(
            self.input_dropout(projected + self.positional_encoding.to(dtype=projected.dtype))
        )
        return self.head(encoded.mean(dim=1))
