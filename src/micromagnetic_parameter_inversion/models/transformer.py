"""Temporal Transformer regressor for a fixed full time series.

Input [B,P,T,3] is arranged into time tokens as p*3+c and outputs two
standardized labels. The positional encoding uses the fixed sinusoidal
algorithm with index=0..T-1 and base=10000 and is not saved into checkpoints.
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
    """Input projection → positional encoding → pre-norm Encoder → time mean → GELU head."""

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
            raise ValueError(f"input_shape must be (P, T, 3) (got {input_shape!r})")
        self.input_shape = (shape[0], shape[1], shape[2])
        self.d_model = _require_positive_int(d_model, "d_model")
        self.nhead = _require_positive_int(nhead, "nhead")
        self.num_layers = _require_positive_int(num_layers, "num_layers")
        self.dim_feedforward = _require_positive_int(dim_feedforward, "dim_feedforward")
        if d_model % 2 or d_model % nhead:
            raise ValueError("d_model must be even and divisible by nhead")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite real number with 0 <= dropout < 1")
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
        # Encoder layers are deep copies with identical default init; sample each layer
        # independently, including the packed QKV matrix.
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
        """Full shape is fixed; non-contiguous inputs are allowed; no mask/patch/CLS."""
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"input shape must match input_shape={self.input_shape} (got {tuple(x.shape)})"
            )
        batch = x.shape[0]
        n_pulse, n_time, _ = self.input_shape
        tokens = x.permute(0, 2, 1, 3).reshape(batch, n_time, n_pulse * 3)
        projected = self.input_projection(tokens)
        encoded = self.encoder(
            self.input_dropout(projected + self.positional_encoding.to(dtype=projected.dtype))
        )
        return self.head(encoded.mean(dim=1))
