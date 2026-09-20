"""CNN1D inverse-regression model (implemented).

A 1D convolutional regressor that inverts ``(alpha, Ku)`` from multi-excitation
magnetization trajectories. The input contract shares the same npz data as the
MLP (``[N, P, T, 3]``), but this model is **fully independent**: it does not
share weights, checkpoints, fitted preprocessing statistics, or training
artifacts with the MLP; it performs its own train-only fit during training.

Forward data flow:

1. ``permute(0, 1, 3, 2).contiguous()``: ``[N, P, T, 3] → [N, P, 3, T]``,
   moving the component dimension before the time dimension; the **pulse
   dimension stays separate and time is not concatenated**.
2. ``reshape(N, 3P, T)``: merges ``(pulse, component)`` into the channel
   dimension; the channel order is fixed pulse-major, component-minor (index
   ``= p*3 + c``, i.e. ``p0_mx, p0_my, p0_mz, p1_mx, …``), corresponding to
   ``InputContract.pulse_order`` / ``component_order``.
3. ``self.features``: per-layer ``Conv1d(stride=1, dilation=1, padding=(k-1)//2)
   + ReLU`` (odd kernels keep the length dimension at ``T``).
4. ``self.pool = AdaptiveAvgPool1d(pool_bins)``: aggregates along the time axis
   into a fixed number of bins.
5. ``self.head``: ``Flatten(start_dim=1)`` → each ``Linear + ReLU`` → final
   ``Linear`` producing 2 dimensions (no final activation).

The output is the ``[N, 2]`` **standardized label values (z-score space, not
physical units)** for ``alpha`` and ``Ku``; restoring physical units relies on
the preprocessing statistics saved by the training side and is not this
module's responsibility.

``padding=(k-1)//2`` uses default zero padding, so sequence edges and pooling
bin boundaries may have boundary effects. This module only extracts features:
it does not normalize, read data, or import any training module.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["CNN1DRegressor"]

_N_OUTPUTS = 2  # output columns: (alpha, Ku) standardized labels
_N_CHANNELS = 3  # magnetization component count (mx, my, mz)


def _require_positive_int(value: object, field: str) -> int:
    """Positive integer: rejects bool/float/string; a non-positive value raises ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer (got {value!r})")
    return value


def _require_int_sequence(value: object, field: str, *, allow_empty: bool) -> tuple[int, ...]:
    """Integer sequence: accepts tuple/list and normalizes to tuple; None/scalar/string
    raises ValueError."""
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field} must be an integer sequence (got {value!r})")
    sequence = tuple(value)
    if not sequence and not allow_empty:
        raise ValueError(f"{field} must be a non-empty integer sequence (got {value!r})")
    for index, item in enumerate(sequence):
        _require_positive_int(item, f"{field}[{index}]")
    return sequence


class CNN1DRegressor(nn.Module):
    """A 1D CNN regressor that inverts (alpha, Ku) from multi-excitation magnetization trajectories.

    Attributes:
        input_shape: normalized ``(P, T, 3)``.
        channels: output channel count of each Conv1d layer (tuple).
        kernel_sizes: kernel length of each Conv1d layer (tuple, odd, same length as channels).
        pool_bins: ``AdaptiveAvgPool1d`` output bin count (``1 <= pool_bins <= T``).
        head_hidden_dims: regression head hidden widths (tuple, may be empty).
        features: ``nn.Sequential``: Conv1d/ReLU stack.
        pool: ``nn.AdaptiveAvgPool1d``.
        head: ``nn.Sequential``: Flatten + Linear/ReLU + final Linear.
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
        """Initialize the network layers (no default hyperparameters; all are given explicitly
        by the caller).

        Args:
            input_shape: ``(P, T, 3)``, three positive integers with the last dimension always 3.
            channels: each Conv1d layer's output channel count, non-empty positive int sequence.
            kernel_sizes: each Conv1d layer's kernel length, non-empty positive odd sequence,
                same length as channels.
            pool_bins: positive integer and ``<= T`` (1 and T are allowed).
            head_hidden_dims: regression head hidden widths, may be empty; each element a
                positive integer.

        Raises:
            ValueError: any of the above constraints is violated (including bool/float/string
                masquerading as integers, layer-count mismatch, even kernels,
                ``pool_bins > T``, etc.).
        """
        super().__init__()
        shape = _require_int_sequence(input_shape, "input_shape", allow_empty=False)
        if len(shape) != 3:
            raise ValueError(f"input_shape must be exactly 3D (P, T, 3) (got {input_shape!r})")
        if shape[2] != _N_CHANNELS:
            raise ValueError(
                f"input_shape last dimension must be {_N_CHANNELS} (mx,my,mz) (got {shape[2]})"
            )
        n_pulse, n_time = shape[0], shape[1]

        normalized_channels = _require_int_sequence(channels, "channels", allow_empty=False)
        normalized_kernels = _require_int_sequence(kernel_sizes, "kernel_sizes", allow_empty=False)
        if len(normalized_kernels) != len(normalized_channels):
            raise ValueError(
                f"kernel_sizes layer count must match channels "
                f"(got {len(normalized_kernels)} vs {len(normalized_channels)})"
            )
        if any(kernel % 2 == 0 for kernel in normalized_kernels):
            raise ValueError(f"kernel_sizes must all be odd (got {kernel_sizes!r})")
        normalized_pool_bins = _require_positive_int(pool_bins, "pool_bins")
        if normalized_pool_bins > n_time:
            raise ValueError(f"pool_bins must be <= T={n_time} (got {normalized_pool_bins})")
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
        """Forward pass: ``[N, P, T, 3] → [N, 2]`` standardized label predictions.

        Args:
            x: input batch whose shape must match ``input_shape``; non-contiguous
                tensors are allowed (normalized internally via ``permute + contiguous``).

        Returns:
            standardized label predictions of shape ``[N, 2]`` (z-score space, not physical units).

        Raises:
            ValueError: ``x`` is not 4D, or ``x.shape[1:]`` does not match
                ``input_shape``.
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"input shape must be [N, {self.input_shape}] and match input_shape "
                f"(got {tuple(x.shape)})"
            )
        batch = x.shape[0]
        n_pulse, n_time, _ = self.input_shape
        x = x.permute(0, 1, 3, 2).contiguous().reshape(batch, n_pulse * _N_CHANNELS, n_time)
        return self.head(self.pool(self.features(x)))
