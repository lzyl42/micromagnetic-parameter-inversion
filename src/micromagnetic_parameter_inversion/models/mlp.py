"""MLP inverse-regression model.

Input contract ``input_shape = (P, T, 3)``: a single sample is the raw
time-domain trajectory of all pulses in one parameter set, with batch shape
``[N, P, T, 3]`` (P pulses, T time steps, 3 magnetization components
mx/my/mz). No per-pulse splitting, no hand-crafted statistical features, no
downsampling.

Output semantics: shape ``[N, 2]``, where the two columns are the
**normalized label values (after z-score)** of ``alpha`` and ``Ku``,
**not physical units**; restoring physical units relies on the normalization
statistics saved by the preprocessing module.

This module does not read data, does not perform any normalization, and does
not import other training modules (training_config / training_data /
preprocessing / training, etc.) to avoid circular dependencies.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["MLPRegressor"]

_N_OUTPUTS = 2  # output columns: normalized labels for (alpha, ku_j_per_m3)
_N_CHANNELS = 3  # number of magnetization components (mx, my, mz): the input contract is always 3


class MLPRegressor(nn.Module):
    """MLP regressor that inverts (alpha, Ku) from multi-excitation magnetization trajectories.

    ``input_shape``/``hidden_dims`` are stored on the instance and are the
    source of the checkpoint structure fields; the flattened dimension is
    ``D = P * T * 3``.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dims: tuple[int, ...] = (64, 32, 32),
    ) -> None:
        """Initialize the network layers.

        Args:
            input_shape: ``(P, T, 3)``; all dimensions must be positive and the
                channel dimension is always 3.
            hidden_dims: sequence of hidden widths (may be empty = a direct
                ``Linear D→2``); all entries must be positive; the final linear
                layer outputs normalized labels (not physical units).

        Raises:
            ValueError: ``input_shape`` is not a 3-tuple, P/T is non-positive, or
                the channel dimension is not 3; or ``hidden_dims`` contains a
                non-positive width.
        """
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (P, T, 3) (got {tuple(input_shape)})")
        n_pulse, n_time, n_channels = input_shape
        if n_pulse < 1 or n_time < 1:
            raise ValueError(f"input_shape P/T must be positive (got {tuple(input_shape)})")
        if n_channels != _N_CHANNELS:
            raise ValueError(
                f"input_shape channel dimension is always {_N_CHANNELS} "
                f"(mx,my,mz) (got {n_channels})"
            )
        if any(width < 1 for width in hidden_dims):
            raise ValueError(f"hidden_dims entries must be positive (got {tuple(hidden_dims)})")

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
        """Forward pass: ``[N, P, T, 3] → [N, 2]`` normalized-label predictions.

        Args:
            x: input batch of shape ``[N, P, T, 3]`` — raw time-domain
                trajectories normalized with the training-split statistics
                (normalization is handled by the preprocessing module, not here).

        Returns:
            Tensor of shape ``[N, 2]``: normalized-label predictions for
            ``alpha`` and ``Ku`` (z-score space, not physical units).

        Raises:
            ValueError: ``x`` is not 4-D, or dimensions 1-3 disagree with ``input_shape``.
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"input shape must be [N, P, T, 3] and match input_shape={self.input_shape} "
                f"(got {tuple(x.shape)})"
            )
        return self.network(x)
