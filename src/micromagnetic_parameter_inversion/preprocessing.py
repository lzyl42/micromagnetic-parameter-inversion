"""Input/label preprocessing: statistics and transforms fitted on the train split only.

Core semantics (transforms and ckpt/preprocessing.yaml serialization must use
the same semantics on read and write):

- Input normalization aggregates over the train split only, along the sample
  dimension and the time axis → mean/std per pulse position and per
  magnetization component, shape ``[P, 1, 3]``, broadcast over ``[P, T, 3]``
  or ``[N, P, T, 3]``. Statistics accumulate stably in float64 using chunked
  Welford merging (population std, ddof=0), streaming sample by sample without
  stacking the whole dataset or copying all data.
- ``std <= eps`` → divisor 1. **``InputStats.std`` and ``LabelState.y_std``
  store the effective scale (the actual divisor)**: positions with ``> eps``
  keep the raw std (ddof=0), positions with ``<= eps`` are always 1.0, and the
  raw std itself is not stored; replaced positions are recorded by the
  ``zero_variance_positions``/``zero_variance_outputs`` lists. Serialization
  (ckpt / preprocessing.yaml) stores mean, effective scale, eps, and the
  zero-variance lists as-is with the same semantics, so transforms after
  restore behave exactly as at fit time. Note that zero-variance positions do
  not guarantee a zero transform result on val/test (statistics do not come
  from those splits).
- Labels: ``identity`` | ``logalpha`` (log10 on the alpha column only, requires
  alpha > 0 with no silent clipping; non-finite values always raise), followed
  by per-output z-score (fitted on train); val/test reuse the same state; the
  inverse transform restores physical units (alpha, ku_j_per_m3), and
  non-finite inputs/outputs (e.g. model predictions of NaN/Inf) raise
  explicitly — the evaluation boundary relies on this and does not treat bad
  values as an empty subset. log10 is always computed in **float64** (alpha is
  taken from a float64 copy of the input): computing log10 directly on a
  float32 input would disagree with the float64 fit statistics, and in
  narrow-variance scenarios float32 rounding noise would be amplified by the
  small divisor.
- All transforms output float32 with a **finiteness contract**: the final
  float64 → float32 cast runs under ``np.errstate``, and overflow/invalid →
  ``PreprocessingError`` (never silently returning inf/NaN). State dataclasses
  contain ndarray fields, so ``eq=False`` (structural ndarray equality would be
  ambiguous; states are compared by object identity).

Dependency direction: training_config → training_data → (this module) →
training / evaluation; this module does not import the model/training modules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from micromagnetic_parameter_inversion.training_config import LabelConfig, PreprocessingConfig
from micromagnetic_parameter_inversion.training_data import TrajectoryDataset

# Number of magnetization components (mx, my, mz): the input contract is always 3.
_N_COMPONENTS = 3


class PreprocessingError(ValueError):
    """Preprocessing contract violation (e.g. logalpha with alpha <= 0; no silent clipping)."""


@dataclass(frozen=True, eq=False)
class InputStats:
    """Normalization state for x (fitted on the train split only).

    Attributes:
        mean: float64 ``[P, 1, 3]``: mean aggregated along the sample dimension
            and the time axis.
        std: float64 ``[P, 1, 3]``: **effective scale (actual divisor)** — the
            raw std (ddof=0) is stored as-is where ``> eps``; where ``<= eps``
            it is replaced by 1.0 and the raw value is not stored.
            ``transform_x`` uses this field directly as the divisor;
            serialization writes it as-is with the same semantics.
        eps: threshold for ``std <= eps → divisor 1`` (PreprocessingConfig.std_eps).
        zero_variance_positions: list of ``(pulse position, component)`` pairs
            whose divisor was replaced by 1 (all positions with raw std <= eps,
            sorted by position).
    """

    mean: np.ndarray
    std: np.ndarray
    eps: float
    zero_variance_positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True, eq=False)
class LabelState:
    """Transform and normalization state for y (fitted on train only; saved with the ckpt).

    Attributes:
        transform: ``"identity" | "logalpha"`` (training_config.LabelTransform).
        y_mean: float64 ``[2]``: per-output mean in the transformed space.
        y_std: float64 ``[2]``: **effective scale (actual divisor)**, same
            semantics as ``InputStats.std``.
        eps: protection threshold for divisor 1 (prevents division by zero on
            val/test when an output has zero variance within train).
        zero_variance_outputs: list of output columns whose divisor was replaced
            by 1 (0=alpha, 1=ku_j_per_m3, ascending).
    """

    transform: str
    y_mean: np.ndarray
    y_std: np.ndarray
    eps: float
    zero_variance_outputs: tuple[int, ...] = ()


@dataclass(frozen=True, eq=False)
class PreprocessingState:
    """Complete preprocessing state saved in the ckpt; evaluate always restores it from the ckpt."""

    x_stats: InputStats
    y_stats: LabelState


def _validate_eps(eps: float) -> float:
    """std_eps must be a positive finite value (consistent with training_config.load_config)."""
    value = float(eps)
    if not math.isfinite(value) or value <= 0.0:
        raise PreprocessingError(f"std_eps must be a positive finite value (got {eps!r})")
    return value


def _validate_label_transform(transform: str) -> str:
    if transform not in ("identity", "logalpha"):
        raise PreprocessingError(
            f"unknown label.transform {transform!r}; allowed 'identity' | 'logalpha'"
        )
    return transform


def _apply_label_transform(transform: str, y: np.ndarray) -> np.ndarray:
    """Label transform (float64 semantics): identity passes through; logalpha takes
    log10 of the alpha column only.

    Non-finite values or logalpha with alpha <= 0 → PreprocessingError (no silent
    clipping).
    """
    if not np.all(np.isfinite(y)):
        raise PreprocessingError(
            f"labels contain non-finite values (NaN/Inf); cannot apply {transform!r} transform"
        )
    if transform == "identity":
        return y
    if transform == "logalpha":
        # Uniform float64 precision: take alpha from a float64 copy before log10.
        # Computing directly on a float32 input would give only float32
        # precision, disagreeing with the float64 fit statistics — in
        # narrow-variance scenarios (adjacent float32 alphas), rounding noise
        # would be amplified by the small divisor (normalized values should be
        # about [-1, 1], observed about [-4.1, 3.2]).
        transformed = np.array(y, dtype=np.float64)
        alpha = transformed[..., 0]
        if np.any(alpha <= 0.0):
            raise PreprocessingError(
                f"logalpha requires alpha > 0 (no silent clipping) "
                f"(got minimum {float(np.min(alpha))!r})"
            )
        transformed[..., 0] = np.log10(alpha)
        return transformed
    raise PreprocessingError(
        f"unknown label.transform {transform!r}; allowed 'identity' | 'logalpha'"
    )


def _label_to_physical(transform: str, y_transformed: np.ndarray) -> np.ndarray:
    """Transformed labels → physical units (alpha, ku_j_per_m3) (float64)."""
    if transform == "identity":
        return y_transformed
    if transform == "logalpha":
        physical = np.array(y_transformed, dtype=np.float64)
        physical[..., 0] = 10.0 ** physical[..., 0]
        return physical
    raise PreprocessingError(
        f"unknown label.transform {transform!r}; allowed 'identity' | 'logalpha'"
    )


def _to_float32_or_raise(values: np.ndarray, name: str) -> np.ndarray:
    """Final float64 → float32 cast; overflow/invalid → PreprocessingError (finiteness contract).

    A value finite in float64 (e.g. ``1e40``) may overflow to inf in float32;
    the contract requires the **final float32 output** to be finite and never
    silently returns inf/NaN. ``np.errstate`` only suppresses the overflow/
    invalid warnings during the cast; an explicit error follows.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise PreprocessingError(
            f"{name} exceeds the float32 representable range (overflow to NaN/Inf)"
        )
    return result


def _as_input_array(x: np.ndarray, n_pulse: int) -> np.ndarray:
    """Validate that x has shape ``[P,T,3]`` or ``[N,P,T,3]`` (P consistent with the statistics)."""
    arr = np.asarray(x)
    expected = f"[P,T,3] or [N,P,T,3] (P={n_pulse}, channels={_N_COMPONENTS})"
    if arr.ndim == 3:
        if arr.shape[0] != n_pulse or arr.shape[-1] != _N_COMPONENTS:
            raise PreprocessingError(f"x shape must be {expected} (got shape {tuple(arr.shape)})")
        return arr
    if arr.ndim == 4:
        if arr.shape[1] != n_pulse or arr.shape[-1] != _N_COMPONENTS:
            raise PreprocessingError(f"x shape must be {expected} (got shape {tuple(arr.shape)})")
        return arr
    raise PreprocessingError(f"x shape must be {expected} (got shape {tuple(arr.shape)})")


def _as_label_array(y: np.ndarray, name: str) -> np.ndarray:
    """Validate that labels have shape ``[2]`` or ``[N, 2]``."""
    arr = np.asarray(y)
    if (arr.ndim == 1 and arr.shape[0] == 2) or (arr.ndim == 2 and arr.shape[1] == 2):
        return arr
    raise PreprocessingError(f"{name} shape must be [2] or [N,2] (got shape {tuple(arr.shape)})")


def _validate_sample(
    index: int,
    x64: np.ndarray,
    y64: np.ndarray,
    ref_shape: tuple[int, int, int] | None,
) -> None:
    """Required per-sample read checks: x ``[P,T,3]`` (positive P/T, consistent across
    samples), y ``[2]``.
    """
    if x64.ndim != 3 or x64.shape[-1] != _N_COMPONENTS:
        raise PreprocessingError(
            f"train sample {index} x shape must be [P,T,3] (channels={_N_COMPONENTS}) "
            f"(got shape {tuple(x64.shape)})"
        )
    if x64.shape[0] < 1 or x64.shape[1] < 1:
        raise PreprocessingError(
            f"train sample {index} P/T must be positive (got shape {tuple(x64.shape)})"
        )
    if ref_shape is not None and x64.shape != ref_shape:
        raise PreprocessingError(
            f"train sample {index} x shape {tuple(x64.shape)} does not match the first "
            f"sample {ref_shape}"
        )
    if y64.shape != (2,):
        raise PreprocessingError(
            f"train sample {index} y shape must be [2] (got shape {tuple(y64.shape)})"
        )


def fit(
    train_set: TrajectoryDataset,
    config: PreprocessingConfig,
    label: LabelConfig,
) -> PreprocessingState:
    """Fit preprocessing state on the train split only (no eval involvement; leakage guard).

    Args:
        train_set: ``TrajectoryDataset`` of the **train split**; items are raw
            unnormalized ``SampleItem = (x, y, psid)`` with float32 x/y (Tensor
            or equivalent numpy array, aggregated uniformly via ``np.asarray``).
            Statistics from val/test must not enter this function; psid does not
            participate in the statistics.
        config: provides std_eps (``std <= eps → divisor 1``; must be a positive
            finite value, consistent with ``training_config.load_config``).
        label: provides label.transform (identity | logalpha).

    Returns:
        ``PreprocessingState``: x statistics float64 ``[P, 1, 3]`` and label
        statistics float64 ``[2]``; ``std``/``y_std`` are effective scales (see
        their docstrings), and zero-variance positions/outputs are listed.

    Raises:
        PreprocessingError: train set is empty; a sample x shape is not
            ``[P,T,3]``, P/T is non-positive, or samples disagree; a y shape is
            not ``[2]``; x/y contain non-finite values; logalpha with
            alpha <= 0; illegal eps/transform configuration.

    Implementation: streaming per sample, with a two-pass method per time axis
    to obtain chunk (mean, M2) and then Chan et al.'s parallel formula to merge
    into global float64 accumulators (population std, ddof=0, numerically
    stable), without stacking the whole dataset.
    """
    eps = _validate_eps(config.std_eps)
    transform = _validate_label_transform(label.transform)

    count = 0  # observations per [P,3] position = seen samples × T
    mean: np.ndarray | None = None  # float64 [P,3]
    m2: np.ndarray | None = (
        None  # float64 [P,3]: sum of squared deviations (Welford merge quantity)
    )
    y_rows: list[np.ndarray] = []
    ref_shape: tuple[int, int, int] | None = None

    for index in range(len(train_set)):
        x, y, _psid = train_set[index]
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        _validate_sample(index, x64, y64, ref_shape)
        if ref_shape is None:
            ref_shape = (x64.shape[0], x64.shape[1], x64.shape[2])
        if not np.all(np.isfinite(x64)):
            raise PreprocessingError(f"train sample {index} x contains non-finite values (NaN/Inf)")
        # Chunk statistics (two-pass over the time axis, stable), merged into the
        # global Welford accumulators.
        n_t = x64.shape[1]
        chunk_mean = x64.mean(axis=1)  # [P,3]
        chunk_m2 = ((x64 - chunk_mean[:, None, :]) ** 2).sum(axis=1)  # [P,3]
        if mean is None or m2 is None:
            mean, m2, count = chunk_mean, chunk_m2, n_t
        else:
            total = count + n_t
            delta = chunk_mean - mean
            mean = mean + delta * (n_t / total)
            m2 = m2 + chunk_m2 + delta * delta * ((count * n_t) / total)
            count = total
        y_rows.append(y64)

    if mean is None or m2 is None or ref_shape is None:
        raise PreprocessingError(
            "train set is empty: preprocessing statistics require at least 1 sample"
        )

    y_all = np.stack(y_rows, axis=0)  # [N,2] float64 (labels only, negligible cost)
    y_transformed = _apply_label_transform(transform, y_all)

    std_raw = np.sqrt(m2 / count)  # [P,3]: population std (ddof=0)
    zero_mask = std_raw <= eps
    scale = np.where(zero_mask, 1.0, std_raw)
    rows, cols = np.nonzero(zero_mask)
    zero_positions = tuple(
        (row, col) for row, col in zip(rows.tolist(), cols.tolist(), strict=True)
    )

    y_std_raw = y_transformed.std(axis=0, ddof=0)  # [2]
    y_zero_mask = y_std_raw <= eps
    y_scale = np.where(y_zero_mask, 1.0, y_std_raw)
    y_zero_outputs = tuple(np.nonzero(y_zero_mask)[0].tolist())

    return PreprocessingState(
        x_stats=InputStats(
            mean=mean[:, None, :],
            std=scale[:, None, :],
            eps=eps,
            zero_variance_positions=zero_positions,
        ),
        y_stats=LabelState(
            transform=transform,
            y_mean=y_transformed.mean(axis=0),
            y_std=y_scale,
            eps=eps,
            zero_variance_outputs=y_zero_outputs,
        ),
    )


def transform_x(state: PreprocessingState, x: np.ndarray) -> np.ndarray:
    """Broadcast-normalize x with the ``[P, 1, 3]`` statistics (val/test reuse the same state).

    Args:
        state: preprocessing state produced by ``fit`` (or restored from a ckpt).
        x: raw input ``[P, T, 3]`` or batch ``[N, P, T, 3]``; T may differ from
            fit time (statistics aggregate over the time axis, so broadcasting
            is independent of T).

    Returns:
        float32 array (same shape as the input): ``(x - mean) / effective_scale``;
        the divisor is 1 at zero-variance positions (the result is ``x - mean``,
        which is 0 for a constant value within train).

    Raises:
        PreprocessingError: the shape does not match the state's ``[P, 1, 3]``
            statistics; the normalized result (including propagation of
            non-finite inputs) exceeds the float32 representable range.
    """
    arr = _as_input_array(x, state.x_stats.mean.shape[0])
    out = np.array(arr, dtype=np.float64)  # defensive copy: in-place ops do not touch caller data
    out -= state.x_stats.mean
    out /= state.x_stats.std  # effective scale: zero-variance positions divide by 1
    return _to_float32_or_raise(out, "x normalization result")


def transform_y(state: PreprocessingState, y: np.ndarray) -> np.ndarray:
    """Label transform + z-score (same internal path as ``fit``; val/test reuse the same state).

    Args:
        state: preprocessing state produced by ``fit`` (or restored from a ckpt).
        y: physical-unit labels ``[2]`` or ``[N, 2]`` (column order alpha, ku_j_per_m3).

    Returns:
        float32 array (same shape as the input): transformed ``(y' - y_mean) / y_std``,
        where ``y_std`` is the effective scale.

    Raises:
        PreprocessingError: illegal shape, non-finite labels, logalpha with
            alpha <= 0 (no silent clipping), or a normalized result outside the
            float32 representable range.
    """
    _validate_label_transform(state.y_stats.transform)
    arr = _as_label_array(y, "y")
    transformed = _apply_label_transform(state.y_stats.transform, arr)
    out = (transformed - state.y_stats.y_mean) / state.y_stats.y_std
    return _to_float32_or_raise(out, "label normalization result")


def inverse_transform_y(state: PreprocessingState, y_normalized: np.ndarray) -> np.ndarray:
    """Normalized labels → physical units (alpha, ku_j_per_m3) (for evaluation output).

    Inverse path: reverse the z-score per output (``y_norm * y_std + y_mean``,
    with ``y_std`` the effective scale, so zero-variance outputs return to
    ``y_mean``); identity ends here, while logalpha additionally applies
    ``10 ** alpha'`` to the alpha column.

    Args:
        state: preprocessing state produced by ``fit`` (or restored from a ckpt).
        y_normalized: normalized-space predictions ``[2]`` or ``[N, 2]``.

    Returns:
        float32 array (same shape as the input): physical units (alpha, ku_j_per_m3).

    Raises:
        PreprocessingError: the input/result contains non-finite values (model
            predictions of NaN/Inf raise explicitly at this boundary and are not
            treated as an empty subset); a finite float64 result exceeds the
            float32 representable range (e.g. ``10 ** 40``); or the shape is
            illegal.
    """
    _validate_label_transform(state.y_stats.transform)
    arr = _as_label_array(y_normalized, "y_normalized")
    if not np.all(np.isfinite(arr)):
        raise PreprocessingError(
            "inverse-transform input contains non-finite predictions (NaN/Inf); "
            "check the model output"
        )
    restored = np.array(arr, dtype=np.float64)
    restored *= state.y_stats.y_std  # effective scale
    restored += state.y_stats.y_mean
    physical = _label_to_physical(state.y_stats.transform, restored)
    if not np.all(np.isfinite(physical)):
        raise PreprocessingError(
            "inverse-transform result contains non-finite values (NaN/Inf); "
            "physical-unit restore overflowed or statistics are abnormal"
        )
    return _to_float32_or_raise(physical, "inverse-transform result")


def state_to_mapping(state: PreprocessingState) -> dict[str, Any]:
    """PreprocessingState → nested plain dict (the only mapping schema owned by this module).

    mean/effective scale are preserved exactly in float64 via Python floats
    (``tolist``), and the zero-variance lists are int pairs; all containers are
    primitives/list/dict with no ndarray/dataclass objects — the ckpt and
    preprocessing.yaml share this form.
    """
    x_stats = state.x_stats
    y_stats = state.y_stats
    return {
        "x_stats": {
            "mean": np.asarray(x_stats.mean, dtype=np.float64).tolist(),
            "std": np.asarray(x_stats.std, dtype=np.float64).tolist(),
            "eps": float(x_stats.eps),
            "zero_variance_positions": [
                [int(row), int(col)] for row, col in x_stats.zero_variance_positions
            ],
        },
        "y_stats": {
            "transform": str(y_stats.transform),
            "y_mean": np.asarray(y_stats.y_mean, dtype=np.float64).tolist(),
            "y_std": np.asarray(y_stats.y_std, dtype=np.float64).tolist(),
            "eps": float(y_stats.eps),
            "zero_variance_outputs": [int(o) for o in y_stats.zero_variance_outputs],
        },
    }


def state_from_mapping(mapping: Mapping[str, Any]) -> PreprocessingState:
    """Nested plain dict → PreprocessingState (the only entry point for rebuilding the ckpt /
    preprocessing.yaml state).

    Symmetric with ``state_to_mapping``; the effective scale semantics are
    restored as-is (no refitting). Cross-checking the P dimension against the
    input contract is the caller's responsibility (training boundary).

    Raises:
        PreprocessingError: non-mapping, missing x_stats/y_stats, a shape other
            than ``[P,1,3]`` (P>=1, matching mean/std) or ``[2]``, an illegal
            transform, or a non-positive/non-finite eps.
    """
    if not isinstance(mapping, Mapping):
        raise PreprocessingError(f"preprocessing mapping must be a dict (got {type(mapping)!r})")
    try:
        x_raw = mapping["x_stats"]
        y_raw = mapping["y_stats"]
        if not isinstance(x_raw, Mapping) or not isinstance(y_raw, Mapping):
            raise PreprocessingError("preprocessing mapping is missing x_stats/y_stats")
        mean = np.asarray(x_raw["mean"], dtype=np.float64)
        std = np.asarray(x_raw["std"], dtype=np.float64)
        if mean.ndim != 3 or mean.shape[1:] != (1, 3) or std.shape != mean.shape:
            raise PreprocessingError(
                f"x statistics shape must be [P,1,3] with matching mean/std "
                f"(got mean {mean.shape} / std {std.shape})"
            )
        if mean.shape[0] < 1:
            raise PreprocessingError(f"x statistics P must be positive (got {mean.shape[0]})")
        y_mean = np.asarray(y_raw["y_mean"], dtype=np.float64)
        y_std = np.asarray(y_raw["y_std"], dtype=np.float64)
        if y_mean.shape != (2,) or y_std.shape != (2,):
            raise PreprocessingError(
                f"y statistics shape must be (2,) (got y_mean {y_mean.shape} / y_std {y_std.shape})"
            )
        transform = _validate_label_transform(str(y_raw["transform"]))
        return PreprocessingState(
            x_stats=InputStats(
                mean=mean,
                std=std,
                eps=_validate_eps(x_raw["eps"]),
                zero_variance_positions=tuple(
                    (int(row), int(col)) for row, col in x_raw.get("zero_variance_positions") or []
                ),
            ),
            y_stats=LabelState(
                transform=transform,
                y_mean=y_mean,
                y_std=y_std,
                eps=_validate_eps(y_raw["eps"]),
                zero_variance_outputs=tuple(
                    int(o) for o in y_raw.get("zero_variance_outputs") or []
                ),
            ),
        )
    except PreprocessingError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PreprocessingError(f"corrupted preprocessing mapping: {exc}") from exc
