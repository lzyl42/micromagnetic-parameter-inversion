"""preprocessing unit tests: CPU synthetic data + stub Dataset (never touches real raw).

Coverage: train-only fitting (val statistics do not affect the output),
``[P,1,3]`` broadcast shape, divisor 1 at zero-variance positions for x/y,
label transform/inverse roundtrip (identity/logalpha), logalpha with
alpha <= 0 raising (no silent clipping), non-finite values raising, batch
broadcast and float32 output, logalpha float64 uniform precision
(narrow-variance regression with adjacent float32 alphas), float32 overflow
finiteness contract (transform_x/transform_y/inverse_transform_y never
silently return inf), and state ``eq=False`` semantics.

The stub Dataset implements only ``__len__``/``__getitem__`` and returns
``SampleItem`` (the TrajectoryDataset business constructor is unavailable and
not needed).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from micromagnetic_parameter_inversion.preprocessing import (
    InputStats,
    LabelState,
    PreprocessingError,
    PreprocessingState,
    fit,
    inverse_transform_y,
    state_from_mapping,
    state_to_mapping,
    transform_x,
    transform_y,
)
from micromagnetic_parameter_inversion.training_config import (
    LabelConfig,
    LabelTransform,
    PreprocessingConfig,
)
from micromagnetic_parameter_inversion.training_data import SampleItem, TrajectoryDataset


class _StubTrainSet(TrajectoryDataset):
    """Minimal stub: items are ``SampleItem`` (torch Tensor)."""

    def __init__(self, samples: list[SampleItem]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SampleItem:
        return self._samples[index]


class _NumpyElementTrainSet(TrajectoryDataset):
    """Stub with equivalent numpy arrays (fit must aggregate uniformly via ``np.asarray``)."""

    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray, str]]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SampleItem:
        x, y, psid = self._samples[index]
        return cast(SampleItem, (x, y, psid))


def _make_x(n: int, n_pulse: int, n_time: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_pulse, n_time, 3)).astype(np.float32)


def _make_y(n: int, seed: int = 0) -> np.ndarray:
    """Physical-unit labels [N,2]: (alpha ~ 0.01-0.1, ku_j_per_m3 ~ 1e4-1e5)."""
    rng = np.random.default_rng(seed + 1000)
    alpha = rng.uniform(0.01, 0.1, size=(n, 1))
    ku = rng.uniform(1.0e4, 1.0e5, size=(n, 1))
    return np.concatenate([alpha, ku], axis=1).astype(np.float32)


def _tensor_dataset(x: np.ndarray, y: np.ndarray) -> _StubTrainSet:
    samples: list[SampleItem] = [
        (torch.from_numpy(x[i]), torch.from_numpy(y[i]), f"psid_{i}") for i in range(x.shape[0])
    ]
    return _StubTrainSet(samples)


def _reference_input_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference mean/std aggregated over samples and time (float64, ddof=0), shape [P,1,3]."""
    x64 = x.astype(np.float64)
    return (
        x64.mean(axis=(0, 2))[:, None, :],
        x64.std(axis=(0, 2), ddof=0)[:, None, :],
    )


def test_fit_input_stats_match_reference() -> None:
    x = _make_x(n=5, n_pulse=3, n_time=7, seed=1)
    state = fit(_tensor_dataset(x, _make_y(5, seed=1)), PreprocessingConfig(), LabelConfig())
    mean, std = _reference_input_stats(x)
    assert state.x_stats.mean.shape == (3, 1, 3)
    assert state.x_stats.std.shape == (3, 1, 3)
    assert state.x_stats.mean.dtype == np.float64
    assert state.x_stats.std.dtype == np.float64
    np.testing.assert_allclose(state.x_stats.mean, mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(state.x_stats.std, std, rtol=1e-12, atol=1e-14)
    assert state.x_stats.zero_variance_positions == ()
    assert state.x_stats.eps == 1.0e-8


def test_fit_uses_train_only_statistics() -> None:
    x_train = _make_x(4, 2, 5, seed=2)
    x_val = _make_x(3, 2, 5, seed=3) + 10.0  # val statistics clearly differ
    state = fit(_tensor_dataset(x_train, _make_y(4, seed=2)), PreprocessingConfig(), LabelConfig())
    mean, std = _reference_input_stats(x_train)
    np.testing.assert_allclose(state.x_stats.mean, mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(state.x_stats.std, std, rtol=1e-12, atol=1e-14)
    # val reuses train statistics: the +10 offset is not absorbed to 0 by the train mean.
    expected = (x_val.astype(np.float64) - mean) / std
    np.testing.assert_allclose(
        transform_x(state, x_val), expected.astype(np.float32), rtol=1e-6, atol=1e-6
    )


def test_transform_x_broadcast_single_and_batch() -> None:
    x = _make_x(4, 2, 5, seed=4)
    state = fit(_tensor_dataset(x, _make_y(4, seed=4)), PreprocessingConfig(), LabelConfig())
    single = transform_x(state, x[0])
    batch = transform_x(state, x)
    assert single.shape == (2, 5, 3)
    assert single.dtype == np.float32
    assert batch.shape == (4, 2, 5, 3)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch[0], single, rtol=1e-7)
    # after normalizing the training data, population mean ≈ 0 and population std ≈ 1.
    flat = batch.astype(np.float64)
    assert abs(float(flat.mean())) < 1e-6
    assert abs(float(flat.std(ddof=0)) - 1.0) < 1e-6


def test_transform_x_broadcast_independent_of_time_length() -> None:
    x = _make_x(3, 2, 5, seed=5)
    state = fit(_tensor_dataset(x, _make_y(3, seed=5)), PreprocessingConfig(), LabelConfig())
    probe = np.zeros((2, 9, 3), dtype=np.float32)
    out = transform_x(state, probe)
    assert out.shape == (2, 9, 3)
    expected = (probe.astype(np.float64) - state.x_stats.mean) / state.x_stats.std
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-7)


def test_zero_variance_divisor_is_one_and_masked() -> None:
    x = _make_x(4, 2, 5, seed=6)
    x[:, 1, :, 2] = 0.5  # pulse position 1, component 2 constant → std=0 <= eps → divisor 1
    state = fit(_tensor_dataset(x, _make_y(4, seed=6)), PreprocessingConfig(), LabelConfig())
    assert state.x_stats.zero_variance_positions == ((1, 2),)
    assert state.x_stats.std[1, 0, 2] == 1.0  # effective scale
    # a constant value within train transforms to 0; (v - mean)/1 stays finite outside train
    np.testing.assert_allclose(transform_x(state, x)[:, 1, :, 2], 0.0, atol=1e-7)
    probe = np.zeros((1, 2, 5, 3), dtype=np.float32)
    probe[:, 1, :, 2] = 0.7
    out = transform_x(state, probe)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[:, 1, :, 2], 0.2, atol=1e-6)


def test_std_above_eps_kept_real_and_below_eps_masked() -> None:
    rng = np.random.default_rng(7)
    x = _make_x(3, 2, 2000, seed=8)
    x[:, 0, :, 0] = rng.normal(0.0, 1.0e-4, size=(3, 2000))  # std ~1e-4 <= eps
    x[:, 1, :, 1] = rng.normal(0.0, 1.0e-2, size=(3, 2000))  # std ~1e-2 > eps
    config = PreprocessingConfig(std_eps=1.0e-3)
    state = fit(_tensor_dataset(x, _make_y(3, seed=8)), config, LabelConfig())
    assert state.x_stats.zero_variance_positions == ((0, 0),)
    assert state.x_stats.std[0, 0, 0] == 1.0  # <= eps → divisor 1
    assert 5.0e-3 < state.x_stats.std[1, 0, 1] < 2.0e-2  # > eps → raw std kept
    assert state.x_stats.eps == 1.0e-3


def test_label_zero_variance_divisor_is_one() -> None:
    y = _make_y(4, seed=9)
    y[:, 0] = 0.05  # alpha column constant → std=0 <= eps → divisor 1
    x = _make_x(4, 2, 4, seed=9)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    assert state.y_stats.zero_variance_outputs == (0,)
    assert state.y_stats.y_std[0] == 1.0  # effective scale
    assert state.y_stats.y_std[1] > 1.0e-8
    y_norm = transform_y(state, y)
    np.testing.assert_allclose(y_norm[:, 0], 0.0, atol=1e-7)
    y_back = inverse_transform_y(state, y_norm)
    np.testing.assert_allclose(y_back[:, 0], y[:, 0], rtol=1e-6)


def test_label_identity_roundtrip_single_and_batch() -> None:
    x = _make_x(6, 2, 4, seed=10)
    y = _make_y(6, seed=10)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    assert state.y_stats.transform == "identity"
    y_norm = transform_y(state, y)
    assert y_norm.shape == (6, 2)
    assert y_norm.dtype == np.float32
    assert abs(float(y_norm.astype(np.float64).mean())) < 1e-6
    y_back = inverse_transform_y(state, y_norm)
    assert y_back.dtype == np.float32
    np.testing.assert_allclose(y_back, y, rtol=1e-5)
    # single sample [2].
    y_single = transform_y(state, y[0])
    assert y_single.shape == (2,)
    np.testing.assert_allclose(inverse_transform_y(state, y_single), y[0], rtol=1e-5)


def test_label_logalpha_roundtrip() -> None:
    x = _make_x(5, 2, 4, seed=11)
    y = _make_y(5, seed=11)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig(transform="logalpha"))
    assert state.y_stats.transform == "logalpha"
    y_norm = transform_y(state, y)
    assert y_norm.shape == (5, 2)
    y_back = inverse_transform_y(state, y_norm)
    np.testing.assert_allclose(y_back[:, 0], y[:, 0], rtol=1e-5)
    np.testing.assert_allclose(y_back[:, 1], y[:, 1], rtol=1e-5)


def test_logalpha_narrow_variance_matches_float64_reference() -> None:
    """Narrow-variance regression with adjacent float32 alphas: normalization must
    match the all-float64 reference (about [-1,1]).

    A former defect: transform_y computed log10 directly on float32 input,
    disagreeing with the float64 fit statistics — in the two-point scenario of
    0.05f and its next float32 neighbor upward (std ≈ 1.6e-8 > eps, raw small
    divisor kept) float32 rounding noise was amplified, giving normalized
    results of about [-4.127, 3.241] instead of [-1, 1].
    """
    alpha_lo = np.float32(0.05)
    alpha_hi = np.nextafter(alpha_lo, np.float32(1.0))
    assert alpha_hi > alpha_lo  # indeed the upward neighbor (1 ULP apart)
    y = np.array(
        [[float(alpha_lo), 2.0e4], [float(alpha_hi), 2.5e4]],
        dtype=np.float32,
    )
    x = _make_x(2, 2, 4, seed=30)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig(transform="logalpha"))
    # narrow-variance precondition: the alpha column std is a raw small divisor (not 1).
    assert 1.0e-8 < float(state.y_stats.y_std[0]) < 1.0e-6
    y_norm = transform_y(state, y)
    # all-float64 reference computation (expected after unifying precision).
    y64 = y.astype(np.float64)
    log_alpha = np.log10(y64[:, 0])
    ku = y64[:, 1]
    expected = np.stack(
        [
            (log_alpha - log_alpha.mean()) / log_alpha.std(ddof=0),
            (ku - ku.mean()) / ku.std(ddof=0),
        ],
        axis=1,
    )
    np.testing.assert_allclose(y_norm, expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    # the alpha column of a two-point population z-score should be about ±1 (not ±4).
    assert float(y_norm[0, 0]) == pytest.approx(-1.0, abs=1e-6)
    assert float(y_norm[1, 0]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("bad_alpha", [0.0, -0.05])
def test_fit_logalpha_rejects_nonpositive_alpha(bad_alpha: float) -> None:
    x = _make_x(3, 2, 4, seed=12)
    y = _make_y(3, seed=12)
    y[1, 0] = bad_alpha
    with pytest.raises(PreprocessingError, match="alpha > 0"):
        fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig(transform="logalpha"))


def test_transform_y_logalpha_rejects_nonpositive_alpha() -> None:
    x = _make_x(3, 2, 4, seed=13)
    y = _make_y(3, seed=13)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig(transform="logalpha"))
    y_bad = y.copy()
    y_bad[2, 0] = 0.0
    with pytest.raises(PreprocessingError, match="alpha > 0"):
        transform_y(state, y_bad)


def test_fit_rejects_nonfinite_x() -> None:
    x = _make_x(3, 2, 4, seed=14)
    x[2, 0, 1, 1] = np.inf
    with pytest.raises(PreprocessingError, match="sample 2"):
        fit(_tensor_dataset(x, _make_y(3, seed=14)), PreprocessingConfig(), LabelConfig())


def test_fit_rejects_nonfinite_labels() -> None:
    x = _make_x(3, 2, 4, seed=15)
    y = _make_y(3, seed=15)
    y[0, 1] = np.nan
    with pytest.raises(PreprocessingError, match="non-finite"):
        fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())


def test_inverse_transform_y_rejects_nonfinite_predictions() -> None:
    x = _make_x(2, 2, 4, seed=16)
    state = fit(_tensor_dataset(x, _make_y(2, seed=16)), PreprocessingConfig(), LabelConfig())
    with pytest.raises(PreprocessingError, match="non-finite"):
        inverse_transform_y(state, np.array([np.nan, 0.0], dtype=np.float32))
    with pytest.raises(PreprocessingError, match="non-finite"):
        inverse_transform_y(state, np.array([[0.0, np.inf]], dtype=np.float32))


def _hand_built_logalpha_state() -> PreprocessingState:
    """Minimal logalpha state with mean=0 / effective scale=1 (constructed directly,
    bypassing fit).
    """
    return PreprocessingState(
        x_stats=InputStats(
            mean=np.zeros((1, 1, 3), dtype=np.float64),
            std=np.ones((1, 1, 3), dtype=np.float64),
            eps=1.0e-8,
            zero_variance_positions=(),
        ),
        y_stats=LabelState(
            transform="logalpha",
            y_mean=np.zeros(2, dtype=np.float64),
            y_std=np.ones(2, dtype=np.float64),
            eps=1.0e-8,
            zero_variance_outputs=(),
        ),
    )


def test_inverse_transform_y_rejects_float32_overflow() -> None:
    """An inverse transform finite in float64 but overflowing float32 must raise, never
    silently return inf.

    Regression: finiteness was once checked on the float64 result before the
    float32 cast — ``10 ** 40`` is finite in float64 but overflows to inf in
    float32.
    """
    state = _hand_built_logalpha_state()
    with pytest.raises(PreprocessingError, match="float32"):
        inverse_transform_y(state, np.array([40.0, 0.0]))  # 10^40 → float32 inf
    with pytest.raises(PreprocessingError, match="float32"):
        inverse_transform_y(state, np.array([0.0, 1.0e39]))  # the Ku linear column overflows too
    # a boundary value representable in float32 still returns finite.
    out = inverse_transform_y(state, np.array([38.0, 0.0]))
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_transform_x_rejects_float32_overflow() -> None:
    """An x normalization result outside float32 (tiny raw std × large offset outside
    train) must raise.
    """
    rng = np.random.default_rng(33)
    x = _make_x(3, 2, 64, seed=33)
    x[:, 0, :, 0] = rng.normal(0.0, 1.0e-6, size=(3, 64))  # std ~1e-6 > eps → small divisor kept
    state = fit(_tensor_dataset(x, _make_y(3, seed=33)), PreprocessingConfig(), LabelConfig())
    assert 1.0e-8 < float(state.x_stats.std[0, 0, 0]) < 1.0e-5  # raw small divisor
    probe = np.zeros((1, 2, 4, 3), dtype=np.float32)
    probe[:, 0, :, 0] = np.float32(1.0e33)
    with pytest.raises(PreprocessingError, match="float32"):
        transform_x(state, probe)  # ~1e33 / 1e-6 ≈ 1e39 → float32 inf


def test_transform_y_rejects_float32_overflow() -> None:
    """A label normalization result outside float32 (small raw std × large val offset)
    must raise.
    """
    x = _make_x(3, 2, 4, seed=34)
    y = _make_y(3, seed=34)
    y[:, 0] = np.array([0.05, 0.05 + 1.0e-6, 0.05 - 1.0e-6], dtype=np.float32)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    assert 1.0e-8 < float(state.y_stats.y_std[0]) < 1.0e-5  # raw small divisor (not replaced by 1)
    with pytest.raises(PreprocessingError, match="float32"):
        transform_y(state, np.array([[1.0e33, 2.0e4]], dtype=np.float32))


def test_fit_rejects_inconsistent_sample_shapes() -> None:
    x = _make_x(2, 2, 4, seed=17)
    y = _make_y(2, seed=17)
    other = _make_x(1, 3, 4, seed=18)[0]  # P=3 ≠ first sample P=2
    ds = _StubTrainSet(
        [
            (torch.from_numpy(x[0]), torch.from_numpy(y[0]), "a"),
            (torch.from_numpy(other), torch.from_numpy(y[1]), "b"),
        ]
    )
    with pytest.raises(PreprocessingError, match="does not match"):
        fit(ds, PreprocessingConfig(), LabelConfig())


def test_fit_rejects_bad_label_shape() -> None:
    x = _make_x(1, 2, 4, seed=19)
    ds = _StubTrainSet([(torch.from_numpy(x[0]), torch.zeros(3), "a")])
    with pytest.raises(PreprocessingError, match=r"\[2\]"):
        fit(ds, PreprocessingConfig(), LabelConfig())


def test_fit_empty_train_set_raises() -> None:
    with pytest.raises(PreprocessingError, match="empty"):
        fit(_StubTrainSet([]), PreprocessingConfig(), LabelConfig())


def test_transform_x_rejects_mismatched_shape() -> None:
    x = _make_x(2, 2, 4, seed=20)
    state = fit(_tensor_dataset(x, _make_y(2, seed=20)), PreprocessingConfig(), LabelConfig())
    with pytest.raises(PreprocessingError, match="P=2"):
        transform_x(state, np.zeros((3, 4, 3), dtype=np.float32))  # P=3 ≠ 2
    with pytest.raises(PreprocessingError, match="channels=3"):
        transform_x(state, np.zeros((2, 2, 4, 2), dtype=np.float32))
    with pytest.raises(PreprocessingError):
        transform_x(state, np.zeros((2, 4), dtype=np.float32))


def test_transform_and_inverse_reject_mismatched_label_shape() -> None:
    x = _make_x(2, 2, 4, seed=21)
    state = fit(_tensor_dataset(x, _make_y(2, seed=21)), PreprocessingConfig(), LabelConfig())
    with pytest.raises(PreprocessingError, match=r"must be \[2\] or \[N,2\]"):
        transform_y(state, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(PreprocessingError, match=r"must be \[2\] or \[N,2\]"):
        inverse_transform_y(state, np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("bad_eps", [0.0, -1.0e-8, float("inf"), float("nan")])
def test_fit_rejects_invalid_eps(bad_eps: float) -> None:
    x = _make_x(2, 2, 4, seed=22)
    with pytest.raises(PreprocessingError, match="std_eps"):
        fit(
            _tensor_dataset(x, _make_y(2, seed=22)),
            PreprocessingConfig(std_eps=bad_eps),
            LabelConfig(),
        )


def test_fit_rejects_unknown_transform() -> None:
    x = _make_x(2, 2, 4, seed=23)
    with pytest.raises(PreprocessingError, match="label.transform"):
        fit(
            _tensor_dataset(x, _make_y(2, seed=23)),
            PreprocessingConfig(),
            LabelConfig(transform=cast(LabelTransform, "log2")),
        )


def test_fit_accepts_numpy_elements() -> None:
    x = _make_x(3, 2, 4, seed=24)
    y = _make_y(3, seed=24)
    state = fit(
        _NumpyElementTrainSet([(x[i], y[i], f"psid_{i}") for i in range(3)]),
        PreprocessingConfig(),
        LabelConfig(),
    )
    mean, std = _reference_input_stats(x)
    np.testing.assert_allclose(state.x_stats.mean, mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(state.x_stats.std, std, rtol=1e-12, atol=1e-14)


def test_state_comparison_is_identity_based() -> None:
    # ndarray fields make default structural equality ambiguous; eq=False → identity comparison.
    x = _make_x(2, 2, 4, seed=25)
    y = _make_y(2, seed=25)
    state_a = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    state_b = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    assert (state_a == state_b) is False
    assert (state_a == state_a) is True


def test_state_mapping_roundtrip_and_bad_schema() -> None:
    """state_to_mapping/from_mapping roundtrip: fields preserved exactly, bad schema rejected."""
    x = _make_x(3, 2, 5, seed=40)
    y = _make_y(3, seed=40)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    mapping = state_to_mapping(state)
    rebuilt = state_from_mapping(mapping)
    assert rebuilt.x_stats.mean.dtype == np.float64
    np.testing.assert_array_equal(rebuilt.x_stats.mean, state.x_stats.mean)
    np.testing.assert_array_equal(rebuilt.x_stats.std, state.x_stats.std)  # effective scale
    assert rebuilt.x_stats.eps == state.x_stats.eps
    assert rebuilt.x_stats.zero_variance_positions == state.x_stats.zero_variance_positions
    assert rebuilt.y_stats.transform == state.y_stats.transform
    np.testing.assert_array_equal(rebuilt.y_stats.y_mean, state.y_stats.y_mean)
    np.testing.assert_array_equal(rebuilt.y_stats.y_std, state.y_stats.y_std)
    assert rebuilt.y_stats.zero_variance_outputs == state.y_stats.zero_variance_outputs
    # transform behavior matches the original state exactly (effective scale semantics)
    np.testing.assert_array_equal(transform_x(rebuilt, x[0]), transform_x(state, x[0]))
    # bad schema: non-mapping / missing sections / shape mismatch / illegal transform / illegal eps
    with pytest.raises(PreprocessingError, match="must be a dict"):
        state_from_mapping(cast(Any, ["nope"]))
    with pytest.raises(PreprocessingError, match="x_stats"):
        state_from_mapping({"y_stats": mapping["y_stats"]})
    with pytest.raises(PreprocessingError, match="shape"):
        broken = {"x_stats": dict(mapping["x_stats"], mean=[[0.0]]), "y_stats": mapping["y_stats"]}
        state_from_mapping(broken)
    with pytest.raises(PreprocessingError, match="label.transform"):
        broken = {
            "x_stats": mapping["x_stats"],
            "y_stats": dict(mapping["y_stats"], transform="log2"),
        }
        state_from_mapping(broken)
    with pytest.raises(PreprocessingError, match="std_eps|eps"):
        broken = {"x_stats": dict(mapping["x_stats"], eps=0.0), "y_stats": mapping["y_stats"]}
        state_from_mapping(broken)
