"""preprocessing 单元测试：CPU 合成数据 + stub Dataset（不触真实 raw）。

覆盖：仅 train 拟合（val 统计不影响输出）、
``[P,1,3]`` 广播形状、x/y 零方差位置除数取 1、标签变换-逆变换往返
（identity/logalpha）、logalpha 遇 alpha <= 0 报错（无静默裁剪）、
非有限值报错、batch 广播与 float32 输出、logalpha float64 统一精度
（相邻 float32 alpha 窄方差回归）、float32 溢出有限性契约（transform_x/
transform_y/inverse_transform_y 均不静默返回 inf）、state ``eq=False``
语义。

stub Dataset 仅实现 ``__len__``/``__getitem__`` 并返回 ``SampleItem``
（TrajectoryDataset 骨架的业务构造器不可用，也不需要）。
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
    """最小 stub：元素为 ``SampleItem``（torch Tensor）。"""

    def __init__(self, samples: list[SampleItem]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SampleItem:
        return self._samples[index]


class _NumpyElementTrainSet(TrajectoryDataset):
    """元素为等价 numpy 数组的 stub（fit 须统一 ``np.asarray`` 聚合）。"""

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
    """物理单位标签 [N,2]：(alpha ~ 0.01-0.1, ku_j_per_m3 ~ 1e4-1e5)。"""
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
    """沿样本维与时间轴聚合的参考 mean/std（float64，ddof=0），形状 [P,1,3]。"""
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
    x_val = _make_x(3, 2, 5, seed=3) + 10.0  # val 统计明显不同
    state = fit(_tensor_dataset(x_train, _make_y(4, seed=2)), PreprocessingConfig(), LabelConfig())
    mean, std = _reference_input_stats(x_train)
    np.testing.assert_allclose(state.x_stats.mean, mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(state.x_stats.std, std, rtol=1e-12, atol=1e-14)
    # val 复用 train 统计量：+10 的偏移不会被 train mean 吸收为 0。
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
    # 训练数据标准化后总体均值≈0、总体 std≈1。
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
    x[:, 1, :, 2] = 0.5  # pulse 位置 1、分量 2 恒定 → std=0 <= eps → 除数 1
    state = fit(_tensor_dataset(x, _make_y(4, seed=6)), PreprocessingConfig(), LabelConfig())
    assert state.x_stats.zero_variance_positions == ((1, 2),)
    assert state.x_stats.std[1, 0, 2] == 1.0  # effective scale
    # train 内恒定值变换后为 0；(v - mean)/1 在 train 外仍有限、不放大爆炸。
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
    assert state.x_stats.std[0, 0, 0] == 1.0  # <= eps → 除数 1
    assert 5.0e-3 < state.x_stats.std[1, 0, 1] < 2.0e-2  # > eps → 保留真实 std
    assert state.x_stats.eps == 1.0e-3


def test_label_zero_variance_divisor_is_one() -> None:
    y = _make_y(4, seed=9)
    y[:, 0] = 0.05  # alpha 列恒定 → std=0 <= eps → 除数 1
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
    # 单样本 [2]。
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
    """相邻 float32 alpha 的窄方差回归：归一化须与全 float64 参考一致（约 [-1,1]）。

    曾有缺陷：transform_y 在 float32 输入上直接计算 log10，与 fit 的
    float64 统计不一致——0.05f 与其上邻 float32 的两点场景（std ≈ 1.6e-8
    > eps，保留真实小除数）中 float32 舍入噪声被放大，归一化结果约
    [-4.127, 3.241] 而非 [-1, 1]。
    """
    alpha_lo = np.float32(0.05)
    alpha_hi = np.nextafter(alpha_lo, np.float32(1.0))
    assert alpha_hi > alpha_lo  # 确为上邻（相差 1 ULP）
    y = np.array(
        [[float(alpha_lo), 2.0e4], [float(alpha_hi), 2.5e4]],
        dtype=np.float32,
    )
    x = _make_x(2, 2, 4, seed=30)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig(transform="logalpha"))
    # 窄方差前提：alpha 列 std 为真实小除数（未被 eps 替换为 1）。
    assert 1.0e-8 < float(state.y_stats.y_std[0]) < 1.0e-6
    y_norm = transform_y(state, y)
    # 全 float64 参考计算（统一精度后的期望）。
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
    # 两点总体 z-score 的 alpha 列应恰约 ±1（而非舍入噪声放大的 ±4）。
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
    with pytest.raises(PreprocessingError, match="样本 2"):
        fit(_tensor_dataset(x, _make_y(3, seed=14)), PreprocessingConfig(), LabelConfig())


def test_fit_rejects_nonfinite_labels() -> None:
    x = _make_x(3, 2, 4, seed=15)
    y = _make_y(3, seed=15)
    y[0, 1] = np.nan
    with pytest.raises(PreprocessingError, match="非有限"):
        fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())


def test_inverse_transform_y_rejects_nonfinite_predictions() -> None:
    x = _make_x(2, 2, 4, seed=16)
    state = fit(_tensor_dataset(x, _make_y(2, seed=16)), PreprocessingConfig(), LabelConfig())
    with pytest.raises(PreprocessingError, match="非有限"):
        inverse_transform_y(state, np.array([np.nan, 0.0], dtype=np.float32))
    with pytest.raises(PreprocessingError, match="非有限"):
        inverse_transform_y(state, np.array([[0.0, np.inf]], dtype=np.float32))


def _hand_built_logalpha_state() -> PreprocessingState:
    """mean=0 / effective scale=1 的最小 logalpha 状态（直接构造，绕过 fit）。"""
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
    """float64 有限但 float32 溢出的逆变换须报错，不得静默返回 inf。

    回归：曾在 float64 结果上检查有限性后再 cast float32——``10 ** 40``
    于 float64 有限、于 float32 溢出为 inf。
    """
    state = _hand_built_logalpha_state()
    with pytest.raises(PreprocessingError, match="float32"):
        inverse_transform_y(state, np.array([40.0, 0.0]))  # 10^40 → float32 inf
    with pytest.raises(PreprocessingError, match="float32"):
        inverse_transform_y(state, np.array([0.0, 1.0e39]))  # Ku 线性列同样溢出
    # float32 可表示的边界值仍正常返回且有限。
    out = inverse_transform_y(state, np.array([38.0, 0.0]))
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_transform_x_rejects_float32_overflow() -> None:
    """x 标准化结果超出 float32（极小真实 std × train 外大偏移）须报错。"""
    rng = np.random.default_rng(33)
    x = _make_x(3, 2, 64, seed=33)
    x[:, 0, :, 0] = rng.normal(0.0, 1.0e-6, size=(3, 64))  # std ~1e-6 > eps → 保留小除数
    state = fit(_tensor_dataset(x, _make_y(3, seed=33)), PreprocessingConfig(), LabelConfig())
    assert 1.0e-8 < float(state.x_stats.std[0, 0, 0]) < 1.0e-5  # 真实小除数
    probe = np.zeros((1, 2, 4, 3), dtype=np.float32)
    probe[:, 0, :, 0] = np.float32(1.0e33)
    with pytest.raises(PreprocessingError, match="float32"):
        transform_x(state, probe)  # ~1e33 / 1e-6 ≈ 1e39 → float32 inf


def test_transform_y_rejects_float32_overflow() -> None:
    """标签标准化结果超出 float32（小真实 std × val 大偏移）须报错。"""
    x = _make_x(3, 2, 4, seed=34)
    y = _make_y(3, seed=34)
    y[:, 0] = np.array([0.05, 0.05 + 1.0e-6, 0.05 - 1.0e-6], dtype=np.float32)
    state = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    assert 1.0e-8 < float(state.y_stats.y_std[0]) < 1.0e-5  # 真实小除数（非替换为 1）
    with pytest.raises(PreprocessingError, match="float32"):
        transform_y(state, np.array([[1.0e33, 2.0e4]], dtype=np.float32))


def test_fit_rejects_inconsistent_sample_shapes() -> None:
    x = _make_x(2, 2, 4, seed=17)
    y = _make_y(2, seed=17)
    other = _make_x(1, 3, 4, seed=18)[0]  # P=3 ≠ 首样本 P=2
    ds = _StubTrainSet(
        [
            (torch.from_numpy(x[0]), torch.from_numpy(y[0]), "a"),
            (torch.from_numpy(other), torch.from_numpy(y[1]), "b"),
        ]
    )
    with pytest.raises(PreprocessingError, match="不一致"):
        fit(ds, PreprocessingConfig(), LabelConfig())


def test_fit_rejects_bad_label_shape() -> None:
    x = _make_x(1, 2, 4, seed=19)
    ds = _StubTrainSet([(torch.from_numpy(x[0]), torch.zeros(3), "a")])
    with pytest.raises(PreprocessingError, match=r"\[2\]"):
        fit(ds, PreprocessingConfig(), LabelConfig())


def test_fit_empty_train_set_raises() -> None:
    with pytest.raises(PreprocessingError, match="空"):
        fit(_StubTrainSet([]), PreprocessingConfig(), LabelConfig())


def test_transform_x_rejects_mismatched_shape() -> None:
    x = _make_x(2, 2, 4, seed=20)
    state = fit(_tensor_dataset(x, _make_y(2, seed=20)), PreprocessingConfig(), LabelConfig())
    with pytest.raises(PreprocessingError, match="P=2"):
        transform_x(state, np.zeros((3, 4, 3), dtype=np.float32))  # P=3 ≠ 2
    with pytest.raises(PreprocessingError, match="通道=3"):
        transform_x(state, np.zeros((2, 2, 4, 2), dtype=np.float32))
    with pytest.raises(PreprocessingError):
        transform_x(state, np.zeros((2, 4), dtype=np.float32))


def test_transform_and_inverse_reject_mismatched_label_shape() -> None:
    x = _make_x(2, 2, 4, seed=21)
    state = fit(_tensor_dataset(x, _make_y(2, seed=21)), PreprocessingConfig(), LabelConfig())
    with pytest.raises(PreprocessingError, match=r"须为 \[2\] 或 \[N,2\]"):
        transform_y(state, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(PreprocessingError, match=r"须为 \[2\] 或 \[N,2\]"):
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
    # ndarray 字段令默认结构相等产生真值歧义；eq=False → 按对象身份比较。
    x = _make_x(2, 2, 4, seed=25)
    y = _make_y(2, seed=25)
    state_a = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    state_b = fit(_tensor_dataset(x, y), PreprocessingConfig(), LabelConfig())
    assert (state_a == state_b) is False
    assert (state_a == state_a) is True


def test_state_mapping_roundtrip_and_bad_schema() -> None:
    """state_to_mapping/from_mapping 往返：字段保真、坏 schema 拒绝。"""
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
    # 变换行为与原 state 完全一致（effective scale 语义恢复）
    np.testing.assert_array_equal(transform_x(rebuilt, x[0]), transform_x(state, x[0]))
    # 坏 schema：非映射 / 缺节 / 形状不符 / 非法 transform / 非法 eps
    with pytest.raises(PreprocessingError, match="须为 dict"):
        state_from_mapping(cast(Any, ["nope"]))
    with pytest.raises(PreprocessingError, match="x_stats"):
        state_from_mapping({"y_stats": mapping["y_stats"]})
    with pytest.raises(PreprocessingError, match="形状"):
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
