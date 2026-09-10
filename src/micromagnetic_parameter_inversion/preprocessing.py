"""输入/标签预处理：仅 train 组拟合的统计状态与变换。

对应 ``train.md`` 第 4 节。核心语义（transform 与 ckpt/preprocessing.yaml
序列化必须按同一语义读写）：

- 输入标准化仅对 train 组沿样本维与时间轴聚合 → 每 pulse 位置、每磁化
  分量的 mean/std，形状 ``[P, 1, 3]``，广播作用于 ``[P, T, 3]`` 或
  ``[N, P, T, 3]``。统计以 float64 分块 Welford 合并稳定累计（总体 std，
  ddof=0），逐样本流式进行，不 stack 整个数据集、不复制全量数据。
- ``std <= eps`` → 除数取 1。**``InputStats.std`` 与 ``LabelState.y_std``
  保存的是 effective scale（实际除数）**：``> eps`` 的位置为原始 std
  （ddof=0），``<= eps`` 的位置恒为 1.0，原始 std 本身不保存；被替换
  位置由 ``zero_variance_positions``/``zero_variance_outputs`` 清单记录。
  序列化（ckpt / preprocessing.yaml）按同一语义原样保存 mean、effective
  scale、eps 与零方差清单，恢复后变换行为与拟合时完全一致。注意零方差
  位置在 val/test 上的变换结果不保证为 0（统计不来自那些组）。
- 标签：``identity`` | ``logalpha``（仅对 alpha 列取 log10，须
  alpha > 0，无静默裁剪；非有限值一律报错），随后逐输出 z-score
  （train 拟合）；val/test 复用同一 state；逆变换还原物理单位
  (alpha, ku_j_per_m3)，遇非有限输入/输出（如模型预测 NaN/Inf）显式
  报错——evaluation 边界依赖该行为，不把坏值当空子集。log10 一律在
  **float64** 上计算（alpha 取自输入的 float64 副本）：float32 输入上
  直接算 log10 会与 fit 的 float64 统计不一致，窄方差场景下 float32
  舍入噪声会被小除数放大。
- 所有 transform 输出 float32，且带**有限性契约**：float64 中间结果转
  float32 的最终 cast 在 ``np.errstate`` 下进行，溢出/非法 →
  ``PreprocessingError``（不静默返回 inf/NaN）。状态 dataclass 含
  ndarray 字段，故 ``eq=False``（ndarray 结构相等会产生真值歧义，状态
  按对象身份比较）。

依赖方向：training_config → training_data →（本模块）→ training /
evaluation；自身不导入模型/训练模块。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from micromagnetic_parameter_inversion.training_config import LabelConfig, PreprocessingConfig
from micromagnetic_parameter_inversion.training_data import TrajectoryDataset

# 磁化分量数 (mx, my, mz)：输入契约恒为 3。
_N_COMPONENTS = 3


class PreprocessingError(ValueError):
    """预处理契约违反（如 logalpha 遇 alpha <= 0；无静默裁剪）。"""


@dataclass(frozen=True, eq=False)
class InputStats:
    """x 的标准化状态（仅 train 组拟合）。

    Attributes:
        mean: float64 ``[P, 1, 3]``：沿样本维与时间轴聚合的均值。
        std: float64 ``[P, 1, 3]``：**effective scale（实际除数）**——
            原始 std（ddof=0）在 ``> eps`` 的位置原值保存；``<= eps`` 的
            位置替换为 1.0，原始值不保存。``transform_x`` 直接以本字段为
            除数；序列化按同一语义原样写。
        eps: ``std <= eps → 除数取 1`` 的阈值（PreprocessingConfig.std_eps）。
        zero_variance_positions: 除数被替换为 1 的 ``(pulse 位置, 分量)``
            清单（原始 std <= eps 的全部位置，按位置升序）。
    """

    mean: np.ndarray
    std: np.ndarray
    eps: float
    zero_variance_positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True, eq=False)
class LabelState:
    """y 的变换与标准化状态（仅 train 组拟合；随 ckpt 保存）。

    Attributes:
        transform: ``"identity" | "logalpha"``（training_config.LabelTransform）。
        y_mean: float64 ``[2]``：变换后空间的逐输出均值。
        y_std: float64 ``[2]``：**effective scale（实际除数）**，语义同
            ``InputStats.std``。
        eps: 除数取 1 保护阈值（防 train 内某输出零方差在 val/test 上除 0）。
        zero_variance_outputs: 除数被替换为 1 的输出列清单（0=alpha、
            1=ku_j_per_m3，升序）。
    """

    transform: str
    y_mean: np.ndarray
    y_std: np.ndarray
    eps: float
    zero_variance_outputs: tuple[int, ...] = ()


@dataclass(frozen=True, eq=False)
class PreprocessingState:
    """ckpt 保存的完整预处理状态；evaluate 一律从 ckpt 恢复，不重新拟合。"""

    x_stats: InputStats
    y_stats: LabelState


def _validate_eps(eps: float) -> float:
    """std_eps 须为正有限值（与 training_config.load_config 校验一致）。"""
    value = float(eps)
    if not math.isfinite(value) or value <= 0.0:
        raise PreprocessingError(f"std_eps 必须为正的有限值 (got {eps!r})")
    return value


def _validate_label_transform(transform: str) -> str:
    if transform not in ("identity", "logalpha"):
        raise PreprocessingError(
            f"未知 label.transform {transform!r}；允许 'identity' | 'logalpha'"
        )
    return transform


def _apply_label_transform(transform: str, y: np.ndarray) -> np.ndarray:
    """标签变换（float64 语义）：identity 透传；logalpha 仅 alpha 列取 log10。

    非有限值或 logalpha 遇 alpha <= 0 → PreprocessingError（无静默裁剪）。
    """
    if not np.all(np.isfinite(y)):
        raise PreprocessingError(f"标签含非有限值 (NaN/Inf)，无法应用 {transform!r} 变换")
    if transform == "identity":
        return y
    if transform == "logalpha":
        # 统一 float64 精度：alpha 取自 float64 副本再取 log10。若在
        # float32 输入上直接计算，log10 结果只有 float32 精度，与 fit 的
        # float64 统计不一致——窄方差（相邻 float32 alpha）场景下舍入
        # 噪声会被小除数放大（归一化本应约 [-1, 1]，实测约 [-4.1, 3.2]）。
        transformed = np.array(y, dtype=np.float64)
        alpha = transformed[..., 0]
        if np.any(alpha <= 0.0):
            raise PreprocessingError(
                f"logalpha 要求 alpha > 0（无静默裁剪）(got 最小值 {float(np.min(alpha))!r})"
            )
        transformed[..., 0] = np.log10(alpha)
        return transformed
    raise PreprocessingError(f"未知 label.transform {transform!r}；允许 'identity' | 'logalpha'")


def _label_to_physical(transform: str, y_transformed: np.ndarray) -> np.ndarray:
    """变换后标签 → 物理单位 (alpha, ku_j_per_m3)（float64）。"""
    if transform == "identity":
        return y_transformed
    if transform == "logalpha":
        physical = np.array(y_transformed, dtype=np.float64)
        physical[..., 0] = 10.0 ** physical[..., 0]
        return physical
    raise PreprocessingError(f"未知 label.transform {transform!r}；允许 'identity' | 'logalpha'")


def _to_float32_or_raise(values: np.ndarray, name: str) -> np.ndarray:
    """最终 float64 → float32 cast；溢出/非法 → PreprocessingError（有限性契约）。

    float64 下有限的值（如 ``1e40``）转 float32 可能溢出为 inf；契约以
    **最终 float32 输出**的有限性为准，不静默返回 inf/NaN。
    ``np.errstate`` 仅压制 cast 时的溢出/非法警告，随后显式报错。
    """
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise PreprocessingError(f"{name}超出 float32 可表示范围（溢出为 NaN/Inf）")
    return result


def _as_input_array(x: np.ndarray, n_pulse: int) -> np.ndarray:
    """校验 x 形状为 ``[P,T,3]`` 或 ``[N,P,T,3]``（P 与统计量一致）。"""
    arr = np.asarray(x)
    expected = f"[P,T,3] 或 [N,P,T,3] (P={n_pulse}, 通道={_N_COMPONENTS})"
    if arr.ndim == 3:
        if arr.shape[0] != n_pulse or arr.shape[-1] != _N_COMPONENTS:
            raise PreprocessingError(f"x 形状须为 {expected} (got shape {tuple(arr.shape)})")
        return arr
    if arr.ndim == 4:
        if arr.shape[1] != n_pulse or arr.shape[-1] != _N_COMPONENTS:
            raise PreprocessingError(f"x 形状须为 {expected} (got shape {tuple(arr.shape)})")
        return arr
    raise PreprocessingError(f"x 形状须为 {expected} (got shape {tuple(arr.shape)})")


def _as_label_array(y: np.ndarray, name: str) -> np.ndarray:
    """校验标签形状为 ``[2]`` 或 ``[N, 2]``。"""
    arr = np.asarray(y)
    if (arr.ndim == 1 and arr.shape[0] == 2) or (arr.ndim == 2 and arr.shape[1] == 2):
        return arr
    raise PreprocessingError(f"{name} 形状须为 [2] 或 [N,2] (got shape {tuple(arr.shape)})")


def _validate_sample(
    index: int,
    x64: np.ndarray,
    y64: np.ndarray,
    ref_shape: tuple[int, int, int] | None,
) -> None:
    """逐样本读入必需校验：x ``[P,T,3]``（P/T 为正且样本间一致）、y ``[2]``。"""
    if x64.ndim != 3 or x64.shape[-1] != _N_COMPONENTS:
        raise PreprocessingError(
            f"train 样本 {index} 的 x 形状须为 [P,T,3] (通道={_N_COMPONENTS}) "
            f"(got shape {tuple(x64.shape)})"
        )
    if x64.shape[0] < 1 or x64.shape[1] < 1:
        raise PreprocessingError(
            f"train 样本 {index} 的 P/T 必须为正 (got shape {tuple(x64.shape)})"
        )
    if ref_shape is not None and x64.shape != ref_shape:
        raise PreprocessingError(
            f"train 样本 {index} 的 x 形状 {tuple(x64.shape)} 与首个样本 {ref_shape} 不一致"
        )
    if y64.shape != (2,):
        raise PreprocessingError(
            f"train 样本 {index} 的 y 形状须为 [2] (got shape {tuple(y64.shape)})"
        )


def fit(
    train_set: TrajectoryDataset,
    config: PreprocessingConfig,
    label: LabelConfig,
) -> PreprocessingState:
    """仅在 train 组上拟合预处理状态（无 eval 参与；泄漏防线）。

    Args:
        train_set: **train 组**的 ``TrajectoryDataset``；元素为 raw 未标准化
            的 ``SampleItem = (x, y, psid)``，x/y 为 float32（Tensor 或
            等价 numpy 数组，统一 ``np.asarray`` 聚合）。val/test 的统计
            不得进入本函数；psid 不参与统计。
        config: 提供 std_eps（``std <= eps → 除数取 1``；须为正有限值，与
            ``training_config.load_config`` 的校验一致）。
        label: 提供 label.transform（identity | logalpha）。

    Returns:
        ``PreprocessingState``：x 统计量 float64 ``[P, 1, 3]`` 与标签统计量
        float64 ``[2]``；``std``/``y_std`` 为 effective scale（见各自
        docstring），零方差位置/输出记入清单。

    Raises:
        PreprocessingError: train 集为空；样本 x 形状非 ``[P,T,3]``、P/T
            非正或样本间不一致；y 形状非 ``[2]``；x/y 含非有限值；
            logalpha 遇 alpha <= 0；eps/transform 配置非法。

    实现说明：逐样本流式累计——时间轴内先两遍法求分块 (mean, M2)，再按
    Chan et al. 并行公式合并进全局 float64 累计量（总体 std，ddof=0，
    数值稳定），不 stack 整个数据集。
    """
    eps = _validate_eps(config.std_eps)
    transform = _validate_label_transform(label.transform)

    count = 0  # 每 [P,3] 位置的观测数 = 已见样本数 × T
    mean: np.ndarray | None = None  # float64 [P,3]
    m2: np.ndarray | None = None  # float64 [P,3]：离差平方和（Welford 合并量）
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
            raise PreprocessingError(f"train 样本 {index} 的 x 含非有限值 (NaN/Inf)")
        # 分块统计（时间轴两遍法，稳定），再并入全局 Welford 累计量。
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
        raise PreprocessingError("train 集为空：预处理统计量至少需要 1 个样本")

    y_all = np.stack(y_rows, axis=0)  # [N,2] float64（仅标签，开销可忽略）
    y_transformed = _apply_label_transform(transform, y_all)

    std_raw = np.sqrt(m2 / count)  # [P,3]：总体 std（ddof=0）
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
    """按 ``[P, 1, 3]`` 统计量广播标准化 x（val/test 复用同一 state）。

    Args:
        state: ``fit`` 产出（或 ckpt 恢复）的预处理状态。
        x: raw 输入 ``[P, T, 3]`` 或 batch ``[N, P, T, 3]``；T 可与拟合时
            不同（统计量沿时间轴聚合，广播与 T 无关）。

    Returns:
        float32 数组（形状与输入一致）：``(x - mean) / effective_scale``；
        零方差位置除数为 1（结果为 ``x - mean``，train 内恒定值 → 0）。

    Raises:
        PreprocessingError: 形状与 state 的 ``[P, 1, 3]`` 统计量不匹配；
            标准化结果（含非有限输入的传播）超出 float32 可表示范围。
    """
    arr = _as_input_array(x, state.x_stats.mean.shape[0])
    out = np.array(arr, dtype=np.float64)  # 防御性拷贝：原地运算不改调用方数据
    out -= state.x_stats.mean
    out /= state.x_stats.std  # effective scale：零方差位置除 1
    return _to_float32_or_raise(out, "x 标准化结果")


def transform_y(state: PreprocessingState, y: np.ndarray) -> np.ndarray:
    """标签变换 + z-score（与 ``fit`` 内部路径一致；val/test 复用同一 state）。

    Args:
        state: ``fit`` 产出（或 ckpt 恢复）的预处理状态。
        y: 物理单位标签 ``[2]`` 或 ``[N, 2]``（列序 alpha, ku_j_per_m3）。

    Returns:
        float32 数组（形状与输入一致）：变换后 ``(y' - y_mean) / y_std``，
        其中 ``y_std`` 为 effective scale。

    Raises:
        PreprocessingError: 形状非法、标签含非有限值、logalpha 遇
            alpha <= 0（无静默裁剪），或标准化结果超出 float32 可表示
            范围。
    """
    _validate_label_transform(state.y_stats.transform)
    arr = _as_label_array(y, "y")
    transformed = _apply_label_transform(state.y_stats.transform, arr)
    out = (transformed - state.y_stats.y_mean) / state.y_stats.y_std
    return _to_float32_or_raise(out, "标签标准化结果")


def inverse_transform_y(state: PreprocessingState, y_normalized: np.ndarray) -> np.ndarray:
    """标准化标签 → 物理单位 (alpha, ku_j_per_m3)（评估输出用）。

    逆路径：逐输出反 z-score（``y_norm * y_std + y_mean``，``y_std`` 为
    effective scale，零方差输出回到 ``y_mean``）；identity 至此结束，
    logalpha 再对 alpha 列取 ``10 ** alpha'``。

    Args:
        state: ``fit`` 产出（或 ckpt 恢复）的预处理状态。
        y_normalized: 标准化空间预测 ``[2]`` 或 ``[N, 2]``。

    Returns:
        float32 数组（形状与输入一致）：物理单位 (alpha, ku_j_per_m3)。

    Raises:
        PreprocessingError: 输入/结果含非有限值（模型预测 NaN/Inf 在此
            边界显式报错，不当作空子集）；float64 结果有限但超出 float32
            可表示范围（如 ``10 ** 40``）；或形状非法。
    """
    _validate_label_transform(state.y_stats.transform)
    arr = _as_label_array(y_normalized, "y_normalized")
    if not np.all(np.isfinite(arr)):
        raise PreprocessingError("逆变换输入含非有限值 (NaN/Inf) 预测；请检查模型输出")
    restored = np.array(arr, dtype=np.float64)
    restored *= state.y_stats.y_std  # effective scale
    restored += state.y_stats.y_mean
    physical = _label_to_physical(state.y_stats.transform, restored)
    if not np.all(np.isfinite(physical)):
        raise PreprocessingError("逆变换结果含非有限值 (NaN/Inf)；物理单位还原溢出或统计异常")
    return _to_float32_or_raise(physical, "逆变换结果")


def state_to_mapping(state: PreprocessingState) -> dict[str, Any]:
    """PreprocessingState → 嵌套纯字典（本模块拥有的唯一映射 schema）。

    mean/effective scale 以 float64 经 Python float 精确保真（``tolist``），
    零方差清单为 int 对列表；容器均为 primitives/list/dict，无 ndarray /
    dataclass 对象——ckpt 与 preprocessing.yaml 共用此形态。
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
    """嵌套纯字典 → PreprocessingState（ckpt / preprocessing.yaml 重建的唯一入口）。

    与 ``state_to_mapping`` 对称；effective scale 语义原样恢复（不重新
    拟合）。P 维与输入契约的交叉校验由调用方（training 边界）负责。

    Raises:
        PreprocessingError: 非映射、缺 x_stats/y_stats、形状非 ``[P,1,3]``
            （P>=1，mean/std 一致）或 ``[2]``、transform 非法、eps 非正有限。
    """
    if not isinstance(mapping, Mapping):
        raise PreprocessingError(f"preprocessing 映射须为 dict (got {type(mapping)!r})")
    try:
        x_raw = mapping["x_stats"]
        y_raw = mapping["y_stats"]
        if not isinstance(x_raw, Mapping) or not isinstance(y_raw, Mapping):
            raise PreprocessingError("preprocessing 映射缺 x_stats/y_stats")
        mean = np.asarray(x_raw["mean"], dtype=np.float64)
        std = np.asarray(x_raw["std"], dtype=np.float64)
        if mean.ndim != 3 or mean.shape[1:] != (1, 3) or std.shape != mean.shape:
            raise PreprocessingError(
                f"x 统计量形状须为 [P,1,3] 且 mean/std 一致 "
                f"(got mean {mean.shape} / std {std.shape})"
            )
        if mean.shape[0] < 1:
            raise PreprocessingError(f"x 统计量的 P 必须为正 (got {mean.shape[0]})")
        y_mean = np.asarray(y_raw["y_mean"], dtype=np.float64)
        y_std = np.asarray(y_raw["y_std"], dtype=np.float64)
        if y_mean.shape != (2,) or y_std.shape != (2,):
            raise PreprocessingError(
                f"y 统计量形状须为 (2,) (got y_mean {y_mean.shape} / y_std {y_std.shape})"
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
        raise PreprocessingError(f"preprocessing 映射损坏: {exc}") from exc
