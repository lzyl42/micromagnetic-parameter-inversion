"""MuMax3 模拟配置：frozen dataclass、YAML 加载与派生量。

一份 YAML = 一个 (alpha, Ku) parameter set 的多脉冲模拟输入；研究数值一律
来自 configs/experiments/*.yaml，本模块不携带任何默认数值。load_config 是
唯一校验与单位换算边界，之后流程假定配置合法，不重复防御。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import yaml

type Vector3 = tuple[float, float, float]
type Cells3 = tuple[int, int, int]

_TOP_LEVEL_KEYS = frozenset(
    {"dataset_name", "material", "geometry", "initial_m", "recording", "numerics", "pulses"}
)
_MATERIAL_KEYS = frozenset({"ms_a_per_m", "aex_j_per_m", "alpha", "ku_j_per_m3", "anisotropy_axis"})
_GEOMETRY_KEYS = frozenset({"size_m", "cells"})
_RECORDING_KEYS = frozenset({"sample_interval_s", "sample_count"})
_NUMERICS_KEYS = frozenset(
    {
        "edge_smooth",
        "solver",
        "max_err",
        "max_dt_s",
        "gamma_ll_rad_per_t_s",
        "relax_torque_threshold_t",
    }
)
_PULSE_KEYS = frozenset({"pulse_id", "b_ext_amplitude_mT", "direction", "duration_s"})

# 单位向量范数容差：容忍 YAML 手写值（如 1/sqrt(3)）的浮点舍入。
_UNIT_VECTOR_TOLERANCE = 1e-6


class ConfigError(ValueError):
    """YAML 配置违反 schema/取值约束；消息含字段路径（如 material.alpha）。"""


@dataclass(frozen=True)
class MaterialConfig:
    """单一均匀材料的 SI 参数；alpha 与 Ku 为反演目标。"""

    ms_a_per_m: float  # 饱和磁化强度，A/m
    aex_j_per_m: float  # 交换刚度常数，J/m
    alpha: float  # Gilbert 阻尼系数，无量纲，>= 0（反演目标）
    ku_j_per_m3: float  # 单轴各向异性常数，J/m^3，只须有限（反演目标）
    anisotropy_axis: Vector3  # 易轴单位向量，无量纲


@dataclass(frozen=True)
class GeometryConfig:
    """包围盒尺寸与网格单元数；cell_size 由 size_m/cells 派生（单一真值）。"""

    size_m: Vector3  # 包围盒尺寸 [x, y, z]，m
    cells: Cells3  # 网格单元数 [nx, ny, nz]


@dataclass(frozen=True)
class RecordingConfig:
    """关场后自由衰减的固定采样。"""

    sample_interval_s: float  # 采样间隔，s
    sample_count: int  # 采样点数（决定轨迹行数）


@dataclass(frozen=True)
class NumericsConfig:
    """数值协议（显式渲染进 equilibrium/simulation 两模板；改变即协议改变）。"""

    edge_smooth: int  # EdgeSmooth，非负整数（0=硬阶梯边界）；渲染于 SetGeom 之前
    solver: int  # SetSolver(...) 的 solver ID，正整数
    max_err: float  # MaxErr，正数，无量纲
    max_dt_s: float  # MaxDt，正数，s
    gamma_ll_rad_per_t_s: float  # GammaLL，正数，rad/(T*s)
    relax_torque_threshold_t: float  # RelaxTorqueThreshold，T；只须有限（-1=官方默认）


@dataclass(frozen=True)
class PulseConfig:
    """单个短矩形脉冲激励（每项一次独立 run）。"""

    pulse_id: str  # 脉冲唯一标识，安全单路径段
    b_ext_amplitude_t: float  # 脉冲幅值，运行时 SI 单位 T（load_config 已换算，>= 0）
    direction: Vector3  # 脉冲方向单位向量，无量纲
    duration_s: float  # 脉冲时长，s；关场后 B_ext 恒为 0


@dataclass(frozen=True)
class SimulationConfig:
    """一份配置 = 一个 parameter set 的全部模拟输入（聚合根）。"""

    dataset_name: str  # 安全单路径段；代表固定材料/几何/模拟协议
    material: MaterialConfig
    geometry: GeometryConfig
    initial_m: Vector3  # 均匀初态方向，无量纲单位向量
    recording: RecordingConfig
    numerics: NumericsConfig
    pulses: tuple[PulseConfig, ...]


def _fail(field: str, problem: str) -> NoReturn:
    """抛出带字段路径的 ConfigError，便于定位 YAML 中的具体字段。"""
    raise ConfigError(f"{field}: {problem}")


def _check_mapping_keys(mapping: dict[Any, Any], field: str, known: frozenset[str]) -> None:
    """严格 schema：拒绝未知字段与缺失的必填字段（所有层级适用）。"""
    unknown = sorted((key for key in mapping if key not in known), key=repr)
    if unknown:
        _fail(field, f"未知字段 {unknown!r}；允许的字段 {sorted(known)}")
    missing = sorted(known.difference(mapping))
    if missing:
        _fail(field, f"缺失必填字段 {missing!r}")


def _require_mapping(value: object, field: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        _fail(field, f"必须为映射 (got {value!r})")
    return value


def _require_finite(value: object, field: str) -> float:
    """数值分量：接受 int/float，拒绝 bool 冒充数值与非有限值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, f"必须为数值 (got {value!r})")
    number = float(value)
    if not math.isfinite(number):
        _fail(field, f"必须为有限数值 (got {value!r})")
    return number


def _require_positive_number(value: object, field: str) -> float:
    """按语义应为正的有限数值（0、负数、nan/inf、bool 均拒绝）。"""
    number = _require_finite(value, field)
    if number <= 0:
        _fail(field, f"必须为正数 (got {value!r})")
    return number


def _require_positive_int(value: object, field: str) -> int:
    """正整数：拒绝 bool、浮点（含 3.0 这类整值浮点）与非正值。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(field, f"必须为正整数 (got {value!r})")
    return value


def _require_non_negative_int(value: object, field: str) -> int:
    """非负整数（允许 0）：拒绝 bool、浮点与非负性违反。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(field, f"必须为非负整数 (got {value!r})")
    return value


def _require_non_negative_number(value: object, field: str) -> float:
    """按语义应非负的有限数值（负数、nan/inf、bool 均拒绝）。"""
    number = _require_finite(value, field)
    if number < 0:
        _fail(field, f"必须为非负数 (got {value!r})")
    return number


def _require_length3(value: object, field: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        _fail(field, f"必须为 3 分量序列 (got {value!r})")
    if len(value) != 3:
        _fail(field, f"必须恰为 3 个分量 (got {len(value)} 个)")
    return value


def _require_vector3(
    value: object,
    field: str,
    component: Callable[[object, str], float] = _require_finite,
) -> Vector3:
    """恰 3 个有限分量的向量；component 决定分量的额外约束。"""
    items = _require_length3(value, field)
    return (
        component(items[0], f"{field}[0]"),
        component(items[1], f"{field}[1]"),
        component(items[2], f"{field}[2]"),
    )


def _require_cells3(value: object, field: str) -> Cells3:
    items = _require_length3(value, field)
    return (
        _require_positive_int(items[0], f"{field}[0]"),
        _require_positive_int(items[1], f"{field}[1]"),
        _require_positive_int(items[2], f"{field}[2]"),
    )


def _require_unit_vector(value: object, field: str) -> Vector3:
    """恰 3 个有限分量且范数为 1 的向量（隐含非零向量）。"""
    vector = _require_vector3(value, field)
    norm = math.sqrt(sum(x * x for x in vector))
    if not math.isclose(norm, 1.0, rel_tol=_UNIT_VECTOR_TOLERANCE, abs_tol=_UNIT_VECTOR_TOLERANCE):
        _fail(field, f"必须为单位向量 (got {vector!r}, |v| = {norm!r})")
    return vector


def _require_safe_path_segment(value: object, field: str) -> str:
    """安全单路径段：非空、非 "."/".."、不含 "/"、"\\" 或 NUL（防路径逃逸）。"""
    if not isinstance(value, str) or not value:
        _fail(field, f"必须为非空字符串 (got {value!r})")
    if value in {".", ".."}:
        _fail(field, f"不接受 {value!r} 作为路径段")
    if "/" in value or "\\" in value:
        _fail(field, f"不允许包含 '/' 或 '\\' (got {value!r})")
    if "\x00" in value:
        _fail(field, f"不允许包含 NUL 字符 (got {value!r})")
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    """拒绝重复 YAML 键的 SafeLoader：研究值（如 alpha）不得被静默覆盖。"""

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: list[object] = []
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if any(type(key) is type(other) and key == other for other in seen):
                raise yaml.constructor.ConstructorError(
                    None, None, f"重复的 YAML 键: {key!r}", key_node.start_mark
                )
            seen.append(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path: Path) -> SimulationConfig:
    """读取 YAML、执行唯一入口校验并构造 SimulationConfig。

    校验边界（此后流程假定配置合法，不重复防御）：所有层级严格 schema，
    拒绝缺失/未知字段、null、bool 冒充数值与非有限值；正数/非负数/正
    整数/非负整数按字段语义分别约束（Ku 与 relax_torque_threshold_t 只须
    有限，允许 0/负与 -1）；三维向量须恰 3 个有限分量，
    anisotropy_axis/initial_m/direction 须为单位向量（范数容差 1e-6）；
    pulses 非空且 pulse_id 唯一；dataset_name/pulse_id 为安全单路径段。

    单位边界：b_ext_amplitude_mT 在此乘 1e-3 存为运行时 T，mT 不进入
    运行时模型。
    """
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML 解析失败: {exc}") from exc

    root = _require_mapping(raw, str(path))
    _check_mapping_keys(root, str(path), _TOP_LEVEL_KEYS)

    dataset_name = _require_safe_path_segment(root["dataset_name"], "dataset_name")

    material_raw = _require_mapping(root["material"], "material")
    _check_mapping_keys(material_raw, "material", _MATERIAL_KEYS)
    material = MaterialConfig(
        ms_a_per_m=_require_positive_number(material_raw["ms_a_per_m"], "material.ms_a_per_m"),
        aex_j_per_m=_require_positive_number(material_raw["aex_j_per_m"], "material.aex_j_per_m"),
        alpha=_require_non_negative_number(material_raw["alpha"], "material.alpha"),
        ku_j_per_m3=_require_finite(material_raw["ku_j_per_m3"], "material.ku_j_per_m3"),
        anisotropy_axis=_require_unit_vector(
            material_raw["anisotropy_axis"], "material.anisotropy_axis"
        ),
    )

    geometry_raw = _require_mapping(root["geometry"], "geometry")
    _check_mapping_keys(geometry_raw, "geometry", _GEOMETRY_KEYS)
    geometry = GeometryConfig(
        size_m=_require_vector3(
            geometry_raw["size_m"], "geometry.size_m", component=_require_positive_number
        ),
        cells=_require_cells3(geometry_raw["cells"], "geometry.cells"),
    )

    initial_m = _require_unit_vector(root["initial_m"], "initial_m")

    recording_raw = _require_mapping(root["recording"], "recording")
    _check_mapping_keys(recording_raw, "recording", _RECORDING_KEYS)
    recording = RecordingConfig(
        sample_interval_s=_require_positive_number(
            recording_raw["sample_interval_s"], "recording.sample_interval_s"
        ),
        sample_count=_require_positive_int(recording_raw["sample_count"], "recording.sample_count"),
    )

    numerics_raw = _require_mapping(root["numerics"], "numerics")
    _check_mapping_keys(numerics_raw, "numerics", _NUMERICS_KEYS)
    numerics = NumericsConfig(
        edge_smooth=_require_non_negative_int(numerics_raw["edge_smooth"], "numerics.edge_smooth"),
        solver=_require_positive_int(numerics_raw["solver"], "numerics.solver"),
        max_err=_require_positive_number(numerics_raw["max_err"], "numerics.max_err"),
        max_dt_s=_require_positive_number(numerics_raw["max_dt_s"], "numerics.max_dt_s"),
        gamma_ll_rad_per_t_s=_require_positive_number(
            numerics_raw["gamma_ll_rad_per_t_s"], "numerics.gamma_ll_rad_per_t_s"
        ),
        relax_torque_threshold_t=_require_finite(
            numerics_raw["relax_torque_threshold_t"], "numerics.relax_torque_threshold_t"
        ),
    )

    pulses_raw = root["pulses"]
    if not isinstance(pulses_raw, list) or not pulses_raw:
        _fail("pulses", "必须为非空的 pulse 列表")

    pulses: list[PulseConfig] = []
    seen_pulse_ids: set[str] = set()
    for index, item in enumerate(pulses_raw):
        field = f"pulses[{index}]"
        pulse_raw = _require_mapping(item, field)
        _check_mapping_keys(pulse_raw, field, _PULSE_KEYS)
        pulse_id = _require_safe_path_segment(pulse_raw["pulse_id"], f"{field}.pulse_id")
        if pulse_id in seen_pulse_ids:
            _fail(f"{field}.pulse_id", f"重复的 pulse_id {pulse_id!r}")
        seen_pulse_ids.add(pulse_id)
        amplitude_millitesla = _require_non_negative_number(
            pulse_raw["b_ext_amplitude_mT"], f"{field}.b_ext_amplitude_mT"
        )
        direction = _require_unit_vector(pulse_raw["direction"], f"{field}.direction")
        duration_s = _require_positive_number(pulse_raw["duration_s"], f"{field}.duration_s")
        pulses.append(
            PulseConfig(
                pulse_id=pulse_id,
                # 单位边界：mT 只存在于 YAML，在此乘 1e-3 存为运行时 T。
                b_ext_amplitude_t=amplitude_millitesla * 1e-3,
                direction=direction,
                duration_s=duration_s,
            )
        )

    return SimulationConfig(
        dataset_name=dataset_name,
        material=material,
        geometry=geometry,
        initial_m=initial_m,
        recording=recording,
        numerics=numerics,
        pulses=tuple(pulses),
    )


def derive_cell_size_m(geometry: GeometryConfig) -> Vector3:
    """派生网格单元尺寸：cell_size[i] = size_m[i] / cells[i]（单一真值）。"""
    return (
        geometry.size_m[0] / geometry.cells[0],
        geometry.size_m[1] / geometry.cells[1],
        geometry.size_m[2] / geometry.cells[2],
    )


def parameter_set_id(alpha: float, ku_j_per_m3: float) -> str:
    """由 (alpha, Ku) 生成参数组身份；数据集 split 的唯一分组键。

    ``.17g`` 规范文本的 SHA-256 前 16 位小写十六进制；仅依赖 alpha 与 Ku。
    """
    _require_finite(alpha, "alpha")
    _require_finite(ku_j_per_m3, "ku_j_per_m3")
    canonical = f"alpha={float(alpha):.17g}\nku_j_per_m3={float(ku_j_per_m3):.17g}\n".encode()
    return hashlib.sha256(canonical).hexdigest()[:16]
