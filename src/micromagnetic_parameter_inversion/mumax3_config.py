"""MuMax3 模拟配置：frozen dataclass、YAML 加载与派生量（vertical slice）。

一份 YAML = 一个 (alpha, Ku) parameter set 的多脉冲模拟输入；研究数值一律
来自 configs/experiments/*.yaml，本模块不携带任何默认数值。加载后即假定
配置合法，不在流程中重复设计防御与异常体系。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

type Vector3 = tuple[float, float, float]
type Cells3 = tuple[int, int, int]


@dataclass(frozen=True)
class MaterialConfig:
    """单一均匀材料的 SI 参数；alpha 与 Ku 为反演目标。"""

    ms_a_per_m: float  # 饱和磁化强度，A/m
    aex_j_per_m: float  # 交换刚度常数，J/m
    alpha: float  # Gilbert 阻尼系数，无量纲（反演目标）
    ku_j_per_m3: float  # 单轴各向异性常数，J/m^3（反演目标）
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
class PulseConfig:
    """单个短矩形脉冲激励（每项一次独立 run）。"""

    pulse_id: str  # 脉冲唯一标识
    b_ext_amplitude_mT: float  # 幅值，mT（人类配置单位，内部换算为 T）
    direction: Vector3  # 脉冲方向单位向量，无量纲
    duration_s: float  # 脉冲时长，s；关场后 B_ext 恒为 0


@dataclass(frozen=True)
class SimulationConfig:
    """一份配置 = 一个 parameter set 的全部模拟输入（聚合根）。"""

    dataset_name: str
    material: MaterialConfig
    geometry: GeometryConfig
    initial_m: Vector3  # 均匀初态方向，无量纲单位向量
    recording: RecordingConfig
    pulses: tuple[PulseConfig, ...]


def load_config(path: Path) -> SimulationConfig:
    """读取 YAML 并构造 SimulationConfig。"""
    # TODO: 读取 YAML -> 逐必填字段检查，拒绝任何仍为 null 的研究值
    #  （报出字段名）-> 按上述 dataclass 构造并返回；此后流程假定配置合法。
    raise NotImplementedError


def derive_cell_size_m(geometry: GeometryConfig) -> Vector3:
    """派生网格单元尺寸：cell_size[i] = size_m[i] / cells[i]（单一真值）。"""
    # TODO: 逐分量相除并返回 Vector3。
    raise NotImplementedError


def mt_to_t(value_mT: float) -> float:
    """全项目唯一 mT -> T 换算点（b_ext_amplitude_mT -> B_ext，单位 T）。"""
    # TODO: 返回 value_mT * 1e-3。
    raise NotImplementedError


def parameter_set_id(alpha: float, ku_j_per_m3: float) -> str:
    """由 (alpha, Ku) 生成参数组身份；数据集 split 的唯一分组键。"""
    # TODO: 以固定规范化规则（确定性格式化后哈希）仅由 alpha 与 Ku 生成；
    #  绝不包含激励、几何、执行或输出信息。
    raise NotImplementedError
