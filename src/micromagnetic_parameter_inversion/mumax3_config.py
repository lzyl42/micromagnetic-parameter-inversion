"""MuMax3 模拟配置：frozen dataclass、YAML 加载与派生量（vertical slice）。

一份 YAML = 一个 (alpha, Ku) parameter set 的多脉冲模拟输入；研究数值一律
来自 configs/experiments/*.yaml，本模块不携带任何默认数值。load_config 是
唯一校验与单位换算边界，之后流程假定配置合法，不重复设计防御体系。
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

    pulse_id: str  # 脉冲唯一标识，安全单路径段
    b_ext_amplitude_t: float  # 脉冲幅值，运行时 SI 单位 T（load_config 已换算）
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
    pulses: tuple[PulseConfig, ...]


def load_config(path: Path) -> SimulationConfig:
    """读取 YAML、执行唯一入口校验并构造 SimulationConfig。"""
    # TODO: 读取 YAML 并在此完成唯一校验边界（此后流程假定配置合法）：
    #  - 拒绝所有层级的未知字段；拒绝任何仍为 null 的必填研究值；
    #  - ms_a_per_m/aex_j_per_m/alpha/ku_j_per_m3/sample_interval_s/
    #    duration_s/b_ext_amplitude_mT 须为有限正数；cells/sample_count
    #    须为正整数；
    #  - size_m 恰为 3 个有限正数；cells 恰为 3 个正整数；
    #  - anisotropy_axis/initial_m/pulse direction 均恰为 3 个有限分量
    #    且为单位向量；
    #  - pulses 非空且 pulse_id 唯一；
    #  - dataset_name/pulse_id 须为安全单路径段：非空、非 "."/".."、
    #    不含 "/" 或 "\\"（防路径逃逸）；
    #  - 单位边界：mT 只存在于原始 YAML（b_ext_amplitude_mT），在此直接
    #    乘 1e-3 后构造 PulseConfig.b_ext_amplitude_t；不设转换 helper，
    #    运行时模型与计算只使用 T。
    raise NotImplementedError


def derive_cell_size_m(geometry: GeometryConfig) -> Vector3:
    """派生网格单元尺寸：cell_size[i] = size_m[i] / cells[i]（单一真值）。"""
    # TODO: 逐分量相除并返回 Vector3。
    raise NotImplementedError


def parameter_set_id(alpha: float, ku_j_per_m3: float) -> str:
    r"""由 (alpha, Ku) 生成参数组身份；数据集 split 的唯一分组键。

    固定算法：canonical UTF-8 文本恰为
    `alpha={float(alpha).hex()}\nku_j_per_m3={float(ku_j_per_m3).hex()}\n`，
    返回其完整 SHA-256 十六进制摘要；仅依赖 alpha 与 Ku。
    """
    # TODO: 按上述固定算法实现（float.hex 规范化 + SHA-256 hexdigest）。
    raise NotImplementedError
