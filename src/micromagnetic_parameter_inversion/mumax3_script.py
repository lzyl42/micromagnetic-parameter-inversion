"""MuMax3 .mx3 模板渲染。

模板原件位于 simulations/mumax3/（equilibrium.mx3.in 与 simulation.mx3.in），
只读；模板文件的读取、渲染结果的写盘与哈希归 mumax3_pipeline 管。
"""

from __future__ import annotations

import re

from micromagnetic_parameter_inversion.mumax3_config import (
    PulseConfig,
    SimulationConfig,
    Vector3,
    derive_cell_size_m,
)

# 既定占位符形态：双花括号包裹的大写标识符（如 {{MODEL_SETUP}}）。
_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z0-9_]+\}\}")


class TemplateRenderError(ValueError):
    """模板占位符契约被违反：缺失、重复出现或渲染后仍残留占位符。"""


def _fmt_number(value: float) -> str:
    """确定性数值文本：`.17g` 十进制/科学记法，MuMax3（Go 语法）可直接解析。

    同一浮点值永远得到同一文本（逐字节可复现）；-0 与 0 统一输出为 0。
    """
    number = float(value)
    if number == 0.0:
        return "0"
    return format(number, ".17g")


def _fmt_vector3(components: Vector3) -> str:
    """渲染 3 分量为 `x, y, z` 实参列表文本（供 uniform/vector 括号内使用）。"""
    return ", ".join(_fmt_number(component) for component in components)


def _replace_once(template_text: str, placeholder: str, replacement: str) -> str:
    """占位符必须恰出现一次并整体替换；违反即抛 TemplateRenderError。"""
    parts = template_text.split(placeholder)
    occurrences = len(parts) - 1
    if occurrences != 1:
        raise TemplateRenderError(f"占位符 {placeholder} 出现 {occurrences} 次，必须恰为 1 次")
    return replacement.join((parts[0], parts[1]))


def _reject_residual_placeholders(rendered: str) -> None:
    """渲染结果中残留任何 {{...}} 视为模板/替换契约被破坏。"""
    residual = _PLACEHOLDER_PATTERN.findall(rendered)
    if residual:
        raise TemplateRenderError(f"渲染后仍残留占位符 {residual!r}")


def _render_model_setup(config: SimulationConfig) -> str:
    """公共模型段 {{MODEL_SETUP}} 的唯一渲染器（防两脚本漂移）。

    固定顺序：数值协议（EdgeSmooth 必须先于 SetGeom 设置以影响几何体素化；
    SetSolver/MaxErr/MaxDt/GammaLL 公共积分控制，equilibrium/simulation 显式
    一致）、网格、cell size（size_m/cells 派生，单一真值）、PBC（开放
    边界）、椭球几何（SetGeom 恒取三轴全直径，均来自 size_m，不对 nz 做
    条件分支：nz=1 时单层体素离散自然表现为恒厚椭圆截面薄片，nz>1 时逐层
    解析椭球 z 表面）、demag、材料参数（Msat/Aex/Ku1/易轴）。alpha 与
    RelaxTorqueThreshold 是 per-run/per-script 参数，不属于公共段；输出
    不含路径与占位符。
    """
    material = config.material
    geometry = config.geometry
    numerics = config.numerics
    size_x, size_y, size_z = geometry.size_m
    geom_line = f"SetGeom(Ellipsoid({_fmt_vector3((size_x, size_y, size_z))}))"
    return "\n".join(
        (
            "// 数值协议（YAML numerics 块；equilibrium/simulation 显式一致）",
            "// EdgeSmooth 影响几何体素化，必须先于 SetGeom 设置（0=硬阶梯边界）",
            f"EdgeSmooth = {numerics.edge_smooth}",
            f"SetSolver({numerics.solver})",
            f"MaxErr = {_fmt_number(numerics.max_err)}",
            f"MaxDt = {_fmt_number(numerics.max_dt_s)}",
            f"GammaLL = {_fmt_number(numerics.gamma_ll_rad_per_t_s)}",
            "// 网格与单元尺寸：cell_size = size_m / cells（唯一派生真值），单位 m",
            f"SetGridSize({geometry.cells[0]}, {geometry.cells[1]}, {geometry.cells[2]})",
            f"SetCellSize({_fmt_vector3(derive_cell_size_m(geometry))})",
            "// 开放边界：不启用周期性镜像",
            "SetPBC(0, 0, 0)",
            "// 扁椭球薄纳米磁体：三轴全直径 dx, dy, dz 均来自 size_m（= 包围盒尺寸）",
            geom_line,
            "// 退磁场开启",
            "EnableDemag = true",
            "// 单一均匀材料（SI 单位：Msat A/m；Aex J/m；Ku1 J/m^3）与易轴单位向量",
            f"Msat = {_fmt_number(material.ms_a_per_m)}",
            f"Aex = {_fmt_number(material.aex_j_per_m)}",
            f"Ku1 = {_fmt_number(material.ku_j_per_m3)}",
            f"anisU = vector({_fmt_vector3(material.anisotropy_axis)})",
        )
    )


def render_equilibrium_script(config: SimulationConfig, template_text: str) -> str:
    """把 config 渲染进 equilibrium 模板文本（每 parameter set 执行一次）。"""
    rendered = _replace_once(template_text, "{{MODEL_SETUP}}", _render_model_setup(config))
    rendered = _replace_once(rendered, "{{INIT_M}}", _fmt_vector3(config.initial_m))
    rendered = _replace_once(
        rendered,
        "{{RELAX_TORQUE_THRESHOLD_T}}",
        _fmt_number(config.numerics.relax_torque_threshold_t),
    )
    _reject_residual_placeholders(rendered)
    return rendered


def render_simulation_script(
    config: SimulationConfig, pulse: PulseConfig, template_text: str
) -> str:
    """把 config 与单个 pulse 渲染进 simulation 模板文本（每 pulse 一次）。

    外场三分量在 renderer 内算完（amplitude_t * direction[i]，单位 T，换算
    已在 load_config 边界完成）；同一 config + 同一 pulse 渲染结果逐字节
    相同。
    """
    amplitude_t = pulse.b_ext_amplitude_t
    direction_x, direction_y, direction_z = pulse.direction
    rendered = template_text
    for placeholder, replacement in (
        ("{{MODEL_SETUP}}", _render_model_setup(config)),
        ("{{ALPHA}}", _fmt_number(config.material.alpha)),
        ("{{B_EXT_X_T}}", _fmt_number(amplitude_t * direction_x)),
        ("{{B_EXT_Y_T}}", _fmt_number(amplitude_t * direction_y)),
        ("{{B_EXT_Z_T}}", _fmt_number(amplitude_t * direction_z)),
        ("{{PULSE_DURATION_S}}", _fmt_number(pulse.duration_s)),
        ("{{SAMPLE_COUNT}}", str(config.recording.sample_count)),
        ("{{SAMPLE_INTERVAL_S}}", _fmt_number(config.recording.sample_interval_s)),
    ):
        rendered = _replace_once(rendered, placeholder, replacement)
    _reject_residual_placeholders(rendered)
    return rendered
