"""MuMax3 .mx3 模板渲染（vertical slice）。

模板原件位于 simulations/mumax3/（equilibrium.mx3.in 与 simulation.mx3.in），
只读；模板文件的读取、渲染结果的写盘与哈希归 mumax3_pipeline 管。
"""

from __future__ import annotations

from micromagnetic_parameter_inversion.mumax3_config import PulseConfig, SimulationConfig


def _render_model_setup(config: SimulationConfig) -> str:
    """公共模型段 {{MODEL_SETUP}} 的唯一渲染器（防两脚本漂移）。"""
    # TODO: 渲染网格（cell_size 由 size_m/cells 派生）、椭圆几何、
    #  Msat/Aex/Ku1/易轴/demag/开放边界；alpha 不属于公共段；确定性格式化，
    #  结果不得残留占位符、不含绝对路径与输出目录。
    raise NotImplementedError


def render_equilibrium_script(config: SimulationConfig, template_text: str) -> str:
    """把 config 渲染进 equilibrium 模板文本（每 parameter set 执行一次）。"""
    # TODO: {{MODEL_SETUP}} 取 _render_model_setup(config)；另渲染均匀
    #  初态；确定性格式化替换全部占位符，结果不得残留占位符，也不含
    #  绝对路径与输出目录（输出由工作目录决定）。
    raise NotImplementedError


def render_simulation_script(
    config: SimulationConfig, pulse: PulseConfig, template_text: str
) -> str:
    """把 config 与单个 pulse 渲染进 simulation 模板文本（每 pulse 一次）。"""
    # TODO: {{MODEL_SETUP}} 取 _render_model_setup(config)（与 equilibrium
    #  脚本公共段逐字节相同）；外场三分量占位符 {{B_EXT_X_T}}/{{B_EXT_Y_T}}/
    #  {{B_EXT_Z_T}} 直接替换为 pulse.b_ext_amplitude_t * pulse.direction[i]
    #  的运行时 SI 值（renderer 计算完成，单位 T，换算已在 load_config 边界
    #  完成）；另渲染 alpha、脉冲时长与采样契约；同一 config + 同一 pulse
    #  渲染结果逐字节相同。
    raise NotImplementedError
