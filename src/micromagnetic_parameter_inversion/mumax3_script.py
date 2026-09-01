"""MuMax3 .mx3 模板渲染（vertical slice）。

模板原件位于 simulations/mumax3/（equilibrium.mx3.in 与 simulation.mx3.in），
只读；模板文件的读取、渲染结果的写盘与哈希归 mumax3_pipeline 管。
"""

from __future__ import annotations

from micromagnetic_parameter_inversion.mumax3_config import PulseConfig, SimulationConfig


def render_equilibrium_script(config: SimulationConfig, template_text: str) -> str:
    """把 config 渲染进 equilibrium 模板文本（每 parameter set 执行一次）。"""
    # TODO: 以确定性格式化替换模板占位符（网格/材料/初态）并返回渲染文本；
    #  结果不得残留占位符，也不含绝对路径与输出目录（输出由工作目录决定）。
    raise NotImplementedError


def render_simulation_script(
    config: SimulationConfig, pulse: PulseConfig, template_text: str
) -> str:
    """把 config 与单个 pulse 渲染进 simulation 模板文本（每 pulse 一次）。"""
    # TODO: 占位符含外场三分量（mt_to_t(b_ext_amplitude_mT) 与 direction
    #  逐分量相乘，单位 T）、脉冲时长、采样间隔与采样点数；
    #  同一 config + 同一 pulse 的渲染结果逐字节相同。
    raise NotImplementedError
