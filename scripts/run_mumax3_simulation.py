"""run_mumax3_simulation.py —— MuMax3 模拟命令行入口（TODO 骨架）。

薄封装：解析参数 -> 委托 mumax3_pipeline -> 退出码；不复制编排逻辑。
当前仓库唯一可执行入口仍是 scripts/check_environment.py（本文件无
main 调用块，不作为独立入口启用）。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行解析器；首版仅 --config。"""
    # TODO: 注册必填参数 --config（实验 YAML 路径）；不提供其他参数。
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并委托 run_parameter_set，成功返回 0。"""
    # TODO: _build_arg_parser().parse_args(argv) -> run_parameter_set(config)
    #  -> 返回 0；失败异常直接上抛，不设计退出码体系。
    raise NotImplementedError
