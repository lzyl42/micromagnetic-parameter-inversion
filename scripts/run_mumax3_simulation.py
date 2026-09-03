#!/usr/bin/env python3
"""run_mumax3_simulation.py —— MuMax3 模拟命令行入口（薄封装）。

解析必填的 ``--config``（实验 YAML 路径，``pathlib.Path``），委托
``mumax3_pipeline.run_parameter_set`` 执行整批参数组合模拟；
成功返回退出码 0，任何失败以异常直接上抛。编排逻辑全部位于
``mumax3_pipeline``，本文件不复制编排逻辑、不提供其他参数。
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

from micromagnetic_parameter_inversion import mumax3_pipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行解析器；仅注册必填 --config。"""
    parser = argparse.ArgumentParser(
        description=("Run the MuMax3 parameter set defined by an experiment YAML config."),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to the experiment YAML config defining the parameter set.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并委托 run_parameter_set，成功返回 0；异常直接上抛。"""
    args = _build_arg_parser().parse_args(argv)
    mumax3_pipeline.run_parameter_set(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
