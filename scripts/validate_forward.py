#!/usr/bin/env python3
"""validate_forward.py —— 单 checkpoint + 单原始参数组的正向回代验证。

``--checkpoint`` 指定模型（结构/契约/预处理全部从 ckpt 恢复），``--sample-dir``
指定原始参数组目录（config.yaml/index.csv/各 pulse trajectory.csv），
``--output`` 必填且已存在拒绝覆盖。

流程：推理预测 (alpha, Ku) → 原配置只换 alpha/Ku/dataset_name
（``<源名>_forward_<uuid 短码>``）写出 forward_config.yaml → 经
``mumax3_pipeline.run_parameter_set`` 重新 Relax 并逐 pulse 模拟（默认布局
``data_root()/raw/<forward dataset_name>/<parameter_set_id>/``，真实模拟目录
写入 metrics.json 并打印）→ 与观测轨迹比较并绘图。

``--output`` 目录产物：prediction.json、forward_config.yaml、metrics.json、
comparisons/<pulse_id>.csv、figures/<pulse_id>.png|svg；模拟原始数据保留在
pipeline 默认 data_root()/raw 数据集目录（独立 dataset_name，不覆盖观测数据）。
``--dry-run`` 只写 prediction.json 与 forward_config.yaml，不运行 MuMax3。

已知契约错误向 stderr 友好输出并返回退出码 2；真实模拟属长任务，需用户批准。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

from micromagnetic_parameter_inversion import forward_validation
from micromagnetic_parameter_inversion.mumax3_config import ConfigError
from micromagnetic_parameter_inversion.preprocessing import PreprocessingError
from micromagnetic_parameter_inversion.training import TrainingError
from micromagnetic_parameter_inversion.training_data import DataError


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行解析器：必填 --checkpoint/--sample-dir/--output，可选 --dry-run。"""
    parser = argparse.ArgumentParser(
        description=(
            "Forward validation: predict (alpha, Ku) from one raw parameter group, "
            "re-run the same MuMax3 excitations into a fresh raw dataset, and compare."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=pathlib.Path,
        help="Path to the checkpoint file (structure/contract/preprocessing come from it).",
    )
    parser.add_argument(
        "--sample-dir",
        required=True,
        type=pathlib.Path,
        help="Path to one raw parameter group directory (config.yaml/index.csv/trajectories).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help=(
            "New report directory (prediction.json/forward_config.yaml/metrics.json/"
            "comparisons/figures); existing directories are refused."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write prediction.json and forward_config.yaml; do not run MuMax3.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行正向回代；已知契约错误友好输出并返回 2。"""
    args = _build_arg_parser().parse_args(argv)
    try:
        forward_validation.run_forward_validation(
            args.checkpoint, args.sample_dir, args.output, dry_run=args.dry_run
        )
    except (
        ConfigError,
        DataError,
        PreprocessingError,
        TrainingError,
        forward_validation.ForwardValidationError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
