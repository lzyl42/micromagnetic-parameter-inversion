#!/usr/bin/env python3
"""train_cnn1d.py —— CNN1D 训练入口（已实现，薄封装 ``training.run``）。

用法：仅 ``--config <training YAML>``（无其它开关）。本入口声明
``expected_kind="cnn1d"``，即要求 ``model.kind == "cnn1d"``；配置为 ``mlp``
在加载数据、创建目录之前即抛 ``ConfigError``。

与 ``train_mlp.py`` 共用 ``training.run`` 编排，使用**相同的样本目录与冻结
split**，但完全**不复用** MLP 的权重/checkpoint/预处理统计量/产物：每次
run 自行做 train-only 拟合，产物写入独立目录
``output_root()/training/cnn1d/<dataset_name>/<run_name>``（``output_dir``
显式覆盖时原样使用），与 MLP 输出互不干扰。test 仍只在
``evaluate_model.py`` 中评估，训练与选模只用 train/val。

错误处理：已知契约错误（ConfigError/DataError/PreprocessingError/
FileExistsError/TrainingError）向 stderr 友好输出并返回退出码 2；其余
异常直接上抛（bug）。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence
from pathlib import Path

from micromagnetic_parameter_inversion import preprocessing, training, training_data
from micromagnetic_parameter_inversion.training_config import ConfigError


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行解析器：仅注册必填 --config。"""
    parser = argparse.ArgumentParser(
        prog="train_cnn1d.py",
        description=(
            "Train the CNN1D inverse-regression model from prepared samples "
            "(CPU-friendly; test split is never touched)."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to the training YAML config (e.g. configs/training/cnn1d.yaml).",
    )
    return parser


def run(config_path: Path) -> Path:
    """执行 CNN1D 训练（``training.run`` 的薄封装），返回 run 目录。"""
    return training.run(config_path, expected_kind="cnn1d")


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行训练；已知契约错误友好输出并返回 2。"""
    args = _build_arg_parser().parse_args(argv)
    try:
        run(args.config)
    except (
        ConfigError,
        preprocessing.PreprocessingError,
        training_data.DataError,
        training.TrainingError,
        FileExistsError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
