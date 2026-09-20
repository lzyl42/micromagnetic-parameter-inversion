#!/usr/bin/env python3
"""train_mlp.py —— MLP 训练入口（已实现，薄封装 ``training.run``）。

用法：仅 ``--config <training YAML>``（无其它开关）。本入口声明
``expected_kind="mlp"``，即要求 ``model.kind == "mlp"``；配置为 ``cnn1d``
在加载数据、创建目录之前即抛 ``ConfigError``。

完整流程（同仓库其它入口）由 ``training.run`` 实现：load_config（一次）→
kind 早拒 → 定位 ``data_root()/samples/<dataset_name>/`` 的 dataset_meta/
冻结 split → 仅构造 train/val → train-only ``preprocessing.fit`` → 契约冻结
→ ``training.train_model`` → 训练成功后写 run 目录（split 原字节副本 /
config_resolved.yaml / preprocessing.yaml / metrics.json / best.pt /
final.pt）。默认输出目录
``output_root()/training/mlp/<dataset_name>/<run_name>``（``output_dir``
显式覆盖时原样使用）；目录已存在则拒绝覆盖。

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
        description=(
            "Train the MLP inverse-regression model from prepared samples "
            "(CPU-friendly; test split is never touched)."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to the training YAML config (e.g. configs/training/mlp.yaml).",
    )
    return parser


def run(config_path: Path) -> Path:
    """执行 MLP 训练（``training.run`` 的薄封装），返回 run 目录。"""
    return training.run(config_path, expected_kind="mlp")


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
