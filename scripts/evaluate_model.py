#!/usr/bin/env python3
"""evaluate_model.py —— run/ckpt 定位的常规 evaluate（已实现）。

``--run`` 定位训练 run，``--checkpoint`` 可选（缺省 ``<run>/best.pt``；
无论显式与否，ckpt 必须与 run 内 split 副本 SHA 绑定一致）；不要求提供
当前训练 config，结构/预处理/label 全部从 checkpoint 恢复。
``--split val|test`` 选择绑定 split 副本中的成员（默认 test；val 有 best
checkpoint 选择偏差，test 不得用于调参或模型选择）。

主流程：

1. 预检 ``<run>/<split>_metrics.json`` 与 ``<run>/<split>_predictions.csv``
   不存在（不覆盖旧评估，先于一切计算）；
2. ``evaluation.run_evaluation(run_dir, checkpoint_path, split=split)`` →
   ``(EvaluationReport, rows)``（纯计算）；
3. 复核预检后按序写出：``<split>_metrics.json``（report 序列化：
   ``main``/``control`` + ``provenance``（实际加载 ckpt 的路径/文件
   sha256/来自该 ckpt 的 split_sha256/所选 split），``allow_nan=False``）、
   ``<split>_predictions.csv``（``export_test_predictions``，LF、每 psid
   一行、拒绝覆盖）。

错误处理：已知契约错误（EvaluationError/DataError/PreprocessingError/
TrainingError/ConfigError/FileExistsError/FileNotFoundError）向 stderr
友好输出并返回退出码 2；其余异常直接上抛（bug）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from micromagnetic_parameter_inversion import evaluation, preprocessing, training, training_data
from micromagnetic_parameter_inversion.evaluation import EvaluationReport
from micromagnetic_parameter_inversion.training_config import ConfigError


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行解析器：必填 --run，可选 --checkpoint。"""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained run on its bound val/test split and export "
            "physical-unit metrics (split SHA-bound to the checkpoint)."
        ),
    )
    parser.add_argument(
        "--run",
        required=True,
        type=pathlib.Path,
        help="Path to the training run directory (its split copy binds the evaluation).",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to the checkpoint file (default: <run>/best.pt; split SHA must match).",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help=(
            "Evaluation split (default: test). Val metrics have best-checkpoint selection "
            "bias; test must never be used for model selection."
        ),
    )
    return parser


def _subset_to_dict(subset: evaluation.SubsetMetrics) -> dict[str, Any]:
    """SubsetMetrics → JSON 映射（空子集指标为 null）。"""
    return {
        "n": subset.n,
        "mae_alpha": subset.mae_alpha,
        "rmse_alpha": subset.rmse_alpha,
        "mae_ku": subset.mae_ku,
        "rmse_ku": subset.rmse_ku,
        "mape_percent_alpha": subset.mape_percent_alpha,
        "mape_percent_ku": subset.mape_percent_ku,
    }


def _metrics_document(report: EvaluationReport) -> dict[str, Any]:
    """EvaluationReport → ``<split>_metrics.json`` 文档。

    键：``main``/``control``（n 与 MAE/RMSE/MAPE，空子集或零分母列指标为
    null）+ ``provenance``（实际加载 ckpt 的路径/文件 sha256/来自该 ckpt 的
    split_sha256/所选 split——JSON 来源一定对应实际加载的模型）。
    """
    return {
        "main": _subset_to_dict(report.main),
        "control": _subset_to_dict(report.control),
        "provenance": {
            "checkpoint_path": report.provenance.checkpoint_path,
            "checkpoint_sha256": report.provenance.checkpoint_sha256,
            "split_sha256": report.provenance.split_sha256,
            "split": report.provenance.split,
        },
    }


def _precheck(artifacts: Sequence[Path]) -> None:
    """评估产物预检：任一已存在 → FileExistsError（不覆盖旧评估）。"""
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(f"评估产物已存在，拒绝覆盖: {existing}")


def run(
    run_dir: Path,
    checkpoint_path: Path | None = None,
    *,
    split: Literal["val", "test"] = "test",
) -> Path:
    """执行常规 evaluate 并把产物写入 run 目录，返回该目录。

    产物：``<split>_metrics.json`` 与 ``<split>_predictions.csv``，均拒绝覆盖。
    默认 test；val 受 best checkpoint 选择偏差影响，test 不得用于模型选择。
    """
    if split not in ("val", "test"):
        raise evaluation.EvaluationError("评估 split 仅允许 val 或 test")
    run_dir = Path(run_dir)
    metrics_path = run_dir / f"{split}_metrics.json"
    csv_path = run_dir / f"{split}_predictions.csv"
    _precheck((metrics_path, csv_path))
    report, rows = evaluation.run_evaluation(run_dir, checkpoint_path, split=split)
    _precheck((metrics_path, csv_path))  # 评估期间不得出现（TOCTOU 复核）
    metrics_path.write_text(
        json.dumps(_metrics_document(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    evaluation.export_test_predictions(csv_path, rows)
    print(f"评估完成: {run_dir}")
    print(f"  {split} n: main={report.main.n}, control={report.control.n}")
    print(f"  产物: {metrics_path.name}, {csv_path.name}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行评估；已知契约错误友好输出并返回 2。"""
    args = _build_arg_parser().parse_args(argv)
    try:
        run(args.run, args.checkpoint, split=args.split)
    except (
        ConfigError,
        training_data.DataError,
        preprocessing.PreprocessingError,
        training.TrainingError,
        evaluation.EvaluationError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
