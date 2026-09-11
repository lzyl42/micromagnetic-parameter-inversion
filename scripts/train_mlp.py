#!/usr/bin/env python3
"""train_mlp.py —— 配置 → 样本/划分 → 训练 → artifacts。

输出目录（``config.output_dir`` 或
``output_root()/training/mlp/<dataset_name>/<run_name>``）已存在则先于
一切训练动作拒绝覆盖。仅构造 train/val（test 不参与调参/模型选择，也不
加载）；契约自首个 train 样本冻结并交叉校验；split 副本按 raw 字节保存并
与 ckpt 绑定 SHA；预处理仅 train 组拟合。训练成功后按序写出 split.yaml
副本、config_resolved.yaml、preprocessing.yaml、metrics.json（不含 test
评估）、best.pt/final.pt（拒绝覆盖）。

错误处理：已知契约错误（ConfigError/DataError/PreprocessingError/
FileExistsError/TrainingError）向 stderr 友好输出并返回退出码 2；其余
异常直接上抛（bug）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from micromagnetic_parameter_inversion import (
    paths,
    preprocessing,
    training,
    training_config,
    training_data,
)
from micromagnetic_parameter_inversion.training_config import ConfigError, load_config
from micromagnetic_parameter_inversion.training_data import (
    DataError,
    DatasetMeta,
    InputContract,
    TrajectoryDataset,
)


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


def _freeze_contract(train_set: TrajectoryDataset) -> InputContract:
    """自首个**缓存**样本冻结输入契约（不回磁盘）；train 其余成员在缓存内校验。"""
    first = train_set.samples[0]
    contract = InputContract(
        pulse_order=first.pulse_ids,
        n_time_steps=int(first.x.shape[1]),
        t_s=first.t_s,
    )
    train_set.validate_contract(contract)
    return contract


def _metrics_payload(
    history: Sequence[training.EpochMetrics],
    best_val_loss: float | None,
    stop_reason: str,
    stop_epoch: int,
    detail: str | None,
) -> dict[str, Any]:
    """metrics.json 文档：history + best val + 停止状态（无 test 评估；有限值）。"""
    return {
        "history": [
            {"epoch": m.epoch, "train_loss": m.train_loss, "val_loss": m.val_loss} for m in history
        ],
        "best_val_loss": best_val_loss,
        "best_epoch": max(history, key=lambda m: -m.val_loss).epoch if history else None,
        "stop_reason": stop_reason,
        "stop_epoch": stop_epoch,
        "detail": detail,
    }


def _write_metrics(path: Path, result: training.TrainingResult) -> None:
    """写出 metrics.json（payload 见 ``_metrics_payload``）。"""
    payload = _metrics_payload(
        result.history,
        result.best_checkpoint.best_val_loss,
        result.stop_reason,
        result.stop_epoch,
        result.detail,
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _write_failure_metrics(path: Path, error: training.TrainingError) -> None:
    """首 epoch 数值失败的失败状态 metrics（无 checkpoint 可保留）。"""
    payload = _metrics_payload(
        [],
        None,
        "numerical_failure",
        error.epoch if error.epoch is not None else 0,
        error.detail if error.detail is not None else str(error),
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def run(config_path: Path) -> Path:
    """执行完整训练流程，返回 run 目录。"""
    config = load_config(config_path)
    samples_dir = paths.data_root() / "samples" / config.dataset_name
    if not samples_dir.is_dir():
        raise DataError(f"样本目录不存在: {samples_dir}")
    meta: DatasetMeta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)
    # 训练契约要求 train/val 非空（test 允许为空：不参与训练也不评估）。
    if not split.train:
        raise DataError(
            f"split.train 为空：训练至少需要 1 个 train 成员 "
            f"(split 副本位于 {samples_dir / 'split.yaml'})"
        )
    if not split.val:
        raise DataError(
            f"split.val 为空：early stopping 与 best 选择需要 val 成员 "
            f"(split 副本位于 {samples_dir / 'split.yaml'})"
        )

    output_dir = (
        Path(config.output_dir)
        if config.output_dir is not None
        else paths.output_root() / "training" / "mlp" / config.dataset_name / config.run_name
    )
    if output_dir.exists():
        raise FileExistsError(f"run 目录已存在，拒绝覆盖: {output_dir}")

    # 仅构造 train/val；test 不参与任何调参与模型选择，也不加载。
    # 每 npz 恰好读盘一次：train 先无契约加载并缓存，契约自首个缓存样本
    # 冻结后在缓存内校验；val 一次读盘即按契约验证。
    train_set = TrajectoryDataset(samples_dir, meta, split.train)
    contract = _freeze_contract(train_set)
    val_set = TrajectoryDataset(samples_dir, meta, split.val, contract=contract)
    state = preprocessing.fit(train_set, config.preprocessing, config.label)

    split_source = samples_dir / "split.yaml"
    split_bytes = split_source.read_bytes()
    split_sha256 = training_data.sha256_file(split_source)
    meta_source = samples_dir / "dataset_meta.yaml"
    meta_sha256 = training_data.sha256_file(meta_source)

    try:
        result = training.train_model(
            config,
            train_set,
            val_set,
            state,
            contract,
            split_sha256,
            dataset_meta_relpath=meta_source.name,
            dataset_meta_sha256=meta_sha256,
        )
    except training.TrainingError as exc:
        # 首 epoch 数值失败：无可保留权重 → 仅写失败状态 metrics（无伪 ckpt）。
        output_dir.mkdir(parents=True)
        _write_failure_metrics(output_dir / "metrics.json", exc)
        raise

    # run 目录在训练返回后创建；此后失败可留部分产物（不事务）。
    output_dir.mkdir(parents=True)
    (output_dir / "split.yaml").write_bytes(split_bytes)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            training_config.config_to_mapping(config), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    (output_dir / "preprocessing.yaml").write_text(
        yaml.safe_dump(preprocessing.state_to_mapping(state), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_metrics(output_dir / "metrics.json", result)
    training.save_checkpoint(output_dir / "best.pt", result.best_checkpoint)
    training.save_checkpoint(output_dir / "final.pt", result.final_checkpoint)

    if result.stop_reason == "numerical_failure":
        # 有效快照（上一完整 epoch）已保存，但本次训练视为失败：非零退出、
        # 不打印训练完成。
        raise training.TrainingError(
            f"训练因数值失败停止于 epoch {result.stop_epoch}: {result.detail}",
            epoch=result.stop_epoch,
            detail=result.detail,
        )

    print(f"训练完成: {output_dir}")
    print(f"  epochs={len(result.history)}; best_val_loss={result.best_checkpoint.best_val_loss!r}")
    print(f"  stop: {result.stop_reason} @ epoch {result.stop_epoch}")
    return output_dir


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
