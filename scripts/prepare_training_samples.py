#!/usr/bin/env python3
"""prepare_training_samples.py —— raw → data/samples/<dataset>/。

读取训练 YAML 配置（占位 dataset_name/run_name 拒绝），解析
``--parameter-set-ids``：显式 psid 列表或 ``all``（只展开所选 dataset
目录下的子目录，绝不跨 dataset 扫描；两者不可混用）。随后经
training_data 冻结协议快照、逐组读取样本、判定 split、写前预检拒绝覆盖
（无事务）。生成的 dataset 子目录不入 Git。

schema 数值/字段违反抛 ConfigError，布局/成员违反抛 DataError；main 捕获
后向 stderr 输出友好错误并返回退出码 2，不向用户抛 traceback。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from micromagnetic_parameter_inversion import paths, training_data
from micromagnetic_parameter_inversion.training_config import ConfigError, load_config
from micromagnetic_parameter_inversion.training_data import DataError, DatasetMeta, Sample


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行解析器：必填 --config 与 --parameter-set-ids。"""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize MuMax3 raw outputs into per-parameter-set npz samples "
            "under data/samples/<dataset_name>/."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to the training YAML config (e.g. configs/training/mlp.yaml).",
    )
    parser.add_argument(
        "--parameter-set-ids",
        required=True,
        nargs="+",
        metavar="ID",
        help="Explicit parameter set IDs, or the literal 'all' (dataset-local only).",
    )
    return parser


def _require_psid(name: str) -> str:
    """CLI 侧 psid 预检：非空安全单路径段（详细校验在 training_data）。"""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise DataError(f"非法 parameter_set_id: {name!r}")
    return name


def _resolve_psids(
    raw_dataset_dir: pathlib.Path, parameter_set_ids: Sequence[str]
) -> tuple[str, ...]:
    """解析所选 psid：显式列表或 ``all``（仅单 dataset 目录内展开）。"""
    if len(parameter_set_ids) == 1 and parameter_set_ids[0] == "all":
        if not raw_dataset_dir.is_dir():
            _fail_dir(raw_dataset_dir)
        psids = sorted(entry.name for entry in raw_dataset_dir.iterdir() if entry.is_dir())
        if not psids:
            raise DataError(f"dataset 目录下没有参数组子目录: {raw_dataset_dir}")
        return tuple(psids)
    if "all" in parameter_set_ids:
        raise DataError("'all' 不能与显式 parameter_set_id 混用")
    psids = tuple(_require_psid(name) for name in parameter_set_ids)
    if len(set(psids)) != len(psids):
        raise DataError(f"重复的 parameter_set_id: {list(psids)}")
    for psid in psids:
        if not (raw_dataset_dir / psid).is_dir():
            raise DataError(f"参数组目录不存在: {raw_dataset_dir / psid}")
    return psids


def _fail_dir(raw_dataset_dir: pathlib.Path) -> None:
    raise DataError(f"dataset raw 目录不存在: {raw_dataset_dir}")


def _collect_samples(
    raw_dataset_dir: pathlib.Path,
    psids: Sequence[str],
    protocol: training_data.ProtocolSummary,
    dataset_name: str,
) -> tuple[list[Sample], DatasetMeta]:
    """逐组读取样本并汇总 dataset_meta（含 sha256/来源路径/生成时间溯源）。"""
    samples: list[Sample] = []
    labels: dict[str, tuple[float, float]] = {}
    config_sha256: dict[str, str] = {}
    for psid in psids:
        config_path = raw_dataset_dir / psid / "config.yaml"
        if not config_path.is_file():
            raise DataError(f"config.yaml 快照缺失: {config_path}")
        config_sha256[psid] = training_data.sha256_file(config_path)
        sample, group_labels = training_data.read_parameter_group(
            raw_dataset_dir, psid, protocol.pulse_order, protocol.n_time_steps
        )
        samples.append(sample)
        labels[psid] = group_labels
    meta = DatasetMeta(
        dataset_name=dataset_name,
        pulse_order=protocol.pulse_order,
        n_time_steps=protocol.n_time_steps,
        labels=labels,
        members=tuple(psids),
        protocol={
            "sample_interval_s": protocol.sample_interval_s,
            "pulses": {pid: dict(fields) for pid, fields in protocol.pulses.items()},
        },
        config_sha256=config_sha256,
        source_index_relpaths={psid: f"{psid}/index.csv" for psid in psids},
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return samples, meta


def run(config_path: pathlib.Path, parameter_set_ids: Sequence[str]) -> pathlib.Path:
    """执行完整 prepare 流程，返回输出目录。"""
    config = load_config(config_path)
    raw_dataset_dir = paths.data_root() / "raw" / config.dataset_name
    if not raw_dataset_dir.is_dir():
        _fail_dir(raw_dataset_dir)
    psids = tuple(sorted(_resolve_psids(raw_dataset_dir, parameter_set_ids)))
    protocol = training_data.load_protocol_snapshot(raw_dataset_dir, psids, config.data.pulse_order)
    samples, meta = _collect_samples(raw_dataset_dir, psids, protocol, config.dataset_name)
    split = training_data.make_split(psids, config.split)
    samples_dir = paths.data_root() / "samples" / config.dataset_name
    training_data.write_prepared_dataset(samples_dir, samples, meta, split)
    print(f"prepare 完成: {samples_dir}")
    print(
        f"  成员 {len(psids)} 组; split train/val/test = "
        f"{len(split.train)}/{len(split.val)}/{len(split.test)}"
    )
    print(f"  pulse 顺序: {', '.join(protocol.pulse_order)}; T = {protocol.n_time_steps}")
    return samples_dir


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行 prepare；已知契约错误友好输出，返回 2。"""
    args = _build_arg_parser().parse_args(argv)
    try:
        run(args.config, args.parameter_set_ids)
    except (ConfigError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
