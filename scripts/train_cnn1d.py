#!/usr/bin/env python3
"""train_cnn1d.py —— CNN1D 训练入口（**架构审核骨架，未实现**）。

本脚本目前只占位未来的 CNN1D 训练入口；**直接执行不会创建目录、不会加载
数据、不会调用任何训练逻辑**，仅打印未实现提示并以非 0 退出码结束
（``--help`` 正常显示骨架用途）。

未来实现步骤（评审通过后落地，暂不实现）：

1. 解析训练 YAML 一次并校验 ``model.kind == "cnn1d"``；kind 不匹配时在加载
   数据或创建目录**之前**早拒；
2. 调用 ``training.run``（未来与 ``train_model`` 同模块的共享编排），使用与
   MLP **相同的样本目录与冻结 split**，但不复用 MLP 权重/ckpt/预处理/产物；
   CNN 自行做 train-only 拟合；
3. 产物写入独立目录
   ``output_root()/training/cnn1d/<dataset_name>/<run_name>``，与 MLP 输出
   互不干扰；旧 ``train_mlp.py`` 命令与输出路径保持不变；
4. 保留 ``output_dir`` 覆盖与失败/best-final 语义；
5. test 仍只在独立 ``evaluate_model.py`` 中评估；训练与选模只用 train/val，
   **test 不得用于调参**。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

_NOT_IMPLEMENTED_MESSAGE = (
    "train_cnn1d.py 仅为架构审核骨架，尚未实现，不可训练；待架构评审通过后再按批准阶段实现。"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建骨架参数解析器：仅提供 ``--help`` 说明用途，暂不注册 ``--config``。"""
    return argparse.ArgumentParser(
        prog="train_cnn1d.py",
        description=(
            "[骨架/未实现] 拟议的 CNN1D 反演训练入口，仅供架构审核；"
            "本版本不可训练、不创建目录、不加载数据。"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并明确提示未实现；恒返回非 0（骨架不可训练）。"""
    _build_arg_parser().parse_args(argv)
    print(f"error: {_NOT_IMPLEMENTED_MESSAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
