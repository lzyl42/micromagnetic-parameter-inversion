#!/usr/bin/env python3
"""generate_dataset.py —— 生成实验 YAML 并顺序运行 MuMax3 模拟。

当前预设：F1 z16 测试（详见 progress.md「F1」节）。运行前必须确认
data/raw/cofeb_qc04_es12_xy40_z16_a2_v1/ 与
artifacts/generated_configs/cofeb_qc04_es12_xy40_z16_a2_v1/ 均不存在；
若存在则不得运行（pipeline 会 FileExistsError 拒绝，不得删除已有数据）。

FIXED_CONFIGS（所有固定协议字段）× PARAMETERS（仅 alpha/Ku）的每个组合
构造一份实验 YAML（schema 严格同 configs/experiments/mumax3_simulation.yaml），
写入 artifacts/generated_configs/<dataset_name>/，每份写完立即经
scripts/run_mumax3_simulation.py 顺序运行；失败直接上抛（fail-fast），
不提供 resume/清理/并行。多套协议候选的 dataset_name 必须各不相同；
协议不同的数据不得混用；正式训练生成时只保留冻结后的单套 fixed config。
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_mumax3_simulation.py"

# F1（z16 收敛测试）唯一 fixed config：ES12、xy40、z16、A2-only、R4、RT=-1。
# 若改变任一固定协议（material/geometry/recording/numerics/pulses），必须更换
# dataset_name；协议不同的数据不得混用。
FIXED_CONFIGS: list[dict[str, Any]] = [
    {
        "dataset_name": "cofeb_qc04_es12_xy40_z16_a2_v1",
        "material": {
            "ms_a_per_m": 1.25e6,  # A/m
            "aex_j_per_m": 15e-12,  # J/m
            "anisotropy_axis": [1.0, 0.0, 0.0],  # 易轴 +x
        },
        "geometry": {
            "size_m": [100e-9, 50e-9, 2e-9],  # 真三轴椭球全直径 [m]
            "cells": [40, 20, 16],
        },
        "initial_m": [1.0, 0.0, 0.0],  # +x
        "recording": {
            "sample_interval_s": 10e-12,  # 10 ps
            "sample_count": 401,  # 0..4 ns（R4）
        },
        "numerics": {
            "edge_smooth": 12,
            "solver": 5,
            "max_err": 1.0e-5,
            "max_dt_s": 1.0e-11,
            "gamma_ll_rad_per_t_s": 1.7595e11,
            "relax_torque_threshold_t": -1.0,  # 官方默认
        },
        "pulses": [
            {
                "pulse_id": "pulse_A2",
                "b_ext_amplitude_mT": 2.0,
                "direction": [0.0, 1.0, 0.0],  # y
                "duration_s": 50e-12,  # 50 ps
            },
        ],
    },
]

# F1 压力点：先 PL，后 PH（顺序执行）。
PARAMETERS: list[dict[str, float]] = [
    {"alpha": 0.004, "ku_j_per_m3": 2000.0},  # PL
    {"alpha": 0.004, "ku_j_per_m3": 30000.0},  # PH
]


def build_config(fixed_config: dict[str, Any], parameter: dict[str, float]) -> dict[str, Any]:
    """深拷贝 fixed config 并把 (alpha, Ku) 注入 material；不改顶部常量。"""
    config = copy.deepcopy(fixed_config)
    config["material"]["alpha"] = parameter["alpha"]
    config["material"]["ku_j_per_m3"] = parameter["ku_j_per_m3"]
    return config


def _config_filename(index: int, alpha: float, ku_j_per_m3: float) -> str:
    """简单序号 + alpha/Ku 文件名（目录已按 dataset_name 分开，无需协议索引）。"""
    return f"{index:02d}_alpha{alpha:.6f}_ku{ku_j_per_m3:g}.yaml"


def run_simulation(config_path: Path) -> None:
    """运行单份实验配置（阻塞至结束）；非零退出码抛 CalledProcessError。"""
    subprocess.run(
        [sys.executable, str(_RUN_SCRIPT), "--config", str(config_path)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def generate_dataset(
    fixed_configs: list[dict[str, Any]], parameters: list[dict[str, float]]
) -> None:
    """对每套 fixed config × 每个 parameter 写出 YAML，写完立即顺序运行。"""
    for fixed_config in fixed_configs:
        dataset_name = str(fixed_config["dataset_name"])
        output_dir = PROJECT_ROOT / "artifacts" / "generated_configs" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, parameter in enumerate(parameters, start=1):
            config = build_config(fixed_config, parameter)
            config_path = output_dir / _config_filename(
                index, parameter["alpha"], parameter["ku_j_per_m3"]
            )
            config_path.write_text(
                yaml.safe_dump(
                    config, sort_keys=False, default_flow_style=False, allow_unicode=True
                ),
                encoding="utf-8",
                newline="\n",
            )
            run_simulation(config_path)


def main() -> None:
    generate_dataset(FIXED_CONFIGS, PARAMETERS)


if __name__ == "__main__":
    main()
