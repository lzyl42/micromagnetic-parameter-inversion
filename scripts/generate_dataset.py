#!/usr/bin/env python3
"""generate_dataset.py —— 生成实验 YAML 并有界并发运行 MuMax3 模拟。

当前预设：Protocol B（固定离散 benchmark，训练优先）Sobol 1024 主域点集
（dataset cofeb_protocol_b_a2_sobol1024_v1：ES12、40×20×4、10 ps × 401 点
= 关场后 0..4 ns、仅 pulse_A2）。运行前必须确认目标输出 set 目录
data/raw/cofeb_protocol_b_a2_sobol1024_v1/<parameter_set_id>/ 不存在；
若存在则不得运行（pipeline 会 FileExistsError 拒绝，不得删除已有数据）。

FIXED_CONFIGS（所有固定协议字段）× PARAMETERS（仅 alpha/Ku）的每个组合
构造一份实验 YAML（schema 严格同 configs/experiments/mumax3_simulation.yaml），
写入 artifacts/generated_configs/<dataset_name>/，每份写完立即经
mumax3_pipeline.run_parameter_set 运行。并发由脚本常量 MAX_WORKERS 唯一控制
（无 CLI/外部文件/框架）：每套 fixed config 内任意时刻最多 MAX_WORKERS 个
已提交未结束的模拟在途，不一次排队全部参数；每组原始顺序编号（即文件名
序号）在调度前确定，与完成顺序无关。任一任务（配置写出或模拟）失败后停
止补充提交，等待已启动任务自然结束并记录其结果，最后上抛第一个原异常
（fail-fast）；Ctrl-C 同样停止补充提交并等待已提交任务自然结束（不强杀
子进程）并重抛。不提供 resume/清理/跳过已有目录。PARAMETERS 使用固定
seed 生成 1024 个 Sobol 点（见其定义），不读取外部参数文件；协议不同的
数据不得混用。
"""

from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import qmc

from micromagnetic_parameter_inversion import mumax3_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 唯一并发配置：每套 fixed config 内最多 MAX_WORKERS 个在途模拟。
MAX_WORKERS = 2

# Protocol B（固定离散 benchmark）唯一 fixed config：与
# configs/experiments/mumax3_simulation.yaml 固定字段一致——ES12、
# 40×20×4、10 ps × 401 点（关场后 0..4 ns）、RT=-1、仅 pulse_A2
# （y 2 mT、50 ps）。若改变任一固定协议（material/geometry/recording/
# numerics/pulses），必须更换 dataset_name；协议不同的数据不得混用。
FIXED_CONFIGS: list[dict[str, Any]] = [
    {
        "dataset_name": "cofeb_protocol_b_a2_sobol1024_v1",
        "material": {
            "ms_a_per_m": 1.25e6,  # A/m
            "aex_j_per_m": 15e-12,  # J/m
            "anisotropy_axis": [1.0, 0.0, 0.0],  # 易轴 +x
        },
        "geometry": {
            "size_m": [100e-9, 50e-9, 2e-9],  # 真三轴椭球全直径 [m]
            "cells": [40, 20, 4],
        },
        "initial_m": [1.0, 0.0, 0.0],  # +x
        "recording": {
            "sample_interval_s": 10e-12,  # 10 ps
            "sample_count": 401,  # 关场后 0..4 ns
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

# Sobol 1024 点，scramble=True、rng=42；alpha 对数采样 [0.004, 0.020]，
# Ku 线性采样 [2000, 30000] J/m³。
PARAMETERS: list[dict[str, float]] = [
    {
        "alpha": 0.004 * 5 ** float(u),
        "ku_j_per_m3": 2000.0 + 28000.0 * float(v),
    }
    for u, v in qmc.Sobol(d=2, scramble=True, rng=42).random_base2(m=10)
]

_LOG_LOCK = threading.Lock()


def _log(message: str) -> None:
    """线程安全单行 stdout 日志：时间戳前缀、压平换行、立即 flush。"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    line = line.replace("\r", " ").replace("\n", " ")
    with _LOG_LOCK:
        print(line, flush=True)


def build_config(fixed_config: dict[str, Any], parameter: dict[str, float]) -> dict[str, Any]:
    """深拷贝 fixed config 并把 (alpha, Ku) 注入 material；不改顶部常量。"""
    config = copy.deepcopy(fixed_config)
    config["material"]["alpha"] = parameter["alpha"]
    config["material"]["ku_j_per_m3"] = parameter["ku_j_per_m3"]
    return config


def _config_filename(index: int, alpha: float, ku_j_per_m3: float) -> str:
    """简单序号 + alpha/Ku 文件名（目录已按 dataset_name 分开，无需协议索引）。"""
    return f"{index:02d}_alpha{alpha:.6f}_ku{ku_j_per_m3:g}.yaml"


def _task_prefix(dataset_name: str, index: int, total: int, parameter: dict[str, float]) -> str:
    """单任务日志前缀：dataset、组序号/总数与 alpha/Ku。"""
    return (
        f"[{dataset_name}] 组 {index}/{total} "
        f"alpha={parameter['alpha']:.6f} ku={parameter['ku_j_per_m3']:g}"
    )


def _write_config_yaml(config: dict[str, Any], config_path: Path) -> None:
    """按原语义写出 YAML（目标已存在则覆盖，不声称不覆盖）。"""
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def run_simulation(config_path: Path) -> None:
    """运行单份实验配置（阻塞至结束）；失败直接上抛（薄封装，便于替换/mock）。"""
    mumax3_pipeline.run_parameter_set(config_path)


def _run_task(task: tuple[str, dict[str, Any], Path]) -> None:
    """单任务 helper：写出配置、记录启动，运行模拟，记录完成/失败与耗时。

    配置写出在 try 内：写失败同样走失败日志并经 future 异常收割。
    """
    prefix, config, config_path = task
    started = time.perf_counter()
    try:
        _write_config_yaml(config, config_path)
        _log(f"{prefix} 配置已写出: {config_path.name}")
        _log(f"{prefix} 模拟启动: {config_path.name}")
        run_simulation(config_path)
    except Exception as exc:
        duration = time.perf_counter() - started
        _log(f"{prefix} 模拟失败 耗时={duration:.3f}s {type(exc).__name__}: {exc}")
        raise
    duration = time.perf_counter() - started
    _log(f"{prefix} 模拟完成 耗时={duration:.3f}s")


def _validate_max_workers(max_workers: int) -> int:
    """校验并发上限为正整数；bool/非整数/非正值一律拒绝。"""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError(
            f"max_workers 必须为 int，得到 {type(max_workers).__name__}: {max_workers!r}"
        )
    if max_workers <= 0:
        raise ValueError(f"max_workers 必须为正整数，得到 {max_workers}")
    return max_workers


def _run_parameter_group(
    fixed_config: dict[str, Any], parameters: list[dict[str, float]], max_workers: int
) -> None:
    """单套 fixed config：写出编号稳定的 YAML 并有界并发运行模拟。

    调度前先确定每组的原始顺序编号（即文件名序号，与完成顺序无关）；
    任意时刻最多 max_workers 个已提交未结束 future，不一次排队全部参数。
    wait(FIRST_COMPLETED) 收割整个 done 批次：先记录全部结果（含失败），
    再决定是否补充提交；发现失败后停止补充提交，等待已启动任务自然结束
    并记录其结果，最后上抛第一个原异常。Ctrl-C 同样停止补充提交并等待
    已提交任务自然结束（不强杀子进程），随后重抛。
    """
    dataset_name = str(fixed_config["dataset_name"])
    output_dir = PROJECT_ROOT / "artifacts" / "generated_configs" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 调度前确定原始顺序编号与文件名（编号稳定，与完成顺序无关）。
    # 任务元组：(日志前缀, 完整配置 dict, 配置文件路径)。
    tasks: list[tuple[str, dict[str, Any], Path]] = []
    for index, parameter in enumerate(parameters, start=1):
        config = build_config(fixed_config, parameter)
        config_path = output_dir / _config_filename(
            index, parameter["alpha"], parameter["ku_j_per_m3"]
        )
        prefix = _task_prefix(dataset_name, index, len(parameters), parameter)
        tasks.append((prefix, config, config_path))

    _log(f"[{dataset_name}] 开始: 总数={len(tasks)} 并发={max_workers}")

    first_error: Exception | None = None
    completed = 0
    failed = 0
    submitted = 0
    pending_interrupt: KeyboardInterrupt | None = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: set[Future[None]] = set()
        next_index = 0

        def submit_ready() -> None:
            """把在途数量补足到 max_workers；发现失败后不再补充提交。"""
            nonlocal next_index, submitted
            while len(futures) < max_workers and next_index < len(tasks) and first_error is None:
                task = tasks[next_index]
                next_index += 1
                futures.add(executor.submit(_run_task, task))
                submitted += 1

        def collect(done: set[Future[None]]) -> None:
            """收割整个 done 批次并记录结果与累计进度；是否补充提交由调用方决定。"""
            nonlocal completed, failed, first_error
            for future in done:
                futures.discard(future)
                try:
                    future.result()
                except Exception as exc:
                    failed += 1
                    if first_error is None:
                        first_error = exc
                else:
                    completed += 1
            _log(
                f"[{dataset_name}] 进度: 已完成={completed} 已失败={failed} "
                f"未启动={len(tasks) - submitted}"
            )

        try:
            submit_ready()
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                collect(done)  # 先处理整组 done（含失败），再决定是否补充提交
                if first_error is None:
                    submit_ready()
        except KeyboardInterrupt as interrupt:
            _log(f"[{dataset_name}] 收到中断，停止补充提交，等待已提交任务自然结束")
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                collect(done)
            pending_interrupt = interrupt

    not_started = len(tasks) - submitted
    _log(
        f"[{dataset_name}] 汇总: 总计={len(tasks)} 已完成={completed} "
        f"已失败={failed} 未启动={not_started}"
    )

    if first_error is not None:
        raise first_error
    if pending_interrupt is not None:
        raise pending_interrupt


def generate_dataset(
    fixed_configs: list[dict[str, Any]],
    parameters: list[dict[str, float]],
    max_workers: int = MAX_WORKERS,
) -> None:
    """对每套 fixed config × 每个 parameter 写出 YAML 并有界并发运行模拟。

    max_workers 缺省取脚本常量 MAX_WORKERS；非法值在任何文件写出前拒绝。
    """
    workers = _validate_max_workers(max_workers)
    for fixed_config in fixed_configs:
        _run_parameter_group(fixed_config, parameters, workers)


def main() -> None:
    generate_dataset(FIXED_CONFIGS, PARAMETERS, max_workers=MAX_WORKERS)


if __name__ == "__main__":
    main()
