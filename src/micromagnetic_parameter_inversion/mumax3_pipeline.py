"""MuMax3 模拟流水线：参数组校验 → equilibrium（一次）→ 各 pulse simulation → 轨迹/索引导出。

dataset_name 是固定协议（材料/几何/模拟）的命名约定，协议改变必须换新
dataset_name；parameter_set_id 只代表 alpha+Ku（split key）。不自动验证
同 dataset 跨 parameter_set_id 的协议一致性。

任一步失败直接上抛并停止；部分运行文件供人工诊断（临时 index.csv.tmp
会清除），不做自动恢复。MuMax3 一律经 external.run_mumax3 调用
（subprocess 参数列表，禁止 shell=True）。
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Final

from micromagnetic_parameter_inversion import external
from micromagnetic_parameter_inversion.mumax3_config import load_config, parameter_set_id
from micromagnetic_parameter_inversion.mumax3_results import parse_table, write_trajectory_csv
from micromagnetic_parameter_inversion.mumax3_script import (
    render_equilibrium_script,
    render_simulation_script,
)
from micromagnetic_parameter_inversion.paths import PROJECT_ROOT, data_root

# 模板原件只读，固定位于 PROJECT_ROOT/simulations/mumax3。
_TEMPLATE_DIR: Final[Path] = PROJECT_ROOT / "simulations" / "mumax3"
_EQUILIBRIUM_TEMPLATE_NAME: Final[str] = "equilibrium.mx3.in"
_SIMULATION_TEMPLATE_NAME: Final[str] = "simulation.mx3.in"

# 输出布局固定名（不进 YAML）。
_CONFIG_SNAPSHOT_NAME: Final[str] = "config.yaml"
_INDEX_NAME: Final[str] = "index.csv"
_RUN_LOG_NAME: Final[str] = "run.log"
_EQUILIBRIUM_DIR_NAME: Final[str] = "equilibrium"
_RUNS_DIR_NAME: Final[str] = "runs"
_EQUILIBRIUM_SCRIPT_NAME: Final[str] = "equilibrium.mx3"
_SIMULATION_SCRIPT_NAME: Final[str] = "simulation.mx3"
_EQUILIBRIUM_OUT_DIR_NAME: Final[str] = "equilibrium.out"
_SIMULATION_OUT_DIR_NAME: Final[str] = "simulation.out"

# 每次 MuMax3 成功执行后按契约必须存在且非空的输出（位于 .out/ 目录内）；
# 原生 log.txt/references.bib 不作要求。
_EQUILIBRIUM_REQUIRED_OUTPUTS: Final[tuple[str, ...]] = ("equilibrium.ovf",)
_SIMULATION_REQUIRED_OUTPUTS: Final[tuple[str, ...]] = (
    "table.txt",
    "m_t0.ovf",
    "m_tfinal.ovf",
)

# index.csv 固定最小字段。
_INDEX_FIELDS: Final[tuple[str, ...]] = (
    "parameter_set_id",
    "pulse_id",
    "alpha",
    "ku_j_per_m3",
    "b_ext_x_T",
    "b_ext_y_T",
    "b_ext_z_T",
    "pulse_duration_s",
    "trajectory_path",
)


def _write_text_deterministic(path: Path, text: str) -> None:
    """UTF-8 写出且强制 LF 换行，保证跨平台字节确定性。"""
    path.write_text(text, encoding="utf-8", newline="\n")


def _format_float(value: float) -> str:
    """浮点确定性格式：CPython 最短往返 repr（IEEE 754 下跨平台稳定）。"""
    return repr(float(value))


def _ensure_trailing_newline(text: str) -> str:
    """非空且缺尾换行的文本补一个换行，保证后续章节标题另起新行。"""
    return text if not text or text.endswith("\n") else text + "\n"


def _run_and_log(workdir: Path, script_name: str) -> None:
    """在工作目录内经 external.run_mumax3 执行脚本并确定性写出 run.log。

    run.log 记录脚本名、returncode 与捕获的 stdout/stderr 原文（stdout
    与 "--- stderr ---" 标题交界处缺尾换行时补一个换行）；非零
    returncode 抛含脚本路径与 returncode 的 RuntimeError（不重试）。
    """
    script_path = workdir / script_name
    completed = external.run_mumax3(script_path, workdir=workdir)
    log_text = (
        f"# script: {script_name}\n"
        f"# returncode: {completed.returncode}\n"
        "--- stdout ---\n"
        f"{_ensure_trailing_newline(completed.stdout)}"
        "--- stderr ---\n"
        f"{completed.stderr}"
    )
    _write_text_deterministic(workdir / _RUN_LOG_NAME, log_text)
    if completed.returncode != 0:
        raise RuntimeError(
            f"MuMax3 执行失败: {script_path} (workdir={workdir}, "
            f"returncode={completed.returncode})；现场见 {workdir / _RUN_LOG_NAME}"
        )


def _require_output_files(base_dir: Path, names: tuple[str, ...], context: str) -> None:
    """检查契约要求的产物文件存在且非空（is_file 且 st_size > 0）。"""
    missing_or_empty: list[str] = []
    for name in names:
        path = base_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            missing_or_empty.append(name)
    if missing_or_empty:
        raise FileNotFoundError(
            f"{context}: 缺失或为空的契约要求输出文件 {missing_or_empty} (目录 {base_dir})"
        )


def _write_index_csv(index_path: Path, rows: list[list[str]]) -> None:
    """原子写出 index.csv：固定最小字段、确定性行序（每 pulse 一行）。

    先写同目录 index.csv.tmp，成功后 Path.replace 到最终名；写出/替换
    异常时清除 tmp 并上抛，最终 index 保持不存在。
    """
    tmp_path = index_path.with_name(index_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(_INDEX_FIELDS)
            writer.writerows(rows)
        tmp_path.replace(index_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def run_parameter_set(config_path: Path) -> Path:
    """执行一个 parameter set 的全部模拟，返回其输出目录。

    输出根为 ``data_root()/raw/<dataset_name>/<parameter_set_id>/``，
    已存在抛 ``FileExistsError``；equilibrium 仅执行一次；每 pulse 独立
    工作目录；index.csv 仅在全部 pulse 成功后原子写出。
    """
    config = load_config(config_path)
    set_id = parameter_set_id(config.material.alpha, config.material.ku_j_per_m3)

    # 已存在即拒绝（不隐式覆盖、不清理失败现场）。
    out_dir = data_root() / "raw" / config.dataset_name / set_id
    if out_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {out_dir}")

    # 模板只读，先读再建目录：模板缺失时不留下半成品输出目录。
    equilibrium_template = (_TEMPLATE_DIR / _EQUILIBRIUM_TEMPLATE_NAME).read_text(encoding="utf-8")
    simulation_template = (_TEMPLATE_DIR / _SIMULATION_TEMPLATE_NAME).read_text(encoding="utf-8")

    equilibrium_dir = out_dir / _EQUILIBRIUM_DIR_NAME
    runs_dir = out_dir / _RUNS_DIR_NAME
    equilibrium_out_dir = equilibrium_dir / _EQUILIBRIUM_OUT_DIR_NAME

    out_dir.mkdir(parents=True)
    equilibrium_dir.mkdir()
    runs_dir.mkdir()

    # 配置快照：按原字节复制，不重排、不改写。
    shutil.copyfile(config_path, out_dir / _CONFIG_SNAPSHOT_NAME)

    # equilibrium 只渲染/执行一次，产出供全部 pulse 共享的 equilibrium.ovf。
    _write_text_deterministic(
        equilibrium_dir / _EQUILIBRIUM_SCRIPT_NAME,
        render_equilibrium_script(config, equilibrium_template),
    )
    _run_and_log(equilibrium_dir, _EQUILIBRIUM_SCRIPT_NAME)
    _require_output_files(equilibrium_out_dir, _EQUILIBRIUM_REQUIRED_OUTPUTS, "equilibrium 运行")

    # 每 pulse 独立工作目录；index 行在此收集，pulse 顺序与配置一致。
    index_rows: list[list[str]] = []
    for pulse in config.pulses:
        pulse_dir = runs_dir / pulse.pulse_id
        pulse_out_dir = pulse_dir / _SIMULATION_OUT_DIR_NAME
        pulse_dir.mkdir()
        _write_text_deterministic(
            pulse_dir / _SIMULATION_SCRIPT_NAME,
            render_simulation_script(config, pulse, simulation_template),
        )
        _run_and_log(pulse_dir, _SIMULATION_SCRIPT_NAME)
        _require_output_files(
            pulse_out_dir, _SIMULATION_REQUIRED_OUTPUTS, f"pulse {pulse.pulse_id!r} 运行"
        )

        rows = parse_table(
            pulse_out_dir / "table.txt",
            pulse_duration_s=pulse.duration_s,
            sample_interval_s=config.recording.sample_interval_s,
            sample_count=config.recording.sample_count,
        )
        trajectory_csv = pulse_dir / "trajectory.csv"
        write_trajectory_csv(rows, trajectory_csv)

        b_ext = [pulse.b_ext_amplitude_t * axis for axis in pulse.direction]
        index_rows.append(
            [
                set_id,
                pulse.pulse_id,
                _format_float(config.material.alpha),
                _format_float(config.material.ku_j_per_m3),
                _format_float(b_ext[0]),
                _format_float(b_ext[1]),
                _format_float(b_ext[2]),
                _format_float(pulse.duration_s),
                trajectory_csv.relative_to(out_dir).as_posix(),
            ]
        )

    # 全部 pulse 成功后原子写出最终 index.csv；失败现场保留供诊断。
    _write_index_csv(out_dir / _INDEX_NAME, index_rows)
    return out_dir
