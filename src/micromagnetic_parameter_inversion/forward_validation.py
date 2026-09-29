"""正向回代验证：预测 (alpha, Ku) → 同 pulse 正向模拟 → 与观测严格比较。

流程（轻量闭环；输入为脚本生成的合法产物，按可信处理）：

1. ``training.load_any_checkpoint`` 读取 checkpoint，本模块内联恢复模型
   （CPU、eval、strict 权重加载）；
2. ``read_sample``：``training_data.read_parameter_group`` 按 ckpt 契约读取
   float32 x（推理用；index 真值仅用于参数指标），并按 index 的
   trajectory_path 局部读 float64 轨迹；每 pulse 时间轴与 ``contract.t_s``
   按极小 atol 比对（同形状但不同采样协议不得喂模型）；
3. ``predict_parameter_set`` 推理；原配置只替换 material.alpha /
   material.ku_j_per_m3 / dataset_name（``<源名>_forward_<uuid 短码>``）写出
   ``forward_config.yaml``，写出后经现有 ``mumax3_config.load_config`` 校验；
4. dry-run 到此为止；否则 ``mumax3_pipeline.run_parameter_set(config_path)``
   按默认布局重新 Relax/逐 pulse 模拟，返回真实参数组目录
   （``data_root()/raw/<forward dataset_name>/<parameter_set_id>/``，独立
   dataset 不覆盖观测数据），主流程据此读回；
5. 逐 pulse 在 ``--output`` 写 ``comparisons/<pulse_id>.csv``、``metrics.json``
   （含实际模拟目录）与 ``figures/<pulse_id>.png|svg``；时间轴不一致即报错，
   不重采样。

``<output_dir>/``：prediction.json（真值/预测/绝对参数误差）、
forward_config.yaml、metrics.json、comparisons/<pulse_id>.csv、
figures/<pulse_id>.png|svg。失败保留现场，不自动清理。

绘图（三行两列）：行 = mx/my/mz；左列 = 观测（实线）与正向回代（虚线）重叠；
右列 = 残差 ``forward - observed`` 并带零线；x 轴为 ``t_s × 1e9``（ns），零点
即关场时刻（时间轴已由 pipeline 重锚定；不移相、不裁剪、不归一化、不平滑，
不改动输入数组）。各分量独立纵轴自适应，小振荡不被 |m| ≤ 1 全域压扁；图例
图级共享、置于绘图区之外，不遮挡曲线；图文英文（不依赖中文字库）。后端直接
使用 ``matplotlib.figure.Figure`` + ``FigureCanvasAgg``（不经 pyplot、无 GUI、
无全局状态、不碰 rcParams）；输出 ``str(output_base) + ".png" / ".svg"``，任一
目标已存在即 ``FileExistsError`` 拒绝覆盖，父目录自动创建，写图失败保留现场。
"""

from __future__ import annotations

import csv
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
import yaml
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from torch import nn

from micromagnetic_parameter_inversion import (
    mumax3_pipeline,
    preprocessing,
    training,
    training_data,
)
from micromagnetic_parameter_inversion.mumax3_config import load_config

# 时间轴比较容差：两条轴均来自 pipeline 的 i*dt 重锚，只吸收 float64 舍入；
# 远小于任何采样间隔，不允许用容差掩盖重采样/移相。
TIME_ATOL_S: Final[float] = 1e-24

_COMPONENT_MATH = (r"$m_x$", r"$m_y$", r"$m_z$")
_RESIDUAL_MATH = (r"$\Delta m_x$", r"$\Delta m_y$", r"$\Delta m_z$")

_COLOR_OBSERVED = "#222222"  # 近黑实线：观测（CSV 原始数据）
_COLOR_FORWARD = "#0072B2"  # Okabe-Ito 蓝虚线：正向回代
_COLOR_RESIDUAL = "#D55E00"  # Okabe-Ito 朱红：残差
_COLOR_ZERO_LINE = "#888888"


class ForwardValidationError(ValueError):
    """正向回代流程契约违反（时间轴不一致、权重/结构不匹配等）。"""


@dataclass(frozen=True)
class SampleData:
    """单个参数组观测读取结果（推理用 float32；比较/绘图用 float64 原值）。"""

    labels: tuple[float, float]  # index.csv 原值 (alpha, ku_j_per_m3)，仅用于参数指标
    x: NDArray[np.float32]  # [P, T, 3]，按 ckpt 契约 pulse 顺序
    times: dict[str, NDArray[np.float64]]  # pulse_id -> [T]
    trajectories: dict[str, NDArray[np.float64]]  # pulse_id -> [T, 3]，CSV 原值


def read_sample(sample_dir: Path, contract: training_data.InputContract) -> SampleData:
    """读取观测参数组：float32 x（推理）+ float64 轨迹/时间（比较、绘图）。

    复用 ``training_data.read_parameter_group``：index pulse 集合与轨迹形状按
    ckpt 契约校验（``contract.pulse_order`` / ``n_time_steps``），x 为 float32
    与训练/评估读入一致；float64 轨迹另从 index 的 trajectory_path 局部读取，
    每 pulse 时间轴再与 ``contract.t_s`` 按极小 atol 比对（同形状但不同采样
    协议的轨迹不得喂模型）。
    """
    sample_dir = Path(sample_dir)
    sample, labels = training_data.read_parameter_group(
        sample_dir.parent, sample_dir.name, contract.pulse_order, contract.n_time_steps
    )
    times, trajectories = _read_float64_trajectories(sample_dir, contract.pulse_order)
    for pulse_id in contract.pulse_order:
        _require_matching_times(pulse_id, contract.t_s, times[pulse_id])
    return SampleData(
        labels=(float(labels[0]), float(labels[1])),
        x=np.asarray(sample.x, dtype=np.float32),
        times=times,
        trajectories=trajectories,
    )


def _read_float64_trajectories(
    group_dir: Path, pulse_order: tuple[str, ...]
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    """从 index.csv 的 trajectory_path 读各 pulse 的 float64 时间列与轨迹。

    数据为脚本产物，只按已知相对路径读取；float64 原值供比较/绘图，与推理
    用的 float32 输入相互独立（不把推理中间量当原始轨迹）。
    """
    with (group_dir / "index.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    path_by_pulse = {row["pulse_id"].strip(): row["trajectory_path"].strip() for row in rows}
    times: dict[str, NDArray[np.float64]] = {}
    trajectories: dict[str, NDArray[np.float64]] = {}
    for pulse_id in pulse_order:
        if pulse_id not in path_by_pulse:
            raise ForwardValidationError(f"参数组缺少 pulse {pulse_id!r} 的轨迹: {group_dir}")
        values = np.loadtxt(
            group_dir / path_by_pulse[pulse_id],
            delimiter=",",
            skiprows=1,
            usecols=(1, 2, 3, 4),
            ndmin=2,
        )
        times[pulse_id] = np.asarray(values[:, 0], dtype=np.float64)
        trajectories[pulse_id] = np.asarray(values[:, 1:4], dtype=np.float64)
    return times, trajectories


def _restore_model(ckpt: training.ModelCheckpoint) -> nn.Module:
    """按已解析 checkpoint 恢复模型（CPU、eval、strict 权重加载；不重新读文件）。"""
    if isinstance(ckpt, training.CNNCheckpoint):
        model: nn.Module = training.build_cnn_model(
            ckpt.contract,
            channels=ckpt.channels,
            kernel_sizes=ckpt.kernel_sizes,
            pool_bins=ckpt.pool_bins,
            head_hidden_dims=ckpt.head_hidden_dims,
        )
    elif isinstance(ckpt, training.TransformerCheckpoint):
        model = training.build_transformer_model(
            ckpt.contract,
            d_model=ckpt.d_model,
            nhead=ckpt.nhead,
            num_layers=ckpt.num_layers,
            dim_feedforward=ckpt.dim_feedforward,
            dropout=ckpt.dropout,
            head_hidden_dims=ckpt.head_hidden_dims,
        )
    elif isinstance(ckpt, training.Checkpoint):
        model = training.build_model(ckpt.contract, ckpt.hidden_dims)
    else:
        raise ForwardValidationError(f"不支持的 checkpoint 类型: {type(ckpt).__name__}")
    try:
        model.load_state_dict(dict(ckpt.model_state_dict), strict=True)
    except RuntimeError as exc:
        raise ForwardValidationError(
            f"checkpoint 权重与恢复的模型结构不匹配（keys/尺寸）: {exc}"
        ) from exc
    model.to("cpu")
    model.eval()
    return model


def predict_parameter_set(
    model: nn.Module,
    ckpt: training.ModelCheckpoint,
    x: NDArray[np.float32] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """单组推理：float32 量化 → ``transform_x`` → 模型 → ``inverse_transform_y``。

    先显式量化 float32（与训练/评估读入一致；原始 float64 轨迹不进入模型）；
    真值不参与；返回物理单位 (alpha, ku_j_per_m3) float64 ``[2]``。
    """
    x32 = np.asarray(x, dtype=np.float32)
    x_norm = preprocessing.transform_x(ckpt.preprocessing, x32[None])
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_norm))
    physical = preprocessing.inverse_transform_y(ckpt.preprocessing, pred_norm.numpy())
    return np.asarray(physical[0], dtype=np.float64)


def write_forward_config(
    original_config_path: Path,
    forward_config_path: Path,
    *,
    alpha: float,
    ku_j_per_m3: float,
) -> None:
    """原配置只替换 alpha/Ku/dataset_name，写出 ``forward_config.yaml``。

    ``dataset_name`` 改为 ``<源名>_forward_<uuid 短码>``（8 位十六进制），保证
    多个模型/重复预测的回代模拟互不碰撞；其余字段保持原值。写出后经现有
    ``mumax3_config.load_config`` 校验（alpha<0 等非法值直接抛 ``ConfigError``，
    不裁剪）；目标已存在拒绝覆盖。
    """
    if forward_config_path.exists():
        raise FileExistsError(f"forward 配置已存在，拒绝覆盖: {forward_config_path}")
    mapping = yaml.safe_load(original_config_path.read_text(encoding="utf-8"))
    mapping["dataset_name"] = f"{mapping['dataset_name']}_forward_{uuid.uuid4().hex[:8]}"
    mapping["material"]["alpha"] = float(alpha)
    mapping["material"]["ku_j_per_m3"] = float(ku_j_per_m3)
    forward_config_path.parent.mkdir(parents=True, exist_ok=True)
    forward_config_path.write_text(
        yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    load_config(forward_config_path)


def run_forward_simulation(forward_config_path: Path) -> Path:
    """经既有 pipeline 重新 Relax 并逐 pulse 模拟，返回真实参数组目录。

    输出走 pipeline 默认布局 ``data_root()/raw/<dataset_name>/<psid>/``
    （独立 ``_forward_<uuid>`` dataset，不覆盖观测数据）；外部模拟仅经
    ``mumax3_pipeline`` → ``external``，不直接 subprocess。
    """
    return mumax3_pipeline.run_parameter_set(forward_config_path)


def trajectory_metrics(
    observed: NDArray[np.float64], forward: NDArray[np.float64]
) -> dict[str, float]:
    """逐 pulse 轨迹指标：三分量 RMSE、向量 RMSE、最大向量误差（float64）。"""
    diff = np.asarray(forward, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    component_rmse = np.sqrt(np.mean(diff**2, axis=0))
    return {
        "rmse_mx": float(component_rmse[0]),
        "rmse_my": float(component_rmse[1]),
        "rmse_mz": float(component_rmse[2]),
        "rmse_vector": float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))),
        "max_vector_error": float(np.max(np.linalg.norm(diff, axis=1))),
    }


def write_prediction_json(
    path: Path,
    *,
    labels: tuple[float, float],
    prediction: NDArray[np.float64],
) -> None:
    """写出 ``prediction.json``：真值、预测与绝对参数误差（拒绝覆盖）。

    真值只用于本文件的参数误差指标，不进入推理。
    """
    if path.exists():
        raise FileExistsError(f"prediction.json 已存在，拒绝覆盖: {path}")
    alpha_true, ku_true = float(labels[0]), float(labels[1])
    alpha_pred, ku_pred = float(prediction[0]), float(prediction[1])
    document = {
        "alpha_true": alpha_true,
        "ku_j_per_m3_true": ku_true,
        "alpha_pred": alpha_pred,
        "ku_j_per_m3_pred": ku_pred,
        "alpha_abs_error": abs(alpha_pred - alpha_true),
        "ku_j_per_m3_abs_error": abs(ku_pred - ku_true),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_comparison_csv(
    path: Path,
    times: NDArray[np.float64],
    observed: NDArray[np.float64],
    forward: NDArray[np.float64],
) -> None:
    """写出 ``sample_index,t_s_s,observed_*,forward_*,residual_*`` 比较表（LF、拒绝覆盖）。"""
    if path.exists():
        raise FileExistsError(f"比较表已存在，拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    residual = np.asarray(forward, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    lines = [
        "sample_index,t_s_s,observed_mx,observed_my,observed_mz,"
        "forward_mx,forward_my,forward_mz,residual_mx,residual_my,residual_mz"
    ]
    for index in range(len(times)):
        values = [
            float(times[index]),
            *(float(value) for value in observed[index]),
            *(float(value) for value in forward[index]),
            *(float(value) for value in residual[index]),
        ]
        lines.append(f"{index}," + ",".join(repr(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_metrics_json(
    path: Path,
    pulse_metrics: Mapping[str, Mapping[str, float]],
    *,
    simulation_dir: Path,
) -> None:
    """写出 ``metrics.json``：实际模拟目录 + 逐 pulse 指标（不聚合总分）。"""
    if path.exists():
        raise FileExistsError(f"metrics.json 已存在，拒绝覆盖: {path}")
    document = {
        "simulation_dir": str(simulation_dir),
        "pulses": {pulse_id: dict(values) for pulse_id, values in pulse_metrics.items()},
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _format_value(value: object) -> str:
    """数值格式化为 4 位有效数字；其他类型原样 str（context 不做强 schema）。"""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _parameter_segment(symbol: str, pred: object | None, true: object | None, unit: str) -> str:
    """拼一段 ``symbol: pred <v> true <v> [unit]``（只列存在的项）。"""
    segment = symbol + ":"
    if pred is not None:
        segment += f" pred {_format_value(pred)}"
    if true is not None:
        segment += f" true {_format_value(true)}"
    if unit:
        segment += f" {unit}"
    return segment


def _build_title(pulse_id: str, context: Mapping[str, object] | None) -> str:
    """首行 pulse（+ model_kind）；次行按需列 α/Ku 的 pred/true（Ku 带 J/m³）。"""
    ctx = context if context is not None else {}
    model_kind = ctx.get("model_kind")
    first = f"Pulse {pulse_id}" + (f" — {model_kind}" if model_kind is not None else "")
    segments: list[str] = []
    alpha_pred, alpha_true = ctx.get("alpha_pred"), ctx.get("alpha_true")
    if alpha_pred is not None or alpha_true is not None:
        segments.append(_parameter_segment(r"$\alpha$", alpha_pred, alpha_true, ""))
    ku_pred, ku_true = ctx.get("ku_pred"), ctx.get("ku_true")
    if ku_pred is not None or ku_true is not None:
        segments.append(_parameter_segment(r"$K_u$", ku_pred, ku_true, "J/m³"))
    if not segments:
        return first
    return first + "\n" + "  |  ".join(segments)


def _fit_residual_ylim(ax: Axes, residual_1d: NDArray[np.float64]) -> None:
    """残差纵轴紧贴数据并始终包含零线（零线必须落在可视范围内）。"""
    lo = min(float(residual_1d.min()), 0.0)
    hi = max(float(residual_1d.max()), 0.0)
    pad = 0.08 * (hi - lo) or 1e-6  # 恒定残差时给一个极小可视窗口
    ax.set_ylim(lo - pad, hi + pad)


def plot_pulse_comparison(
    t_s: NDArray[np.float64],
    observed: NDArray[np.float64],
    forward: NDArray[np.float64],
    *,
    pulse_id: str,
    output_base: Path,
    context: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    """绘制单 pulse 三行两列对比图并同时写出 PNG 与 SVG。

    Args:
        t_s: ``[T]`` float64 秒；零点即关场时刻（已重锚定，原样使用，仅换算
            为 ns 显示）。
        observed / forward: ``[T, 3]`` float64，列序 mx, my, mz；调用方已
            校验对齐，本函数不重复校验，也不修改输入数组。
        pulse_id: 标题用标识（可含点号）。
        output_base: 输出基名；实际文件为 ``str(output_base) + ".png"`` 与
            ``str(output_base) + ".svg"``（追加后缀，保留基名中的点号）。
        context: 可选标题上下文；识别键 ``model_kind`` / ``alpha_pred`` /
            ``ku_pred`` / ``alpha_true`` / ``ku_true``（缺失即省略对应片段；
            真值只作标题标注，不参与任何计算），其余键忽略。

    Returns:
        ``(png_path, svg_path)``。

    Raises:
        FileExistsError: 任一输出文件已存在（拒绝覆盖）。
    """
    png_path = Path(f"{output_base}.png")
    svg_path = Path(f"{output_base}.svg")
    for target in (png_path, svg_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {target}")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    t_ns = t_s * 1e9  # 新数组；输入保持不动
    residual = forward - observed  # 新数组；残差定义 = forward - observed

    fig = Figure(figsize=(10.5, 8.0))
    FigureCanvasAgg(fig)  # 无 GUI 后端；不引入 pyplot 全局状态
    try:
        axes = fig.subplots(3, 2, sharex="col")
        legend_handles: list[Line2D] = []
        for row in range(3):
            ax_signal = axes[row, 0]
            ax_residual = axes[row, 1]
            (line_observed,) = ax_signal.plot(
                t_ns, observed[:, row], color=_COLOR_OBSERVED, lw=1.4, label="observed"
            )
            (line_forward,) = ax_signal.plot(
                t_ns, forward[:, row], color=_COLOR_FORWARD, lw=1.2, ls="--", label="forward"
            )
            (line_residual,) = ax_residual.plot(
                t_ns, residual[:, row], color=_COLOR_RESIDUAL, lw=1.0, label="forward - observed"
            )
            ax_residual.axhline(0.0, color=_COLOR_ZERO_LINE, lw=0.8, zorder=1)
            _fit_residual_ylim(ax_residual, residual[:, row])
            for ax in (ax_signal, ax_residual):
                ax.grid(True, which="major", lw=0.6, alpha=0.35)
                ax.tick_params(labelsize=9)
            ax_signal.margins(y=0.1)  # 分量各自自适应纵轴，小振荡可见
            ax_signal.set_ylabel(_COMPONENT_MATH[row], fontsize=10)
            ax_residual.set_ylabel(_RESIDUAL_MATH[row], fontsize=10)
            if row == 0:
                ax_signal.set_title("observed vs forward", fontsize=10.5)
                ax_residual.set_title("residual (forward - observed)", fontsize=10.5)
                legend_handles = [line_observed, line_forward, line_residual]
        axes[2, 0].set_xlabel("t (ns)", fontsize=10)
        axes[2, 1].set_xlabel("t (ns)", fontsize=10)
        # 固定边距（不用 constrained layout：它不会为 fig.legend 预留空间，
        # 底部图例会撞上 x 轴标签）。底部 0.13 内依次放 xlabel 与图例条。
        fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.13, wspace=0.16, hspace=0.25)
        fig.suptitle(_build_title(pulse_id, context), fontsize=12.5, y=0.985)
        fig.legend(
            legend_handles,
            [str(handle.get_label()) for handle in legend_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),  # 图例条在 xlabel 之下，不遮挡任何曲线/标签
            ncols=3,
            frameon=False,
            fontsize=10,
        )
        fig.savefig(png_path, dpi=200)
        fig.savefig(svg_path)
    finally:
        fig.clf()  # 显式释放 artist/canvas（无 pyplot 注册表，GC 兜底）
    return png_path, svg_path


def _require_matching_times(
    pulse_id: str, observed: NDArray[np.float64], forward: NDArray[np.float64]
) -> None:
    """两条时间轴须在极小 atol 内逐点一致（只容忍 float64 舍入；不重采样）。"""
    if observed.shape != forward.shape or not np.allclose(
        observed, forward, rtol=0.0, atol=TIME_ATOL_S
    ):
        raise ForwardValidationError(
            f"pulse {pulse_id!r} 的时间轴不一致（要求同一采样协议，不重采样）"
        )


def run_forward_validation(
    checkpoint_path: Path,
    sample_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> Path:
    """正向回代主编排：恢复/推理 → 写 forward 配置 → 模拟 → 比较/指标/绘图。

    Args:
        checkpoint_path: 单个 checkpoint（结构/契约/预处理全部来自它）。
        sample_dir: 单个原始参数组目录（config.yaml/index.csv/trajectory.csv）。
        output_dir: 新报告输出目录；已存在（含符号链接）拒绝覆盖；失败保留现场。
        dry_run: 只写 prediction.json 与 forward_config.yaml，不运行模拟。

    Returns:
        ``output_dir``（模拟原始数据位于 pipeline 返回的真实参数组目录，
        写入 metrics.json 的 ``simulation_dir`` 并打印）。
    """
    checkpoint_path = Path(checkpoint_path)
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")

    ckpt = training.load_any_checkpoint(checkpoint_path)
    model = _restore_model(ckpt)
    sample = read_sample(sample_dir, ckpt.contract)
    prediction = predict_parameter_set(model, ckpt, sample.x)

    output_dir.mkdir(parents=True)
    write_prediction_json(
        output_dir / "prediction.json", labels=sample.labels, prediction=prediction
    )
    forward_config_path = output_dir / "forward_config.yaml"
    write_forward_config(
        sample_dir / "config.yaml",
        forward_config_path,
        alpha=float(prediction[0]),
        ku_j_per_m3=float(prediction[1]),
    )
    if dry_run:
        print(f"dry-run 完成（未运行模拟）: {output_dir}")
        return output_dir

    simulation_dir = run_forward_simulation(forward_config_path)
    print(f"  正向模拟目录: {simulation_dir}")
    forward_times, forward_trajectories = _read_float64_trajectories(
        simulation_dir, ckpt.contract.pulse_order
    )

    context: dict[str, object] = {
        "model_kind": ckpt.config.model.kind,
        "alpha_pred": float(prediction[0]),
        "ku_pred": float(prediction[1]),
        "alpha_true": float(sample.labels[0]),
        "ku_true": float(sample.labels[1]),
    }
    pulse_metrics: dict[str, dict[str, float]] = {}
    for pulse_id in ckpt.contract.pulse_order:
        times = sample.times[pulse_id]
        observed = sample.trajectories[pulse_id]
        forward = forward_trajectories[pulse_id]
        _require_matching_times(pulse_id, times, forward_times[pulse_id])
        write_comparison_csv(
            output_dir / "comparisons" / f"{pulse_id}.csv", times, observed, forward
        )
        pulse_metrics[pulse_id] = trajectory_metrics(observed, forward)
        plot_pulse_comparison(
            times,
            observed,
            forward,
            pulse_id=pulse_id,
            output_base=output_dir / "figures" / pulse_id,
            context=context,
        )
    write_metrics_json(output_dir / "metrics.json", pulse_metrics, simulation_dir=simulation_dir)
    print(f"正向回代完成: {output_dir}")
    return output_dir
