"""Forward re-simulation validation: predict (alpha, Ku) → simulate the same pulses
→ strict comparison against observations.

Flow (lightweight closed loop; inputs are legal script products and are trusted):

1. ``training.load_any_checkpoint`` reads the checkpoint and this module restores the
   model inline (CPU, eval, strict weight loading);
2. ``read_sample``: ``training_data.read_parameter_group`` reads float32 x per the ckpt
   contract (used for inference; the index ground truth is used only for parameter
   metrics) and reads float64 trajectories locally via the index's trajectory_path;
   each pulse's time axis is compared against ``contract.t_s`` with a tiny atol (a
   same-shape trajectory with a different sampling protocol must not be fed to the
   model);
3. ``predict_parameter_set`` inference; the original config is written out as
   ``forward_config.yaml`` with only material.alpha / material.ku_j_per_m3 /
   dataset_name replaced (``<source name>_forward_<uuid short code>``), then
   validated through the existing ``mumax3_config.load_config``;
4. dry-run stops here; otherwise ``mumax3_pipeline.run_parameter_set(config_path)``
   re-runs Relax and the per-pulse simulations into the default layout and returns
   the real parameter set directory; the independent dataset
   (``data_root()/raw/<forward dataset_name>/<parameter_set_id>/``) never overwrites
   observed data, and the main flow reads that directory back;
5. per pulse, ``comparisons/<pulse_id>.csv``, ``metrics.json`` (with the actual
   simulation directory) and ``figures/<pulse_id>.png|svg`` are written under
   ``--output``; a time-axis mismatch is an error and no resampling is performed.

``<output_dir>/``: prediction.json (ground truth/prediction/absolute parameter errors),
forward_config.yaml, metrics.json, comparisons/<pulse_id>.csv,
figures/<pulse_id>.png|svg. A failure keeps the scene as-is; nothing is cleaned up
automatically.

Plots (three rows, two columns): rows = mx/my/mz; left column = observed (solid)
overlaid with forward (dashed); right column = residual ``forward - observed`` with a
zero line; the x-axis is ``t_s × 1e9`` (ns) with zero at the field switch-off
instant (the time axis has already been re-anchored by the pipeline; no phase shift,
no cropping, no normalization, no smoothing, and input arrays are not modified).
Each component gets its own adaptive y-axis so small oscillations are not flattened
by the full |m| ≤ 1 range; the legend is figure-level, placed outside the plotting
area and does not cover curves; all text is English (no dependency on Chinese
fonts). The backend uses ``matplotlib.figure.Figure`` + ``FigureCanvasAgg`` directly
(no pyplot, no GUI, no global state, no rcParams mutation); outputs are
``str(output_base) + ".png" / ".svg"``; if either target already exists it is refused
with ``FileExistsError``, parent directories are created automatically, and a failed
figure write keeps the scene as-is.
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

# Time-axis comparison tolerance: both axes come from the pipeline's i*dt
# re-anchoring and only absorb float64 rounding; far below any sampling interval,
# so the tolerance must never mask resampling/phase shift.
TIME_ATOL_S: Final[float] = 1e-24

_COMPONENT_MATH = (r"$m_x$", r"$m_y$", r"$m_z$")
_RESIDUAL_MATH = (r"$\Delta m_x$", r"$\Delta m_y$", r"$\Delta m_z$")

_COLOR_OBSERVED = "#222222"  # near-black solid line: observed (raw CSV data)
_COLOR_FORWARD = "#0072B2"  # Okabe-Ito blue dashed line: forward re-simulation
_COLOR_RESIDUAL = "#D55E00"  # Okabe-Ito vermilion: residual
_COLOR_ZERO_LINE = "#888888"


class ForwardValidationError(ValueError):
    """Forward-validation flow contract violation.

    Covers time-axis mismatches and weight/structure mismatches.
    """


@dataclass(frozen=True)
class SampleData:
    """Observed data read for one parameter group.

    float32 is used for inference; raw float64 for comparison/plots.
    """

    labels: tuple[float, float]  # raw index.csv (alpha, ku_j_per_m3); parameter metrics only
    x: NDArray[np.float32]  # [P, T, 3], in ckpt contract pulse order
    times: dict[str, NDArray[np.float64]]  # pulse_id -> [T]
    trajectories: dict[str, NDArray[np.float64]]  # pulse_id -> [T, 3], raw CSV values


def read_sample(sample_dir: Path, contract: training_data.InputContract) -> SampleData:
    """Read an observed parameter group: float32 x (inference) + float64 trajectories.

    Reuses ``training_data.read_parameter_group``: the index pulse set and trajectory
    shapes are validated against the ckpt contract (``contract.pulse_order`` /
    ``n_time_steps``) and x is float32, consistent with training/evaluation reads;
    float64 trajectories are read locally via the index's trajectory_path, and each
    pulse's time axis is additionally compared against ``contract.t_s`` with a tiny
    atol (a same-shape trajectory with a different sampling protocol must not be fed
    to the model).
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
    """Read each pulse's float64 time column and trajectory from the index trajectory_path.

    The data is a script product and is read only via known relative paths; the raw
    float64 values feed comparison/plots and are independent of the float32 inference
    input (inference intermediates are never treated as raw trajectories).
    """
    with (group_dir / "index.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    path_by_pulse = {row["pulse_id"].strip(): row["trajectory_path"].strip() for row in rows}
    times: dict[str, NDArray[np.float64]] = {}
    trajectories: dict[str, NDArray[np.float64]] = {}
    for pulse_id in pulse_order:
        if pulse_id not in path_by_pulse:
            raise ForwardValidationError(f"pulse {pulse_id!r} has no trajectory: {group_dir}")
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
    """Restore the model from the parsed checkpoint (CPU, eval, strict weights; no re-read)."""
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
        raise ForwardValidationError(f"unsupported checkpoint type: {type(ckpt).__name__}")
    try:
        model.load_state_dict(dict(ckpt.model_state_dict), strict=True)
    except RuntimeError as exc:
        raise ForwardValidationError(
            f"checkpoint weights do not match the restored model structure (keys/sizes): {exc}"
        ) from exc
    model.to("cpu")
    model.eval()
    return model


def predict_parameter_set(
    model: nn.Module,
    ckpt: training.ModelCheckpoint,
    x: NDArray[np.float32] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Single-group inference: ``transform_x`` → model → ``inverse_transform_y``.

    Quantize float32 explicitly first (consistent with training/evaluation reads; raw
    float64 trajectories never enter the model); ground truth does not participate;
    returns physical units (alpha, ku_j_per_m3) as float64 ``[2]``.
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
    """Replace only alpha/Ku/dataset_name in the original config; write forward_config.yaml.

    ``dataset_name`` becomes ``<source name>_forward_<uuid short code>`` (8 hex
    digits) so forward simulations of multiple models/repeated predictions never
    collide; all other fields keep their original values. After writing, the file is
    validated through the existing ``mumax3_config.load_config`` (illegal values such
    as alpha<0 raise ``ConfigError`` directly, with no clipping); an existing target
    is refused.
    """
    if forward_config_path.exists():
        raise FileExistsError(
            f"forward config already exists, refusing to overwrite: {forward_config_path}"
        )
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
    """Re-run Relax and per-pulse simulations via the pipeline; return the real group dir.

    Output follows the pipeline's default layout
    ``data_root()/raw/<dataset_name>/<psid>/`` (an independent ``_forward_<uuid>``
    dataset that never overwrites observed data); external simulation goes only
    through ``mumax3_pipeline`` → ``external``, never a direct subprocess.
    """
    return mumax3_pipeline.run_parameter_set(forward_config_path)


def trajectory_metrics(
    observed: NDArray[np.float64], forward: NDArray[np.float64]
) -> dict[str, float]:
    """Per-pulse trajectory metrics: per-component RMSE, vector RMSE, max vector error.

    All values are float64.
    """
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
    """Write ``prediction.json``: ground truth, prediction, absolute parameter errors.

    Refuses to overwrite. The ground truth is used only for the parameter error
    metrics in this file and never enters inference.
    """
    if path.exists():
        raise FileExistsError(f"prediction.json already exists, refusing to overwrite: {path}")
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
    """Write the ``sample_index,t_s_s,observed_*,forward_*,residual_*`` comparison table.

    LF newlines only; an existing file is refused.
    """
    if path.exists():
        raise FileExistsError(f"comparison table already exists, refusing to overwrite: {path}")
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
    """Write ``metrics.json``: actual simulation directory + per-pulse metrics (no aggregate)."""
    if path.exists():
        raise FileExistsError(f"metrics.json already exists, refusing to overwrite: {path}")
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
    """Format numbers to 4 significant digits; other types via str.

    The context is not schema-checked.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _parameter_segment(symbol: str, pred: object | None, true: object | None, unit: str) -> str:
    """Build one ``symbol: pred <v> true <v> [unit]`` segment.

    Only entries that are present are listed.
    """
    segment = symbol + ":"
    if pred is not None:
        segment += f" pred {_format_value(pred)}"
    if true is not None:
        segment += f" true {_format_value(true)}"
    if unit:
        segment += f" {unit}"
    return segment


def _build_title(pulse_id: str, context: Mapping[str, object] | None) -> str:
    """First line: pulse (plus model_kind); second line: alpha/Ku pred/true on demand.

    Ku carries the J/m³ unit.
    """
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
    """Residual y-axis hugs the data and always includes the zero line (must stay visible)."""
    lo = min(float(residual_1d.min()), 0.0)
    hi = max(float(residual_1d.max()), 0.0)
    pad = 0.08 * (hi - lo) or 1e-6  # tiny visible window when the residual is constant
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
    """Plot one pulse as a three-row two-column comparison and write both PNG and SVG.

    Args:
        t_s: ``[T]`` float64 seconds; zero is the field switch-off instant (already
            re-anchored, used as-is, only converted to ns for display).
        observed / forward: ``[T, 3]`` float64, column order mx, my, mz; the caller
            has already validated alignment, so this function does not re-check and
            does not modify the input arrays.
        pulse_id: identifier used in the title (may contain dots).
        output_base: output base name; the actual files are
            ``str(output_base) + ".png"`` and ``str(output_base) + ".svg"`` (suffix
            appended, dots in the base name preserved).
        context: optional title context; recognized keys ``model_kind`` /
            ``alpha_pred`` / ``ku_pred`` / ``alpha_true`` / ``ku_true`` (a missing
            key omits that segment; ground truth is a title annotation only and
            participates in no computation), all other keys ignored.

    Returns:
        ``(png_path, svg_path)``.

    Raises:
        FileExistsError: either output file already exists (overwrite refused).
    """
    png_path = Path(f"{output_base}.png")
    svg_path = Path(f"{output_base}.svg")
    for target in (png_path, svg_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {target}")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    t_ns = t_s * 1e9  # new array; the input stays untouched
    residual = forward - observed  # new array; residual definition = forward - observed

    fig = Figure(figsize=(10.5, 8.0))
    FigureCanvasAgg(fig)  # no-GUI backend; no pyplot global state
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
            ax_signal.margins(y=0.1)  # per-component adaptive y-axis keeps oscillations visible
            ax_signal.set_ylabel(_COMPONENT_MATH[row], fontsize=10)
            ax_residual.set_ylabel(_RESIDUAL_MATH[row], fontsize=10)
            if row == 0:
                ax_signal.set_title("observed vs forward", fontsize=10.5)
                ax_residual.set_title("residual (forward - observed)", fontsize=10.5)
                legend_handles = [line_observed, line_forward, line_residual]
        axes[2, 0].set_xlabel("t (ns)", fontsize=10)
        axes[2, 1].set_xlabel("t (ns)", fontsize=10)
        # Fixed margins (not constrained layout: it does not reserve space for
        # fig.legend, so a bottom legend would collide with the x-axis labels).
        # The bottom 0.13 holds the xlabel and the legend strip in order.
        fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.13, wspace=0.16, hspace=0.25)
        fig.suptitle(_build_title(pulse_id, context), fontsize=12.5, y=0.985)
        fig.legend(
            legend_handles,
            [str(handle.get_label()) for handle in legend_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),  # legend strip below the xlabel, covers nothing
            ncols=3,
            frameon=False,
            fontsize=10,
        )
        fig.savefig(png_path, dpi=200)
        fig.savefig(svg_path)
    finally:
        fig.clf()  # explicitly release artists/canvas (no pyplot registry; GC fallback)
    return png_path, svg_path


def _require_matching_times(
    pulse_id: str, observed: NDArray[np.float64], forward: NDArray[np.float64]
) -> None:
    """Both time axes must match pointwise within a tiny atol.

    Only float64 rounding is tolerated; no resampling.
    """
    if observed.shape != forward.shape or not np.allclose(
        observed, forward, rtol=0.0, atol=TIME_ATOL_S
    ):
        raise ForwardValidationError(
            f"pulse {pulse_id!r} has a mismatched time axis "
            f"(the same sampling protocol is required; no resampling)"
        )


def run_forward_validation(
    checkpoint_path: Path,
    sample_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> Path:
    """Main forward-validation orchestration: restore/infer → write config → simulate.

    Args:
        checkpoint_path: a single checkpoint (structure/contract/preprocessing all
            come from it).
        sample_dir: a single raw parameter group directory
            (config.yaml/index.csv/trajectory.csv).
        output_dir: new report output directory; an existing one (including
            symlinks) is refused; a failure keeps the scene as-is.
        dry_run: write only prediction.json and forward_config.yaml; do not run the
            simulation.

    Returns:
        ``output_dir`` (the raw simulation data lives in the real parameter set
        directory returned by the pipeline, is written to metrics.json as
        ``simulation_dir`` and printed).
    """
    checkpoint_path = Path(checkpoint_path)
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"output directory already exists, refusing to overwrite: {output_dir}"
        )

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
        print(f"dry-run complete (simulation not run): {output_dir}")
        return output_dir

    simulation_dir = run_forward_simulation(forward_config_path)
    print(f"  forward simulation directory: {simulation_dir}")
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
    print(f"forward validation complete: {output_dir}")
    return output_dir
