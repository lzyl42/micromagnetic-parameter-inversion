"""评估：run/ckpt 定位的常规 evaluate 与物理单位指标。

对应 ``train.md`` 第 8 节。定位：``--run <run_dir> [--checkpoint]``；
网络结构、预处理与 label 变换**全部从 checkpoint 恢复**（不经当前 YAML）。
绑定校验（任一不符即 EvaluationError）：

- split：run 内保存的 ``split.yaml`` 副本（唯一权威），sha256 与
  ``ckpt.split_sha256`` 核对；副本经 ``training_data.load_split(…,
  split_path=…)`` 同一加载边界校验（互斥/并集/npz 存在性针对真实样本
  目录）——samples 目录下的 split.yaml 即使被重切也不影响评估；
- dataset_meta：锚点 ``data_root()/samples/<ckpt.config.dataset_name>/``
  下的 ``ckpt.dataset_meta_relpath``（resolve 后强制留在锚点内，防逃逸），
  sha256 与 ``ckpt.dataset_meta_sha256`` 核对，dataset_name 与 ckpt 配置
  交叉核对；psid→Ku 表据此划分主域/control。
- provenance：``EvaluationReport.provenance``（``EvaluationProvenance``）
  记录**实际加载**的 ckpt——路径（``resolve_checkpoint_path`` 的结果，
  best/final/外部路径原样字符串）、文件字节 sha256（一次读取，非二次
  ``torch.load``）与来自该 ckpt 的 ``split_sha256``（已与 run 内副本核对
  一致）。``test_metrics.json`` 的 ``provenance`` 键据此生成：JSON 来源
  一定对应实际加载的模型，而非仅默认路径；跨 run 相同 split 合法，仅
  如实记录来源，不引入新限制。

指标（物理单位，alpha/Ku 分别报告）：首版仅 MAE/RMSE；主域排除 Ku = 0
的 control，control 单独报告、不计入主域；每个子集输出样本数 ``n``，
空子集（n=0）指标为 ``None``（JSON null，非 NaN/0）；不重切凑指标。
预测/输入含非有限值 → 明确拒绝（PreprocessingError/EvaluationError），
不当作空子集。

``run_evaluation`` 为纯计算（不写盘），返回 ``(EvaluationReport, rows)``；
test_metrics.json / test_predictions.csv 的写出由 ``evaluate_model.py``
编排（经 ``export_test_predictions``，拒绝覆盖、LF 行尾、每 psid 一行）。

依赖方向：经 ``training.load_checkpoint``/``build_model`` 读 ckpt 与重建
模型（training 禁止反向导入本模块）；split 副本经
``training_data.load_split(…, split_path=…)``、dataset_meta 经
``training_data.load_dataset_meta(…, meta_path=…)`` 走同一加载边界（本
模块只做 SHA/逃逸/dataset_name 核对）；metrics 计算在本模块。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from micromagnetic_parameter_inversion import paths, preprocessing, training, training_data


class EvaluationError(ValueError):
    """评估契约违反（split/meta SHA 不一致、meta 路径逃逸、npz 输入契约
    错配、指标输入非有限等）。"""


@dataclass(frozen=True)
class SubsetMetrics:
    """单个指标子集（主域或 control）的结果。

    空子集（n=0）：``n`` 如实输出 0，四个指标均为 ``None``（序列化为
    JSON null，不允许 NaN 或 0 充数）。
    """

    n: int  # 子集样本数（总是输出）
    mae_alpha: float | None  # 物理单位 MAE（alpha）
    rmse_alpha: float | None  # 物理单位 RMSE（alpha）
    mae_ku: float | None  # 物理单位 MAE（Ku, J/m^3）
    rmse_ku: float | None  # 物理单位 RMSE（Ku, J/m^3）


@dataclass(frozen=True)
class EvaluationProvenance:
    """评估来源留档（test_metrics.json 的 ``provenance`` 键来源）。

    全部字段描述**实际加载**的 checkpoint，而非默认路径假设：

    - ``checkpoint_path``：``resolve_checkpoint_path`` 的结果原样字符串
      （best/final/外部路径；未 resolve，即实际打开的路径）；
    - ``checkpoint_sha256``：该 ckpt 文件字节的 sha256（一次读取；与
      ``torch.load`` 反序列化相互独立，不构成二次加载）；
    - ``split_sha256``：来自实际 ckpt 的 ``split_sha256``（已与 run 内
      split 副本核对一致后才生成 provenance）。
    """

    checkpoint_path: str
    checkpoint_sha256: str
    split_sha256: str


@dataclass(frozen=True)
class EvaluationReport:
    """test 集评估报告（test_metrics.json 的来源）。"""

    main: SubsetMetrics  # 主域：排除 Ku = 0 的 control
    control: SubsetMetrics  # Ku = 0 物理 control，单独报告、不计入主域
    provenance: EvaluationProvenance  # 实际加载 ckpt 的来源留档


@dataclass(frozen=True)
class PredictionRow:
    """test_predictions.csv 的一行（每参数组合一行，无 pulse_id 列）。"""

    parameter_set_id: str
    split: str  # 常规 evaluate 仅评估 run 内 split 副本的 test
    alpha_true: float
    alpha_pred: float
    ku_true: float
    ku_pred: float


def compute_subset_metrics(
    y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]
) -> SubsetMetrics:
    """物理单位 MAE/RMSE（alpha/Ku 分别报告，不跨列合并、不重切子集）。

    Args:
        y_true/y_pred: ``[n, 2]`` 序列（物理单位，列序 alpha/ku）。

    Returns:
        ``n == 0`` → ``n=0`` 且四个指标为 ``None``（JSON null，非 NaN 非 0）。

    Raises:
        EvaluationError: 两序列长度不一致、形状非 ``[n,2]``，或含非有限值
            （NaN/Inf 明确拒绝，不当作空子集）。
    """
    true_list = list(y_true)
    pred_list = list(y_pred)
    if len(true_list) != len(pred_list):
        raise EvaluationError(f"y_true/y_pred 长度不一致: {len(true_list)} vs {len(pred_list)}")
    if not true_list:
        return SubsetMetrics(n=0, mae_alpha=None, rmse_alpha=None, mae_ku=None, rmse_ku=None)
    true = np.asarray(true_list, dtype=np.float64)
    pred = np.asarray(pred_list, dtype=np.float64)
    if true.ndim != 2 or true.shape != pred.shape or true.shape[1] != 2:
        raise EvaluationError(
            f"y_true/y_pred 形状须为 [n,2] 且一致 (got {true.shape} vs {pred.shape})"
        )
    if not (np.all(np.isfinite(true)) and np.all(np.isfinite(pred))):
        raise EvaluationError("指标输入含非有限值 (NaN/Inf)；明确拒绝，不当作空子集")
    diff = pred - true
    mae = np.abs(diff).mean(axis=0)
    rmse = np.sqrt((diff**2).mean(axis=0))
    return SubsetMetrics(
        n=int(true.shape[0]),
        mae_alpha=float(mae[0]),
        rmse_alpha=float(rmse[0]),
        mae_ku=float(mae[1]),
        rmse_ku=float(rmse[1]),
    )


def resolve_checkpoint_path(run_dir: Path, checkpoint_path: Path | None = None) -> Path:
    """定位 run 的 checkpoint：缺省 ``<run_dir>/best.pt``（纯路径逻辑）。"""
    return checkpoint_path if checkpoint_path is not None else run_dir / "best.pt"


def run_evaluation(
    run_dir: Path,
    checkpoint_path: Path | None = None,
) -> tuple[EvaluationReport, tuple[PredictionRow, ...]]:
    """常规 evaluate 主流程：返回 ``(EvaluationReport, rows)``（纯计算，不写盘）。

    Args:
        run_dir: 训练 run 目录（split 副本 / ckpt 的定位来源）。
        checkpoint_path: 缺省 ``<run_dir>/best.pt``；无论显式与否，ckpt 的
            split SHA 必须与 run 内副本一致。

    编排：

    1. ``ckpt = training.load_checkpoint(...)``；结构/预处理/label 全部
       来自 ckpt（不经当前 YAML）；
    2. run 内 split 副本 sha256 与 ``ckpt.split_sha256`` 核对；随后生成
       ``EvaluationProvenance``（实际 ckpt 路径、文件字节 sha256 一次
       读取、来自该 ckpt 的 split_sha256）；
    3. dataset_meta 按锚点 + relpath 定位（防逃逸）+ SHA 核对 + dataset_name
       交叉核对；
    4. ``training_data.load_split(samples_dir, meta, split_path=副本)``：
       同一加载边界校验（npz 存在性针对真实样本目录）；
    5. test 成员按 ``ckpt.config.training.batch_size`` 分批：CPU、eval、
       ``no_grad`` 前向，``load_sample`` 契约校验（x 形状/pulse_ids 顺序/
       t_s），``inverse_transform_y`` 还原物理单位；
    6. 按 meta psid→Ku 表划分主域/control（Ku = 0 → control），各自
       ``compute_subset_metrics``；
    7. 组装 ``PredictionRow`` 序列（写盘由 ``evaluate_model.py`` 编排）。

    Raises:
        EvaluationError: split/meta 绑定不一致、路径逃逸、指标输入非法。
        DataError/PreprocessingError/TrainingError: npz 契约、逆变换非有限、
            ckpt 加载失败（原样上抛，调用方统一呈现）。
    """
    run_dir = Path(run_dir)
    ckpt_path = resolve_checkpoint_path(run_dir, checkpoint_path)
    ckpt = training.load_checkpoint(ckpt_path)

    samples_dir = paths.data_root() / "samples" / ckpt.config.dataset_name
    split_copy = run_dir / "split.yaml"
    if not split_copy.is_file():
        raise EvaluationError(f"run 内缺少 split.yaml 副本: {split_copy}")
    if training_data.sha256_file(split_copy) != ckpt.split_sha256:
        raise EvaluationError(
            f"run 内 split 副本与 ckpt.split_sha256 不一致（run 绑定被破坏）: {split_copy}"
        )
    provenance = EvaluationProvenance(
        checkpoint_path=str(ckpt_path),
        checkpoint_sha256=training_data.sha256_file(ckpt_path),
        split_sha256=ckpt.split_sha256,
    )

    meta_path = _resolve_meta_path(ckpt)
    meta = training_data.load_dataset_meta(samples_dir, meta_path=meta_path)
    if meta.dataset_name != ckpt.config.dataset_name:
        raise EvaluationError(
            f"dataset_meta.dataset_name {meta.dataset_name!r} 与 ckpt 配置 "
            f"{ckpt.config.dataset_name!r} 不一致 ({meta_path})"
        )
    split = training_data.load_split(samples_dir, meta, split_path=split_copy)

    rows = _evaluate_test_rows(ckpt, samples_dir, meta, split)
    report = _build_report(rows, meta, provenance)
    return report, tuple(rows)


def _resolve_meta_path(ckpt: training.Checkpoint) -> Path:
    """dataset_meta 锚点定位：resolve 后强制留在锚点内（防逃逸）+ SHA 核对。

    锚点由 ``data_root()/samples/<ckpt.config.dataset_name>/`` 推导（与
    ``run_evaluation`` 的 ``samples_dir`` 同源），故无需额外参数。
    """
    anchor = (paths.data_root() / "samples" / ckpt.config.dataset_name).resolve()
    candidate = (anchor / ckpt.dataset_meta_relpath).resolve()
    try:
        candidate.relative_to(anchor)
    except ValueError:
        raise EvaluationError(
            f"dataset_meta_relpath 逃逸出锚点目录 {anchor}: {ckpt.dataset_meta_relpath!r}"
        ) from None
    if not candidate.is_file():
        raise EvaluationError(f"dataset_meta 不存在: {candidate}")
    if training_data.sha256_file(candidate) != ckpt.dataset_meta_sha256:
        raise EvaluationError(f"dataset_meta SHA 与 ckpt 不一致: {candidate}")
    return candidate


# TODO(CNN1D-P3): 未来共享 evaluate 入口先明确 ckpt 格式再显式路由（MLP 走
# 现有路径；CNN 走 load_cnn_checkpoint / CNNCheckpoint 恢复），不以类别猜测；
# 完整结构从 ckpt 恢复、不被嵌套 config 覆盖；CNN 损坏不得回退 MLP。
# 旧 MLP 评估路径与指标口径完全保留。本轮不改函数体与调用。
def _evaluate_test_rows(
    ckpt: training.Checkpoint,
    samples_dir: Path,
    meta: training_data.DatasetMeta,
    split: training_data.SplitDefinition,
) -> list[PredictionRow]:
    """test 成员分批推理（CPU/eval/no_grad）→ 物理单位 PredictionRow 列表。"""
    model = training.build_model(ckpt.contract, ckpt.hidden_dims)
    model.load_state_dict(dict(ckpt.model_state_dict))
    model.to("cpu")
    model.eval()
    batch_size = int(ckpt.config.training.batch_size)
    if batch_size < 1:
        raise EvaluationError(f"ckpt.config.training.batch_size 须为正 (got {batch_size})")

    rows: list[PredictionRow] = []
    for start in range(0, len(split.test), batch_size):
        chunk = split.test[start : start + batch_size]
        samples = [
            training_data.load_sample(samples_dir, psid, ckpt.contract)  # 契约校验
            for psid in chunk
        ]
        x_raw = np.stack([sample.x for sample in samples])  # [n,P,T,3] float32
        x_norm = preprocessing.transform_x(ckpt.preprocessing, x_raw)
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(x_norm))  # [n,2] 标准化空间
        y_physical = preprocessing.inverse_transform_y(ckpt.preprocessing, pred_norm.numpy())
        for sample, pred in zip(samples, y_physical, strict=True):
            rows.append(
                PredictionRow(
                    parameter_set_id=sample.parameter_set_id,
                    split="test",
                    alpha_true=float(sample.y[0]),
                    alpha_pred=float(pred[0]),
                    ku_true=float(sample.y[1]),
                    ku_pred=float(pred[1]),
                )
            )
    return rows


def _build_report(
    rows: Sequence[PredictionRow],
    meta: training_data.DatasetMeta,
    provenance: EvaluationProvenance,
) -> EvaluationReport:
    """按 meta psid→Ku 表划分主域/control（Ku = 0 → control），分别计算指标。"""
    main_rows: list[list[float]] = []
    main_pred: list[list[float]] = []
    control_rows: list[list[float]] = []
    control_pred: list[list[float]] = []
    for row in rows:
        if row.parameter_set_id not in meta.labels:
            raise EvaluationError(f"psid 不在 dataset_meta psid→Ku 表中: {row.parameter_set_id!r}")
        ku_true_from_meta = meta.labels[row.parameter_set_id][1]
        if ku_true_from_meta == 0.0:
            control_rows.append([row.alpha_true, row.ku_true])
            control_pred.append([row.alpha_pred, row.ku_pred])
        else:
            main_rows.append([row.alpha_true, row.ku_true])
            main_pred.append([row.alpha_pred, row.ku_pred])
    return EvaluationReport(
        main=compute_subset_metrics(main_rows, main_pred),
        control=compute_subset_metrics(control_rows, control_pred),
        provenance=provenance,
    )


def export_test_predictions(path: Path, rows: Sequence[PredictionRow]) -> None:
    """写出 test_predictions.csv（LF 行尾、每 psid 一行；已存在 → FileExistsError）。

    列序固定：``parameter_set_id, split, alpha_true, alpha_pred, ku_true,
    ku_pred``；浮点以 repr 全精度写入。
    """
    if path.exists():
        raise FileExistsError(f"评估导出已存在，拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "parameter_set_id,split,alpha_true,alpha_pred,ku_true,ku_pred"
    lines = [header]
    lines += [
        f"{row.parameter_set_id},{row.split},{row.alpha_true!r},{row.alpha_pred!r},"
        f"{row.ku_true!r},{row.ku_pred!r}"
        for row in rows
    ]
    # 显式 "\n" 连接 + newline="\n"：跨平台保证 LF 行尾
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
