"""训练循环与 checkpoint 读写。

对应 ``train.md`` 第 5/8 节。职责边界：

- seed 入口：``train_model`` 在**模型构造之前**调用 ``set_seed``（训练 seed
  唯一归 train_model 管）；DataLoader shuffle 使用显式 seeded
  ``torch.Generator``（``make_data_generator``）。
- 训练：Adam + 标准化标签空间 MSE（按样本数加权聚合）；batch 在 CPU 上
  经 preprocessing 变换为 float32 numpy 再送 device；``num_workers=0``；
  非有限 loss/grad/pred 立即停止且不保存坏权重（best/final 只含完成
  epoch 的权重）；early stopping 使用独立 reference（仅当改善 >
  ``min_delta`` 才更新），与绝对 best（严格更低即更新）分开。
- checkpoint：``save_checkpoint`` 落盘字典仅含 Python primitives/list/dict
  + CPU Tensor（无 dataclass/numpy 对象），同名文件拒绝覆盖；
  ``load_checkpoint`` 以 ``torch.load(weights_only=True, map_location="cpu")``
  显式安全读取，校验格式版本与关键字段/契约形状后重建 dataclass（config
  嵌套重建不经临时文件）。
- 数据流：``train_model`` 返回 ``TrainingResult``（best/final checkpoint +
  逐 epoch history）；best.pt/final.pt/metrics.json 的磁盘写出由
  ``train_mlp.py`` 编排，checkpoint 读写仅在本模块。

依赖方向：training_config / training_data / preprocessing / runtime →
（本模块）→ evaluation；本模块禁止导入 evaluation（评估侧反向调用
``load_checkpoint`` / ``build_model``）。
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from micromagnetic_parameter_inversion import preprocessing, runtime, training_config
from micromagnetic_parameter_inversion.models.mlp import MLPRegressor
from micromagnetic_parameter_inversion.preprocessing import (
    PreprocessingError,
    PreprocessingState,
    transform_x,
    transform_y,
)
from micromagnetic_parameter_inversion.training_config import (
    CKPT_FORMAT_VERSION,
    ActivationName,
    ConfigError,
    ExperimentConfig,
)
from micromagnetic_parameter_inversion.training_data import (
    InputContract,
    SampleItem,
    TrajectoryDataset,
)

# ckpt_format_version 取 training_config.CKPT_FORMAT_VERSION（train.md 第 8 节）。
type StateDict = Mapping[str, Tensor]

# 停止原因：正常上限 / 早停 / 数值失败（ckpt 格式版本不变；停止元数据仅
# 存于 TrainingResult 与 metrics.json，不进 ckpt）。
type StopReason = Literal["max_epochs", "early_stopping", "numerical_failure"]

_ACTIVATIONS = ("relu",)  # 首版固定 ReLU（Checkpoint.activation 显式留档）
_N_OUTPUTS = 2  # 标准化标签列数 (alpha, ku_j_per_m3)


class TrainingError(RuntimeError):
    """训练/ckpt 契约违反（非有限训练状态、格式版本不符、字段/形状损坏）。

    ``epoch``/``detail``：数值失败时由 ``train_model`` 携带触发 epoch
    （1-based）与具体原因，供 CLI 写失败状态 metrics；其他违反可缺省。
    """

    def __init__(
        self, message: str, *, epoch: int | None = None, detail: str | None = None
    ) -> None:
        super().__init__(message)
        self.epoch = epoch
        self.detail = detail


@dataclass(frozen=True)
class EpochMetrics:
    """单个 epoch 的标准化 MSE 记录（metrics.json「逐 epoch train/val MSE」）。

    ``train_loss`` 是该 epoch 内**在线累计**的 batch 损失按样本数加权平均
    （训练进行到 epoch 结束时的累计值）；``val_loss`` 为整体验证集加权
    MSE。二者均处于标准化标签空间。
    """

    epoch: int  # 从 1 计
    train_loss: float  # 在线 batch 加权平均标准化 MSE
    val_loss: float  # val 集加权平均标准化 MSE（best 与 early stopping 判据）


@dataclass(frozen=True, eq=False)
class Checkpoint:
    """best.pt / final.pt 的 schema（仅凭 ckpt + npz 即可独立推理）。

    含嵌套 ``InputContract``（带 ndarray 字段），``eq=False`` 避免逐元素
    比较歧义。evaluate 侧用途：``hidden_dims``/``activation``/``contract``
    恢复模型结构与输入契约（不经当前 YAML）；``preprocessing`` 恢复 x/y
    变换（不重新拟合）；``split_sha256`` 与 run 内保存的 split 副本绑定
    校验；``dataset_meta_relpath``/``dataset_meta_sha256`` 以
    ``data_root()/samples/<dataset_name>/`` 为锚定位并校验 dataset_meta。
    ``config`` 仅记录，不作为恢复来源。
    """

    ckpt_format_version: int  # 写出时的 schema 版本
    model_state_dict: StateDict  # 模型权重（磁盘形态为 CPU Tensor 字典）
    hidden_dims: tuple[int, ...]  # 模型结构显式字段（不只藏在 config 副本里）
    activation: ActivationName  # 激活函数显式留档（首版固定 "relu"）
    contract: InputContract  # 输入契约：pulse 顺序、T、分量序、t_s
    preprocessing: (
        PreprocessingState  # 预处理状态：x [P,1,3] mean、effective scale、label、y 统计量
    )
    seed: int  # training.seed
    config: ExperimentConfig  # 生效配置副本（仅记录）
    dataset_meta_relpath: str  # 相对锚点 data_root()/samples/<dataset>/ 的路径
    dataset_meta_sha256: str  # dataset_meta 内容指纹（轻量溯源）
    split_sha256: str  # run 内 split 副本 sha256（split 绑定）
    best_val_loss: float | None  # best.pt 必有；final.pt 可为 None
    git_sha: str | None = None  # 可得时记录；与 dirty 标记相互独立
    git_dirty: bool | None = None  # 工作区是否有未提交变更（可得时记录）
    torch_version: str = ""
    numpy_version: str = ""


@dataclass(frozen=True)
class TrainingResult:
    """``train_model`` 的返回值：入口脚本据此编排产物写出。

    ``best_checkpoint`` → best.pt（best_val_loss 必填，绝对最优权重）；
    ``final_checkpoint`` → final.pt（实际最后完成 epoch 的权重，
    best_val_loss 为 None）；``history`` → metrics.json 的逐 epoch MSE。
    ``stop_reason``/``stop_epoch``/``detail`` 记录停止状态（ckpt 格式版本
    不变，停止元数据仅存于 TrainingResult 与 metrics.json）：

    - ``max_epochs``：跑满上限，``stop_epoch`` = max_epochs；
    - ``early_stopping``：patience 触发，``stop_epoch`` = 触发停止的 epoch；
    - ``numerical_failure``：非有限 loss/grad/pred/聚合，``detail`` 携带
      具体原因；此时 best/final 为最后一个**完整且有限** epoch 的权重，
      CLI 须以非零退出并写失败状态 metrics（不打印训练完成）。
    """

    best_checkpoint: Checkpoint
    final_checkpoint: Checkpoint
    history: tuple[EpochMetrics, ...]
    stop_reason: StopReason
    stop_epoch: int  # 触发停止的 epoch（1-based）
    detail: str | None = None  # 数值失败等的具体原因；正常停止为 None


def set_seed(seed: int) -> None:
    """seed 入口：在模型构造之前调用（权重初始化可复现的前提）。

    设定 ``torch.manual_seed(seed)`` 与 numpy 全局种子（``seed`` 取
    ``training.seed``，非负；numpy 侧按 2^32 取模）。不承诺跨硬件位级
    一致，可复现性以同机同版本为准。
    """
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)


def make_data_generator(seed: int) -> torch.Generator:
    """DataLoader 专用显式 seeded generator（shuffle 可复现；CPU 生成器）。"""
    return torch.Generator().manual_seed(seed)


def build_model(contract: InputContract, hidden_dims: tuple[int, ...]) -> MLPRegressor:
    """按输入契约与隐层宽度构建 MLP（激活固定 ReLU，见 MLPRegressor）。

    ``contract.input_shape == (P, T, 3)`` 推导展平维度 ``D = P*T*3``；
    hidden_dims 默认 ``(64, 32, 32)``。evaluate 侧以
    ``build_model(ckpt.contract, ckpt.hidden_dims)`` 恢复结构。
    """
    return MLPRegressor(input_shape=contract.input_shape, hidden_dims=tuple(hidden_dims))


def _batch_to_device(
    batch: tuple[Tensor, Tensor, object], state: PreprocessingState, device: str
) -> tuple[Tensor, Tensor]:
    """raw CPU batch → CPU 上按 state 变换为 float32 numpy → device 张量。"""
    x, y, _psids = batch
    x_dev = torch.from_numpy(transform_x(state, x.numpy())).to(device)
    y_dev = torch.from_numpy(transform_y(state, y.numpy())).to(device)
    return x_dev, y_dev


def run_validation(
    model: MLPRegressor,
    loader: DataLoader[SampleItem],
    state: PreprocessingState,
    device: str,
) -> float:
    """在给定 DataLoader 上计算标准化 MSE（按样本数加权聚合）。

    元素为 raw 未标准化的 ``SampleItem``，变换在 CPU 上按 ``state`` 施加。
    模型临时切到 eval 模式，结束后恢复原模式。预测/平方误差含非有限值 →
    TrainingError（调用方据此停止训练，不保存坏权重）。
    """
    was_training = model.training
    model.eval()
    total_se = 0.0
    total_n = 0
    try:
        with torch.no_grad():
            for batch in loader:
                x, y = _batch_to_device(batch, state, device)
                pred = model(x)
                if not bool(torch.isfinite(pred).all()):
                    raise TrainingError("验证预测含非有限值 (NaN/Inf)")
                se = (pred - y) ** 2
                if not bool(torch.isfinite(se).all()):
                    raise TrainingError("验证损失含非有限值 (NaN/Inf)")
                # float32 元素有限但 float32 求和可能溢出（如多个 1e38）：
                # 聚合一律升 float64。
                total_se += float(se.double().sum().item())
                total_n += int(x.shape[0])
    finally:
        if was_training:
            model.train()
    if total_n == 0:
        raise TrainingError("验证集为空，无法计算 val MSE")
    aggregate = total_se / (total_n * _N_OUTPUTS)
    if not math.isfinite(aggregate):
        raise TrainingError("验证聚合损失非有限 (NaN/Inf)")
    return aggregate


def _train_one_epoch(
    model: MLPRegressor,
    loader: DataLoader[SampleItem],
    optimizer: torch.optim.Optimizer,
    state: PreprocessingState,
    device: str,
) -> float | None:
    """单 epoch 训练；返回在线累计的加权平均 MSE，非有限状态返回 None（停止）。

    顺序：zero_grad → 前向（pred 非有限即停）→ batch 平均损失（非有限即停）
    → backward（梯度非有限即停，不 step）→ step → epoch 末检查权重仍有限。
    """
    model.train()
    total_se = 0.0
    total_n = 0
    for batch in loader:
        x, y = _batch_to_device(batch, state, device)
        optimizer.zero_grad()
        pred = model(x)
        if not bool(torch.isfinite(pred).all()):
            return None
        se = (pred - y) ** 2
        loss = se.mean()
        if not bool(torch.isfinite(loss)):
            return None
        loss.backward()
        if any(
            param.grad is not None and not bool(torch.isfinite(param.grad).all())
            for param in model.parameters()
        ):
            return None  # 梯度爆炸/非有限：不 step，避免坏权重
        optimizer.step()
        # float32 元素有限但 float32 求和可能溢出：聚合升 float64。
        total_se += float(se.detach().double().sum().item())
        total_n += int(x.shape[0])
        if any(not bool(torch.isfinite(param).all()) for param in model.parameters()):
            return None
    if total_n == 0:
        return None
    aggregate = total_se / (total_n * _N_OUTPUTS)
    return aggregate if math.isfinite(aggregate) else None


def _clone_state_dict(model: MLPRegressor) -> dict[str, Tensor]:
    """state_dict 的 CPU 深拷贝（后续训练不再影响已保存权重）。"""
    return {name: value.detach().to("cpu").clone() for name, value in model.state_dict().items()}


def _git_info() -> tuple[str | None, bool | None]:
    """仓库 git SHA 与 dirty 标记（只读 subprocess，限定 repo 路径；失败 → None）。"""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    if not sha:
        return None, None
    return sha, bool(status.strip())


def train_model(
    config: ExperimentConfig,
    train_set: TrajectoryDataset,
    val_set: TrajectoryDataset,
    state: PreprocessingState,
    contract: InputContract,
    split_sha256: str,
    *,
    dataset_meta_relpath: str = "dataset_meta.yaml",
    dataset_meta_sha256: str | None = None,
) -> TrainingResult:
    """训练主流程（train.md 第 8 节）。

    Args:
        config: 实验配置（seed/device/超参/early stopping）。
        train_set/val_set: raw 未标准化样本的 Dataset（test 不进入训练）。
        state: 仅 train 组拟合的预处理状态。
        contract: 输入契约（由调用方自首个 train 样本冻结）。
        split_sha256: run 内 split 副本指纹（split 绑定）。
        dataset_meta_relpath: 相对锚点 ``data_root()/samples/<dataset>/`` 的
            meta 路径（默认 "dataset_meta.yaml"）。
        dataset_meta_sha256: dataset_meta 实际文件指纹；None → 记 ""。

    编排：``set_seed``（先于模型构造）→ ``build_model`` →
    ``runtime.select_device`` → DataLoader（train shuffle=True +
    seeded generator；num_workers=0）→ Adam → 逐 epoch
    ``_train_one_epoch`` + ``run_validation`` → best（绝对最低，严格小于
    才更新，CPU clone）与 early stopping（独立 reference，仅改善 >
    min_delta 才更新）分开推进 → 非有限 loss/grad/pred/聚合立即停止且不
    保存坏权重（原因记入 ``stop_reason``/``stop_epoch``/``detail``，不吞
    掉）→ 返回 ``TrainingResult``（磁盘写出由 ``train_mlp.py`` 编排）。

    Raises:
        TrainingError: 首个 epoch 即因数值失败停止（无可保存权重）；
            异常携带 ``epoch``/``detail``。
    """
    set_seed(config.training.seed)  # 先于模型构造：权重初始化可复现
    model = build_model(contract, config.model.hidden_dims)
    device = runtime.select_device(config.training.device)
    model.to(device)
    train_loader: DataLoader[SampleItem] = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=make_data_generator(config.training.seed),
        num_workers=0,
    )
    val_loader: DataLoader[SampleItem] = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: list[EpochMetrics] = []
    best_val = math.inf
    best_state: dict[str, Tensor] | None = None
    final_state: dict[str, Tensor] | None = None
    stop_reference = math.inf  # early stopping 独立 reference（与绝对 best 分开）
    epochs_without_improvement = 0
    stop_reason: StopReason | None = None
    stop_epoch = 0
    failure_detail: str | None = None

    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, state, device)
        if train_loss is None:
            stop_reason, stop_epoch = "numerical_failure", epoch
            failure_detail = "训练侧损失/梯度/权重非有限"
            break  # 保留既有 best，不保存坏权重
        try:
            val_loss = run_validation(model, val_loader, state, device)
        except TrainingError as exc:
            stop_reason, stop_epoch, failure_detail = "numerical_failure", epoch, str(exc)
            break  # 验证侧非有限：原因经 detail 上抛，不吞掉
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            stop_reason, stop_epoch = "numerical_failure", epoch
            failure_detail = "聚合损失非有限 (NaN/Inf)"
            break  # 聚合损失非有限：保留上一完整 epoch 的 best/final，
            # 本 epoch 不写入 history、不保存当前权重
        history.append(EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        final_state = _clone_state_dict(model)  # 实际最后完成 epoch 的权重
        if val_loss < best_val:  # 绝对 best：严格更低才更新
            best_val = val_loss
            best_state = final_state
        if stop_reference - val_loss > config.training.early_stopping.min_delta:
            stop_reference = val_loss  # 仅显著改善才推进 reference
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.training.early_stopping.patience:
            stop_reason, stop_epoch = "early_stopping", epoch
            break

    if stop_reason is None:
        stop_reason, stop_epoch = "max_epochs", config.training.max_epochs

    if not history or best_state is None or final_state is None:
        raise TrainingError(
            "训练在首个 epoch 即因数值失败停止，无可保存权重"
            f"（epoch={stop_epoch}, detail={failure_detail}）",
            epoch=stop_epoch,
            detail=failure_detail,
        )

    git_sha, git_dirty = _git_info()
    common: dict[str, Any] = dict(
        ckpt_format_version=CKPT_FORMAT_VERSION,
        hidden_dims=tuple(config.model.hidden_dims),
        activation="relu",
        contract=contract,
        preprocessing=state,
        seed=config.training.seed,
        config=config,
        dataset_meta_relpath=dataset_meta_relpath,
        dataset_meta_sha256=dataset_meta_sha256 if dataset_meta_sha256 is not None else "",
        split_sha256=split_sha256,
        git_sha=git_sha,
        git_dirty=git_dirty,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
    best_checkpoint = Checkpoint(model_state_dict=best_state, best_val_loss=best_val, **common)
    final_checkpoint = replace(best_checkpoint, model_state_dict=final_state, best_val_loss=None)
    return TrainingResult(
        best_checkpoint=best_checkpoint,
        final_checkpoint=final_checkpoint,
        history=tuple(history),
        stop_reason=stop_reason,
        stop_epoch=stop_epoch,
        detail=failure_detail,
    )


def save_checkpoint(path: Path, ckpt: Checkpoint) -> None:
    """``torch.save`` 序列化 Checkpoint；同名文件已存在 → FileExistsError。

    磁盘字典仅含 Python primitives/list/dict + CPU Tensor（dataclass 与
    numpy 对象一律展开/转换，见 ``_checkpoint_to_dict``）；权重张量落盘前
    detach + CPU + clone。
    """
    if path.exists():
        raise FileExistsError(f"checkpoint 已存在，拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_to_dict(ckpt), path)


def load_checkpoint(path: Path) -> Checkpoint:
    """安全读取 Checkpoint（evaluation 侧恢复契约的唯一入口）。

    ``torch.load(weights_only=True, map_location="cpu")`` 显式安全模式；
    校验格式版本、必需键与关键字段/契约形状（t_s 长度、[P,1,3]/[2] 统计
    量形状、state_dict 为 CPU 张量字典）后重建嵌套 dataclass。

    Raises:
        FileNotFoundError: 文件不存在。
        TrainingError: 格式版本不符、键缺失或形状/类型损坏。
    """
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch.load 异常类型随版本变化，统一收敛
        raise TrainingError(f"checkpoint 加载失败 ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingError(f"checkpoint 内容须为映射 (got {type(payload)!r}) ({path})")
    required = {
        "ckpt_format_version",
        "model_state_dict",
        "hidden_dims",
        "activation",
        "contract",
        "preprocessing",
        "seed",
        "config",
        "dataset_meta_relpath",
        "dataset_meta_sha256",
        "split_sha256",
        "best_val_loss",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise TrainingError(f"checkpoint 缺失键 {missing} ({path})")
    if payload["ckpt_format_version"] != CKPT_FORMAT_VERSION:
        raise TrainingError(
            f"ckpt_format_version 不兼容: {payload['ckpt_format_version']!r} "
            f"!= {CKPT_FORMAT_VERSION} ({path})"
        )
    if payload["activation"] not in _ACTIVATIONS:
        raise TrainingError(f"未知激活函数 {payload['activation']!r} ({path})")

    state_raw = payload["model_state_dict"]
    if not isinstance(state_raw, Mapping) or not state_raw:
        raise TrainingError(f"model_state_dict 须为非空映射 ({path})")
    state_dict = {
        str(name): value for name, value in state_raw.items() if isinstance(value, Tensor)
    }
    if len(state_dict) != len(state_raw):
        raise TrainingError(f"model_state_dict 含非张量项 ({path})")
    if any(value.device.type != "cpu" for value in state_dict.values()):
        raise TrainingError(f"model_state_dict 含非 CPU 张量 ({path})")

    contract = _contract_from_dict(payload["contract"], path)
    preprocessing = _preprocessing_from_dict(
        payload["preprocessing"], len(contract.pulse_order), path
    )
    config = _config_from_dict(payload["config"], path)
    best_val_loss = payload["best_val_loss"]
    if best_val_loss is not None and not isinstance(best_val_loss, (int, float)):
        raise TrainingError(f"best_val_loss 类型非法 ({path})")
    return Checkpoint(
        ckpt_format_version=int(payload["ckpt_format_version"]),
        model_state_dict=state_dict,
        hidden_dims=tuple(int(dim) for dim in payload["hidden_dims"]),
        activation=payload["activation"],
        contract=contract,
        preprocessing=preprocessing,
        seed=int(payload["seed"]),
        config=config,
        dataset_meta_relpath=str(payload["dataset_meta_relpath"]),
        dataset_meta_sha256=str(payload["dataset_meta_sha256"]),
        split_sha256=str(payload["split_sha256"]),
        best_val_loss=float(best_val_loss) if best_val_loss is not None else None,
        git_sha=payload.get("git_sha"),
        git_dirty=payload.get("git_dirty"),
        torch_version=str(payload.get("torch_version", "")),
        numpy_version=str(payload.get("numpy_version", "")),
    )


def _contract_from_dict(raw: Any, path: Path) -> InputContract:
    """契约重建 + 形状校验（t_s 长度 == T、通道恒 3、P 与 pulse 顺序一致）。"""
    if not isinstance(raw, Mapping):
        raise TrainingError(f"contract 须为映射 ({path})")
    pulse_order = tuple(str(p) for p in raw["pulse_order"])
    n_time_steps = int(raw["n_time_steps"])
    n_channels = int(raw["n_channels"])
    t_s = np.asarray(raw["t_s"], dtype=np.float64)
    if n_time_steps < 1 or len(pulse_order) < 1:
        raise TrainingError(f"contract 的 P/T 必须为正 ({path})")
    if n_channels != 3:
        raise TrainingError(f"contract 通道数恒为 3 (got {n_channels}) ({path})")
    if t_s.shape != (n_time_steps,):
        raise TrainingError(f"contract t_s 形状 {t_s.shape} 与 T={n_time_steps} 不符 ({path})")
    return InputContract(
        pulse_order=pulse_order,
        n_time_steps=n_time_steps,
        t_s=t_s,
        n_channels=n_channels,
        component_order=tuple(str(c) for c in raw.get("component_order") or ("mx", "my", "mz")),
    )


def _preprocessing_from_dict(raw: Any, n_pulse: int, path: Path) -> PreprocessingState:
    """ckpt 预处理状态重建：委托 preprocessing.state_from_mapping（属主），
    并与契约 P 交叉校验；PreprocessingError 在 ckpt 边界包裹为 TrainingError。
    """
    try:
        state = preprocessing.state_from_mapping(raw)
    except PreprocessingError as exc:
        raise TrainingError(f"preprocessing 状态损坏 ({path}): {exc}") from exc
    if state.x_stats.mean.shape[0] != n_pulse:
        raise TrainingError(
            f"preprocessing P={state.x_stats.mean.shape[0]} 与契约 P={n_pulse} 不符 ({path})"
        )
    return state


def _config_from_dict(raw: Any, path: Path) -> ExperimentConfig:
    """ckpt 配置副本重建：委托 training_config.config_from_mapping（属主）；
    ConfigError 在 ckpt 边界包裹为 TrainingError（不反向依赖 evaluation，
    也不在 training 重复映射 schema）。
    """
    try:
        return training_config.config_from_mapping(raw)
    except ConfigError as exc:
        raise TrainingError(f"config 副本损坏 ({path}): {exc}") from exc


def _checkpoint_to_dict(ckpt: Checkpoint) -> dict[str, Any]:
    """Checkpoint → torch.save 落盘字典（仅 primitives/list/dict + CPU Tensor）。"""
    return {
        "ckpt_format_version": int(ckpt.ckpt_format_version),
        "model_state_dict": {
            name: value.detach().to("cpu").clone() for name, value in ckpt.model_state_dict.items()
        },
        "hidden_dims": [int(dim) for dim in ckpt.hidden_dims],
        "activation": str(ckpt.activation),
        "contract": {
            "pulse_order": [str(p) for p in ckpt.contract.pulse_order],
            "n_time_steps": int(ckpt.contract.n_time_steps),
            "t_s": np.asarray(ckpt.contract.t_s, dtype=np.float64).tolist(),
            "n_channels": int(ckpt.contract.n_channels),
            "component_order": [str(c) for c in ckpt.contract.component_order],
        },
        "preprocessing": preprocessing.state_to_mapping(ckpt.preprocessing),
        "seed": int(ckpt.seed),
        "config": training_config.config_to_mapping(ckpt.config),
        "dataset_meta_relpath": str(ckpt.dataset_meta_relpath),
        "dataset_meta_sha256": str(ckpt.dataset_meta_sha256),
        "split_sha256": str(ckpt.split_sha256),
        "best_val_loss": (float(ckpt.best_val_loss) if ckpt.best_val_loss is not None else None),
        "git_sha": ckpt.git_sha,
        "git_dirty": ckpt.git_dirty,
        "torch_version": str(ckpt.torch_version),
        "numpy_version": str(ckpt.numpy_version),
    }
