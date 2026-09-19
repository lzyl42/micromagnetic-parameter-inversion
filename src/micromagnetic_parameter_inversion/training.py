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
- checkpoint：``save_checkpoint``/``load_checkpoint`` 为 **MLP 专用**，
  ``CKPT_FORMAT_VERSION`` 与既有 payload/schema 不变。CNN 使用独立的
  ``CNNCheckpoint``/``save_cnn_checkpoint``/``load_cnn_checkpoint``
  （``CNN_CKPT_FORMAT_VERSION`` + ``model_kind="cnn1d"``）；
  ``load_any_checkpoint`` 单次安全读取后按显式 ``model_kind`` 路由，缺 kind
  仅接受合法旧 MLP 形态，未知/损坏一律 ``TrainingError``、不回退。CNN save 侧
  接受任意正常 Tensor 设备（不限定 CPU），落盘前统一 ``detach().to("cpu")``；
  load 侧要求 CPU。落盘字典仅含 primitives/list/dict + CPU Tensor（无
  dataclass/numpy 对象），同名文件拒绝覆盖；读取以
  ``torch.load(weights_only=True, map_location="cpu")`` 显式安全模式。
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
from micromagnetic_parameter_inversion.models.cnn1d import CNN1DRegressor
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
    CNN1DModelConfig,
    ConfigError,
    ExperimentConfig,
    ModelConfig,
    _require_positive_int,
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
_N_CHANNELS = 3  # 磁化分量数 (mx, my, mz)

# CNN 独立 checkpoint 格式版本（与 MLP 的 CKPT_FORMAT_VERSION 互不相关）。
CNN_CKPT_FORMAT_VERSION = 1
# CNN 顶层显式结构字段（恢复权威；嵌套 config 的 model 结构仅记录）。
_CNN_STRUCTURE_FIELDS = ("channels", "kernel_sizes", "pool_bins", "head_hidden_dims")
_MODEL_COMPONENT_ORDER = ("mx", "my", "mz")


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


# MLP 专用 checkpoint；CNN 见下方 CNNCheckpoint（schema 完全独立，无迁移）。
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


@dataclass(frozen=True, eq=False)
class CNNCheckpoint:
    """CNN 独立 checkpoint schema（与 ``Checkpoint`` 字段同名同义，仅结构字段不同）。

    与 ``Checkpoint`` 完全独立：以**顶层显式结构字段**为恢复权威，嵌套
    ``config`` 仅记录（不得覆盖结构）；``model_kind="cnn1d"`` 只在 payload 中
    体现，dataclass 不额外接收该参数。评估侧 ``load_state_dict(strict=True)``
    负责键/形状匹配，本模块不预构模型（避免 RNG 副作用）。
    """

    ckpt_format_version: int  # 写出时的 CNN schema 版本（CNN_CKPT_FORMAT_VERSION）
    model_state_dict: StateDict  # 模型权重（磁盘形态为 CPU Tensor 字典）
    channels: tuple[int, ...]  # 显式结构：各 Conv1d 层输出通道数
    kernel_sizes: tuple[int, ...]  # 显式结构：各层卷积核长度（奇数）
    pool_bins: int  # 显式结构：AdaptiveAvgPool1d 输出 bin 数
    head_hidden_dims: tuple[int, ...]  # 显式结构：回归头隐层宽度（可空）
    activation: ActivationName  # 激活函数显式留档（固定 "relu"）
    contract: InputContract  # 输入契约：pulse 顺序、T、分量序、t_s
    preprocessing: PreprocessingState  # 预处理状态（train-only 拟合）
    seed: int  # training.seed
    config: ExperimentConfig  # 生效配置副本（仅记录；model 须为 CNN1DModelConfig）
    dataset_meta_relpath: str  # 相对锚点 data_root()/samples/<dataset>/ 的路径
    dataset_meta_sha256: str  # dataset_meta 内容指纹（轻量溯源）
    split_sha256: str  # run 内 split 副本 sha256（split 绑定）
    best_val_loss: float | None  # best.pt 必有；final.pt 可为 None
    git_sha: str | None = None  # 可得时记录；与 dirty 标记相互独立
    git_dirty: bool | None = None  # 工作区是否有未提交变更（可得时记录）
    torch_version: str = ""
    numpy_version: str = ""


# 显式模型类别路由结果（load_any_checkpoint 返回类型）。
type ModelCheckpoint = Checkpoint | CNNCheckpoint


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


# MLP 工厂（签名/行为不变）；CNN 用下方 build_cnn_model。
def build_model(contract: InputContract, hidden_dims: tuple[int, ...]) -> MLPRegressor:
    """按输入契约与隐层宽度构建 MLP（激活固定 ReLU，见 MLPRegressor）。

    ``contract.input_shape == (P, T, 3)`` 推导展平维度 ``D = P*T*3``；
    hidden_dims 默认 ``(64, 32, 32)``。evaluate 侧以
    ``build_model(ckpt.contract, ckpt.hidden_dims)`` 恢复结构。
    """
    return MLPRegressor(input_shape=contract.input_shape, hidden_dims=tuple(hidden_dims))


def build_cnn_model(
    contract: InputContract,
    *,
    channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    pool_bins: int,
    head_hidden_dims: tuple[int, ...],
) -> CNN1DRegressor:
    """按输入契约与显式结构构建 CNN1DRegressor（无默认超参）。

    先严格校验契约语义（分量序、pulse 顺序、T/C/t_s 一致且 t_s 递增），再交
    模型构造器做结构校验（结构非法由 ``CNN1DRegressor`` 抛 ``ValueError``）。

    Raises:
        TrainingError: 契约非法。
    """
    _validate_cnn_contract(contract, "build_cnn_model")
    return CNN1DRegressor(
        input_shape=contract.input_shape,
        channels=tuple(channels),
        kernel_sizes=tuple(kernel_sizes),
        pool_bins=pool_bins,
        head_hidden_dims=tuple(head_hidden_dims),
    )


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


# TODO(CNN1D-P4): 未来共享编排 run(config_path) 放在本模块（与 train_model
# 同模块、互不 import）：解析配置一次 → 按入口要求早拒 kind 不匹配（在加载
# 数据/建目录前）→ 加载同一冻结 split/Dataset → 各自 train-only 拟合 →
# train_model → 写各自产物；保持 output_dir 覆盖与失败/best-final 语义。
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
    model_config = config.model  # P1：CNN 配置尚未支持训练，先收窄并守卫
    if not isinstance(model_config, ModelConfig):
        raise ConfigError(
            f"train_model 只支持 MLP 模型；检测到 kind={model_config.kind!r}（cnn1d 训练尚未实现）"
        )
    set_seed(config.training.seed)  # 先于模型构造：权重初始化可复现
    model = build_model(contract, model_config.hidden_dims)
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
        hidden_dims=tuple(model_config.hidden_dims),
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


def _validate_cnn_checkpoint_for_save(ckpt: CNNCheckpoint, path: Path) -> None:
    """save 前校验：明显错误（kind/version/结构/config 类型）在 mkdir/write 前拒绝。"""
    if not isinstance(ckpt, CNNCheckpoint):
        raise TrainingError(f"save_cnn_checkpoint 需要 CNNCheckpoint (got {type(ckpt)!r}) ({path})")
    _require_ckpt_version(ckpt.ckpt_format_version, CNN_CKPT_FORMAT_VERSION, path)
    if ckpt.activation not in _ACTIVATIONS:
        raise TrainingError(f"未知激活函数 {ckpt.activation!r} ({path})")
    _validate_cnn_contract(ckpt.contract, path)
    _cnn_structure_from_values(
        ckpt.channels,
        ckpt.kernel_sizes,
        ckpt.pool_bins,
        ckpt.head_hidden_dims,
        ckpt.contract.n_time_steps,
        path,
    )
    if not isinstance(ckpt.config.model, CNN1DModelConfig):
        raise TrainingError(f"CNN checkpoint 的 config.model 须为 CNN1DModelConfig ({path})")
    # 嵌套 config 往返：inf/nan 等非法数值（如 batch_size=inf 经 int 抛
    # OverflowError）在 mkdir/写盘前转为 TrainingError，不写坏数据。
    try:
        training_config.config_from_mapping(training_config.config_to_mapping(ckpt.config))
    except (ConfigError, OverflowError, TypeError, ValueError) as exc:
        raise TrainingError(f"CNN checkpoint 的 config 无法往返 ({path}): {exc}") from exc
    # save 侧只做结构校验（允许任意正常 Tensor 设备/requires_grad）；落盘统一转
    # CPU 并 detach，见 _cnn_checkpoint_to_dict。
    _cnn_state_dict_structure(ckpt.model_state_dict, path)
    _validate_cnn_preprocessing(ckpt.preprocessing, path)
    seed = ckpt.seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingError(f"seed 须为非负整数 (got {seed!r}) ({path})")
    best_val_loss = ckpt.best_val_loss
    if best_val_loss is not None and (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise TrainingError(f"best_val_loss 须为 null 或有限数值 (got {best_val_loss!r}) ({path})")


def save_cnn_checkpoint(path: Path, ckpt: CNNCheckpoint) -> None:
    """``torch.save`` 序列化 CNNCheckpoint；同名文件已存在 → FileExistsError。

    先在 mkdir/写盘前拒绝明显错误（版本/结构/config/state_dict）；落盘字典仅含
    primitives/list/dict + CPU Tensor（权重 detach + CPU + clone）。
    """
    if path.exists():
        raise FileExistsError(f"checkpoint 已存在，拒绝覆盖: {path}")
    _validate_cnn_checkpoint_for_save(ckpt, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cnn_checkpoint_to_dict(ckpt), path)


def _read_checkpoint_payload(path: Path) -> dict[str, Any]:
    """单次安全读取 checkpoint payload（weights_only、CPU、须为 dict）。"""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch.load 异常类型随版本变化，统一收敛
        raise TrainingError(f"checkpoint 加载失败 ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingError(f"checkpoint 内容须为映射 (got {type(payload)!r}) ({path})")
    return payload


def _require_ckpt_version(value: object, expected: int, context: object) -> int:
    """格式版本：非 bool 正整数且须等于 expected，否则 TrainingError。"""
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise TrainingError(f"ckpt_format_version 不兼容: {value!r} != {expected} ({context})")
    return value


_MLP_CKPT_REQUIRED_KEYS = frozenset(
    {
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
)


def _mlp_checkpoint_from_payload(payload: dict[str, Any], path: Path) -> Checkpoint:
    """MLP payload → Checkpoint（原解析主体；仅搬家 + 类别格式守卫）。

    拒绝任何 ``model_kind`` 字段与 CNN 结构字段：MLP ckpt 不使用类别键，出现
    即视为类别错配。
    """
    if "model_kind" in payload:
        raise TrainingError(f"MLP checkpoint 不接受 model_kind 字段 ({path})")
    if any(field in payload for field in _CNN_STRUCTURE_FIELDS):
        raise TrainingError(f"MLP checkpoint 不接受 CNN 结构字段 ({path})")
    missing = sorted(_MLP_CKPT_REQUIRED_KEYS.difference(payload))
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


def load_checkpoint(path: Path) -> Checkpoint:
    """安全读取 **MLP** Checkpoint（evaluation 侧 MLP 恢复入口）。

    拒绝携带 ``model_kind`` 或 CNN 结构字段的文件（类别错配）；CNN 请用
    ``load_cnn_checkpoint``，需要自动路由用 ``load_any_checkpoint``。

    Raises:
        FileNotFoundError: 文件不存在。
        TrainingError: 格式版本不符、键缺失或形状/类型损坏。
    """
    return _mlp_checkpoint_from_payload(_read_checkpoint_payload(path), path)


def load_cnn_checkpoint(path: Path) -> CNNCheckpoint:
    """安全读取 **CNN** checkpoint（要求显式 ``model_kind="cnn1d"``）。

    不接受 MLP ckpt、未知 kind 或缺失 kind；损坏一律 ``TrainingError``，无 fallback。
    """
    return _cnn_checkpoint_from_payload(_read_checkpoint_payload(path), path)


def load_any_checkpoint(path: Path) -> ModelCheckpoint:
    """单次安全读取后按显式 ``model_kind`` 路由：``cnn1d``→CNN；缺 kind→仅合法旧 MLP。

    其它显式 kind（含 ``"mlp"``）、未知/null/坏值均 ``TrainingError``；缺 kind
    但残留任意 CNN 结构字段同样拒绝，不猜测、不回退。
    """
    payload = _read_checkpoint_payload(path)
    if "model_kind" in payload:
        if payload["model_kind"] == "cnn1d":
            return _cnn_checkpoint_from_payload(payload, path)
        raise TrainingError(f"不支持的 model_kind: {payload['model_kind']!r} ({path})")
    if any(field in payload for field in _CNN_STRUCTURE_FIELDS):
        raise TrainingError(f"缺 model_kind 却残留 CNN 结构字段，拒绝按 MLP 读取 ({path})")
    return _mlp_checkpoint_from_payload(payload, path)


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


def _validate_cnn_preprocessing(state: PreprocessingState, path: Path) -> None:
    """CNN 预处理数值边界（仅 CNN 边界，旧 MLP schema/loader 不变）。

    mean（x/y）须有限；effective scale（x/y std）须有限且严格 > 0。零方差位置
    由拟合侧写成 1.0，故 1.0 合法；0/负数/inf/NaN 一律拒（不写坏数据）。
    """
    x_mean = np.asarray(state.x_stats.mean, dtype=np.float64)
    x_std = np.asarray(state.x_stats.std, dtype=np.float64)
    y_mean = np.asarray(state.y_stats.y_mean, dtype=np.float64)
    y_std = np.asarray(state.y_stats.y_std, dtype=np.float64)
    if not bool(np.isfinite(x_mean).all()):
        raise TrainingError(f"preprocessing x mean 含非有限值 ({path})")
    if not bool(np.isfinite(x_std).all()) or bool((x_std <= 0.0).any()):
        raise TrainingError(f"preprocessing x effective scale 须为有限正值 ({path})")
    if not bool(np.isfinite(y_mean).all()):
        raise TrainingError(f"preprocessing y mean 含非有限值 ({path})")
    if not bool(np.isfinite(y_std).all()) or bool((y_std <= 0.0).any()):
        raise TrainingError(f"preprocessing y effective scale 须为有限正值 ({path})")


def _config_from_dict(raw: Any, path: Path) -> ExperimentConfig:
    """ckpt 配置副本重建：委托 training_config.config_from_mapping（属主）；
    ConfigError 在 ckpt 边界包裹为 TrainingError（不反向依赖 evaluation，
    也不在 training 重复映射 schema）。
    """
    try:
        return training_config.config_from_mapping(raw)
    except (ConfigError, OverflowError) as exc:
        raise TrainingError(f"config 副本损坏 ({path}): {exc}") from exc


def _cnn_state_dict_structure(raw: Any, path: Path) -> dict[str, Tensor]:
    """CNN state_dict 结构校验（非空映射、str 键、Tensor 值）；不限定设备。

    save 侧使用：允许任意正常 Tensor 设备（如 CUDA/requires_grad），落盘前由
    ``_cnn_checkpoint_to_dict`` 统一 ``detach().to("cpu").clone()``。
    """
    if not isinstance(raw, Mapping) or not raw:
        raise TrainingError(f"model_state_dict 须为非空映射 ({path})")
    state_dict: dict[str, Tensor] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            raise TrainingError(f"model_state_dict 键须为字符串 (got {type(name)!r}) ({path})")
        if not isinstance(value, Tensor):
            raise TrainingError(f"model_state_dict 含非张量项 ({path})")
        state_dict[name] = value
    return state_dict


def _cnn_state_dict_from_payload(raw: Any, path: Path) -> dict[str, Tensor]:
    """CNN 加载边界：结构校验 + 必须为 CPU 张量（损坏统一 TrainingError）。"""
    state_dict = _cnn_state_dict_structure(raw, path)
    for value in state_dict.values():
        if value.device.type != "cpu":
            raise TrainingError(f"model_state_dict 含非 CPU 张量 ({path})")
    return state_dict


def _cnn_positive_int(value: object, field: str) -> int:
    """正整数（拒 bool/float/string）；ConfigError 统一包为 TrainingError。"""
    try:
        return _require_positive_int(value, field)
    except ConfigError as exc:
        raise TrainingError(str(exc)) from exc


def _cnn_int_sequence(value: object, field: str, *, allow_empty: bool) -> tuple[int, ...]:
    """CNN 结构整数序列：接受 tuple/list，拒 bool/float/string 与非正元素。"""
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TrainingError(f"{field}: 必须为整数序列 (got {value!r})")
    sequence = tuple(value)
    if not sequence and not allow_empty:
        raise TrainingError(f"{field}: 必须为非空整数序列 (got {value!r})")
    for index, item in enumerate(sequence):
        _cnn_positive_int(item, f"{field}[{index}]")
    return sequence


def _cnn_structure_from_values(
    channels: object,
    kernel_sizes: object,
    pool_bins: object,
    head_hidden_dims: object,
    n_time_steps: int,
    context: object,
) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]:
    """CNN 结构严格校验（等价于配置层规范：拒 bool/float/string 蒙混）。"""
    normalized_channels = _cnn_int_sequence(channels, f"{context}: channels", allow_empty=False)
    normalized_kernels = _cnn_int_sequence(
        kernel_sizes, f"{context}: kernel_sizes", allow_empty=False
    )
    normalized_pool_bins = _cnn_positive_int(pool_bins, f"{context}: pool_bins")
    normalized_head = _cnn_int_sequence(
        head_hidden_dims, f"{context}: head_hidden_dims", allow_empty=True
    )
    if len(normalized_kernels) != len(normalized_channels):
        raise TrainingError(
            f"kernel_sizes 层数须与 channels 相同 "
            f"(got {len(normalized_kernels)} vs {len(normalized_channels)}) ({context})"
        )
    if any(kernel % 2 == 0 for kernel in normalized_kernels):
        raise TrainingError(
            f"kernel_sizes 须全为奇数 (got {list(normalized_kernels)!r}) ({context})"
        )
    if normalized_pool_bins > n_time_steps:
        raise TrainingError(
            f"pool_bins 须 <= T={n_time_steps} (got {normalized_pool_bins}) ({context})"
        )
    return normalized_channels, normalized_kernels, normalized_pool_bins, normalized_head


def _validate_cnn_contract(contract: InputContract, context: object) -> None:
    """CNN 契约语义校验：分量序、pulse 顺序、T/C/t_s 一致且 t_s 严格递增。"""
    if not isinstance(contract, InputContract):
        raise TrainingError(f"contract 须为 InputContract (got {type(contract)!r}) ({context})")
    if tuple(contract.component_order) != _MODEL_COMPONENT_ORDER:
        raise TrainingError(
            f"contract 分量序须为 {_MODEL_COMPONENT_ORDER} "
            f"(got {contract.component_order!r}) ({context})"
        )
    pulse_order = tuple(contract.pulse_order)
    if (
        not pulse_order
        or any(not isinstance(p, str) or not p for p in pulse_order)
        or len(set(pulse_order)) != len(pulse_order)
    ):
        raise TrainingError(f"contract pulse_order 须为非空唯一非空字符串序列 ({context})")
    n_time = contract.n_time_steps
    if isinstance(n_time, bool) or not isinstance(n_time, int) or n_time <= 0:
        raise TrainingError(f"contract n_time_steps 须为正整数 (got {n_time!r}) ({context})")
    if contract.n_channels != _N_CHANNELS:
        raise TrainingError(
            f"contract 通道数须为 {_N_CHANNELS} (got {contract.n_channels}) ({context})"
        )
    t_s = np.asarray(contract.t_s, dtype=np.float64)
    if t_s.shape != (n_time,):
        raise TrainingError(f"contract t_s 形状 {t_s.shape} 与 T={n_time} 不符 ({context})")
    if not bool(np.isfinite(t_s).all()):
        raise TrainingError(f"contract t_s 含非有限值 ({context})")
    if n_time > 1 and not bool(np.all(np.diff(t_s) > 0)):
        raise TrainingError(f"contract t_s 须严格递增 ({context})")


def _cnn_contract_from_dict(raw: Any, path: Path) -> InputContract:
    """CNN 契约严格重建（bool/float 冒充 T/C 拒；t_s 长度一致、有限、严格递增）。"""
    if not isinstance(raw, Mapping):
        raise TrainingError(f"contract 须为映射 ({path})")
    missing = sorted(
        {"pulse_order", "n_time_steps", "n_channels", "t_s", "component_order"}.difference(raw)
    )
    if missing:
        raise TrainingError(f"contract 缺失键 {missing} ({path})")
    pulse_raw = raw["pulse_order"]
    if isinstance(pulse_raw, (str, bytes)) or not isinstance(pulse_raw, (list, tuple)):
        raise TrainingError(f"contract pulse_order 须为序列 ({path})")
    pulse_order = tuple(pulse_raw)
    if (
        not pulse_order
        or any(not isinstance(p, str) or not p for p in pulse_order)
        or len(set(pulse_order)) != len(pulse_order)
    ):
        raise TrainingError(f"contract pulse_order 须为非空唯一非空字符串序列 ({path})")
    try:
        n_time_steps = _require_positive_int(raw["n_time_steps"], f"contract.n_time_steps ({path})")
        n_channels = _require_positive_int(raw["n_channels"], f"contract.n_channels ({path})")
    except ConfigError as exc:
        raise TrainingError(str(exc)) from exc
    if n_channels != _N_CHANNELS:
        raise TrainingError(f"contract 通道数须为 {_N_CHANNELS} (got {n_channels}) ({path})")
    component_raw = raw["component_order"]
    if isinstance(component_raw, (str, bytes)) or not isinstance(component_raw, (list, tuple)):
        raise TrainingError(f"contract component_order 须为序列 ({path})")
    component_order = tuple(component_raw)
    if component_order != _MODEL_COMPONENT_ORDER:
        raise TrainingError(
            f"contract 分量序须为 {_MODEL_COMPONENT_ORDER} (got {component_order!r}) ({path})"
        )
    t_raw = raw["t_s"]
    if isinstance(t_raw, (str, bytes)) or not isinstance(t_raw, (list, tuple, np.ndarray)):
        raise TrainingError(f"contract t_s 须为序列 ({path})")
    try:
        t_s = np.asarray(t_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"contract t_s 无法解析为 float64 ({path}): {exc}") from exc
    if t_s.ndim != 1 or t_s.shape != (n_time_steps,):
        raise TrainingError(f"contract t_s 形状 {t_s.shape} 与 T={n_time_steps} 不符 ({path})")
    if not bool(np.isfinite(t_s).all()):
        raise TrainingError(f"contract t_s 含非有限值 ({path})")
    if n_time_steps > 1 and not bool(np.all(np.diff(t_s) > 0)):
        raise TrainingError(f"contract t_s 须严格递增 ({path})")
    return InputContract(
        pulse_order=pulse_order,
        n_time_steps=n_time_steps,
        t_s=t_s,
        n_channels=n_channels,
        component_order=component_order,
    )


_CNN_CKPT_REQUIRED_KEYS = frozenset(
    {
        "ckpt_format_version",
        "model_kind",
        "model_state_dict",
        "channels",
        "kernel_sizes",
        "pool_bins",
        "head_hidden_dims",
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
)


def _cnn_checkpoint_from_payload(payload: dict[str, Any], path: Path) -> CNNCheckpoint:
    """CNN payload → CNNCheckpoint（严格类别/版本/结构/契约；config.model 须 CNN）。"""
    missing = sorted(_CNN_CKPT_REQUIRED_KEYS.difference(payload))
    if missing:
        raise TrainingError(f"CNN checkpoint 缺失键 {missing} ({path})")
    if payload["model_kind"] != "cnn1d":
        raise TrainingError(f"model_kind 须为 'cnn1d' (got {payload['model_kind']!r}) ({path})")
    if "hidden_dims" in payload:
        raise TrainingError(f"CNN checkpoint 不接受 hidden_dims 字段 ({path})")
    _require_ckpt_version(payload["ckpt_format_version"], CNN_CKPT_FORMAT_VERSION, path)
    if payload["activation"] not in _ACTIVATIONS:
        raise TrainingError(f"未知激活函数 {payload['activation']!r} ({path})")
    state_dict = _cnn_state_dict_from_payload(payload["model_state_dict"], path)
    contract = _cnn_contract_from_dict(payload["contract"], path)
    channels, kernel_sizes, pool_bins, head_hidden_dims = _cnn_structure_from_values(
        payload["channels"],
        payload["kernel_sizes"],
        payload["pool_bins"],
        payload["head_hidden_dims"],
        contract.n_time_steps,
        path,
    )
    preprocessing = _preprocessing_from_dict(
        payload["preprocessing"], len(contract.pulse_order), path
    )
    _validate_cnn_preprocessing(preprocessing, path)
    config = _config_from_dict(payload["config"], path)
    if not isinstance(config.model, CNN1DModelConfig):
        raise TrainingError(f"CNN checkpoint 的 config.model 须为 CNN1DModelConfig ({path})")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingError(f"seed 须为非负整数 (got {seed!r}) ({path})")
    best_val_loss = payload["best_val_loss"]
    if best_val_loss is not None and (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise TrainingError(f"best_val_loss 须为 null 或有限数值 (got {best_val_loss!r}) ({path})")
    return CNNCheckpoint(
        ckpt_format_version=CNN_CKPT_FORMAT_VERSION,
        model_state_dict=state_dict,
        channels=channels,
        kernel_sizes=kernel_sizes,
        pool_bins=pool_bins,
        head_hidden_dims=head_hidden_dims,
        activation=payload["activation"],
        contract=contract,
        preprocessing=preprocessing,
        seed=seed,
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


def _contract_to_dict(contract: InputContract) -> dict[str, Any]:
    """InputContract → 纯字典（MLP/CNN 落盘共用；schema 不变）。"""
    return {
        "pulse_order": [str(p) for p in contract.pulse_order],
        "n_time_steps": int(contract.n_time_steps),
        "t_s": np.asarray(contract.t_s, dtype=np.float64).tolist(),
        "n_channels": int(contract.n_channels),
        "component_order": [str(c) for c in contract.component_order],
    }


def _checkpoint_to_dict(ckpt: Checkpoint) -> dict[str, Any]:
    """Checkpoint → torch.save 落盘字典（仅 primitives/list/dict + CPU Tensor）。"""
    return {
        "ckpt_format_version": int(ckpt.ckpt_format_version),
        "model_state_dict": {
            name: value.detach().to("cpu").clone() for name, value in ckpt.model_state_dict.items()
        },
        "hidden_dims": [int(dim) for dim in ckpt.hidden_dims],
        "activation": str(ckpt.activation),
        "contract": _contract_to_dict(ckpt.contract),
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


def _cnn_checkpoint_to_dict(ckpt: CNNCheckpoint) -> dict[str, Any]:
    """CNNCheckpoint → torch.save 落盘字典（primitives/list/dict + CPU Tensor）。"""
    return {
        "ckpt_format_version": int(ckpt.ckpt_format_version),
        "model_kind": "cnn1d",
        "model_state_dict": {
            name: value.detach().to("cpu").clone() for name, value in ckpt.model_state_dict.items()
        },
        "channels": [int(c) for c in ckpt.channels],
        "kernel_sizes": [int(k) for k in ckpt.kernel_sizes],
        "pool_bins": int(ckpt.pool_bins),
        "head_hidden_dims": [int(h) for h in ckpt.head_hidden_dims],
        "activation": str(ckpt.activation),
        "contract": _contract_to_dict(ckpt.contract),
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
