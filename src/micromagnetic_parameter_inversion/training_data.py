"""训练样本数据层：npz schema、协议快照、split 与 Dataset（已实现）。

对应 ``train.md`` 第 2/3/4 节。核心契约：

- npz schema（每参数组一个 ``<psid>.npz``）：``x: float32 [P, T, 3]``（按
  冻结 pulse 顺序 stack，通道序 mx,my,mz）、``y: float32 [2]``（alpha,
  ku_j_per_m3 原始物理单位）、``parameter_set_id: <U16`` 标量（源 psid，
  split 单位）、``t_s: float64 [T]``、``pulse_ids: <U… [P]``；写用
  ``np.savez_compressed``（绝无 object 数组），读取一律
  ``np.load(..., allow_pickle=False)``。
- 成员边界：Dataset/split 仅按显式成员构造与校验（三组互斥、并集恰等于
  清单成员）；不做交集过滤、不扫目录遗留旧 npz、不允许运行时重切。
- 只做读入必需校验与契约校验：不物理 QC、不逐组协议审核。
- 协议快照：所选 psid 升序取首的 config.yaml，safe_load 只提取
  recording/pulses（缺 numerics 的旧快照合法），mT→T 换算 ×1e-3 按
  mumax3_config schema 字段。

dataset_meta.yaml 序列化 schema（``write_prepared_dataset`` 写出、
``load_dataset_meta`` 读回；可选键为 null 时按缺省处理）::

    dataset_name: str
    generated_at: str | null          # ISO-8601 UTC
    n_time_steps: int                 # T
    pulse_order: [pid, ...]
    protocol: {sample_interval_s: float, pulses: {pid: {b_ext_amplitude_t,
              direction: [dx,dy,dz], duration_s: float}}}
    labels: {psid: [alpha, ku_j_per_m3]}
    members: [psid, ...]
    source_index_relpaths: {psid: "<psid>/index.csv"} | null
    config_sha256: {psid: "<hex>"} | null

split.yaml 序列化 schema::

    seed: int
    ratios: {train: float, val: float, test: float}
    train: [psid, ...]
    val: [psid, ...]
    test: [psid, ...]

错误类型：schema 数值/字段违反抛 ConfigError（复用 mumax3_config 校验
器），布局/成员/契约违反抛 DataError（消息含完整路径）。

依赖方向：training_config →（本模块）→ preprocessing/training。
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import Dataset

from micromagnetic_parameter_inversion.mumax3_config import (
    _require_finite,
    _require_mapping,
    _require_non_negative_int,
    _require_non_negative_number,
    _require_positive_int,
    _require_positive_number,
    _require_safe_path_segment,
    _require_vector3,
    _UniqueKeyLoader,
)
from micromagnetic_parameter_inversion.training_config import SplitConfig, SplitRatios

# npz 数组 dtype 契约（train.md 第 2 节第 6 条）。
X_DTYPE = "float32"  # x: [P, T, 3]，原始时域轨迹，通道序 (mx, my, mz)
Y_DTYPE = "float32"  # y: [2] = (alpha, ku_j_per_m3)，原始物理单位
T_S_DTYPE = "float64"  # t_s: [T]，实际时间网格（取自该组首条轨迹时间列）
PSID_DTYPE = "<U16"  # parameter_set_id 标量：SHA-256 前 16 位

# 轨迹 CSV 布局契约（生成侧已核实，表头由生成侧 parse_table 校验）：此处
# 只按列序取值并做形状必需检查。
TRAJECTORY_USECOLS = (2, 3, 4)  # 仅取 m_x, m_y, m_z（sample_index/t_s 不进 x）
T_S_USECOLS = (1,)  # 首条轨迹的 t_s 列

# index.csv 固定列（mumax3_pipeline.py 已核实）。
INDEX_COLUMNS = (
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

# 协议快照所需 schema（对齐 mumax3_config；numerics 节不读取）。
_RECORDING_KEYS = frozenset({"sample_interval_s", "sample_count"})
_PULSE_KEYS = frozenset({"pulse_id", "b_ext_amplitude_mT", "direction", "duration_s"})

# dataset_meta.yaml / split.yaml schema。
_META_REQUIRED_KEYS = frozenset(
    {"dataset_name", "pulse_order", "n_time_steps", "labels", "members"}
)
_META_OPTIONAL_KEYS = frozenset(
    {"protocol", "config_sha256", "source_index_relpaths", "generated_at"}
)
_SPLIT_YAML_KEYS = frozenset({"seed", "ratios", "train", "val", "test"})
_RATIO_KEYS = frozenset({"train", "val", "test"})
_SPLIT_NAMES = ("train", "val", "test")
# ratios 求和容差（与 training_config 一致）。
_RATIO_TOLERANCE = 1e-9

# 数据集元素类型：未标准化的 raw 样本 (x float32[P,T,3], y float32[2], psid)；
# 也是 DataLoader 的 batch 迭代单元（训练/评估侧据此施加预处理）。
type SampleItem = tuple[Tensor, Tensor, str]


class DataError(ValueError):
    """样本/清单/split 布局或成员契约违反；消息含完整路径。"""


def _fail(field: str, problem: str) -> NoReturn:
    """抛出带字段路径的 DataError，便于定位具体文件/字段。"""
    raise DataError(f"{field}: {problem}")


def _check_keys(
    mapping: dict[Any, Any], field: str, known: frozenset[str], required: frozenset[str]
) -> None:
    """严格 schema：拒绝未知字段；仅 ``required`` 子集为必填。"""
    unknown = sorted((key for key in mapping if key not in known), key=repr)
    if unknown:
        _fail(field, f"未知字段 {unknown!r}；允许的字段 {sorted(known)}")
    missing = sorted(required.difference(mapping))
    if missing:
        _fail(field, f"缺失必填字段 {missing!r}")


def _load_yaml(path: Path, field: str) -> dict[Any, Any]:
    """safe_load + 拒绝重复键；非映射即 DataError。"""
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        _fail(field, f"YAML 解析失败 ({path}): {exc}")
    except OSError as exc:
        _fail(field, f"读取失败 ({path}): {exc}")
    return _require_mapping(raw, field)


@dataclass(frozen=True, eq=False)
class InputContract:
    """输入契约：ckpt 携带，evaluate 据此逐 npz 校验（契约校验，非物理 QC）。

    含 ndarray 字段，``eq=False`` 避免逐元素比较语义；一致性以字段值显式
    比较（pulse 顺序 / T / 形状 / t_s 逐位相等）为准。
    """

    pulse_order: tuple[str, ...]  # 冻结 pulse_id 序列（x 的 P 维顺序）
    n_time_steps: int  # T：每 pulse 时间步数（冻结 sample_count）
    t_s: np.ndarray  # float64 [T]，实际时间网格
    n_channels: int = 3  # 磁化分量数，恒为 3
    component_order: tuple[str, ...] = ("mx", "my", "mz")  # 通道序

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """``(P, T, 3)``：MLPRegressor(input_shape=...) 的形状参数。"""
        return (len(self.pulse_order), self.n_time_steps, self.n_channels)


@dataclass(frozen=True, eq=False)
class Sample:
    """单个参数组的规范化样本（一个 npz 的内存表示；含 ndarray，eq=False）。"""

    parameter_set_id: str  # 源 psid（split 单位；仅由 (alpha, Ku) 决定）
    x: np.ndarray  # float32 [P, T, 3]，按冻结 pulse 顺序 stack
    y: np.ndarray  # float32 [2] = (alpha, ku_j_per_m3)，原始单位
    t_s: np.ndarray  # float64 [T]
    pulse_ids: tuple[str, ...]  # [P]，与 x 的 P 维顺序一致


@dataclass(frozen=True)
class ProtocolSummary:
    """首份 config.yaml 快照冻结出的协议摘要（prepare 侧只读不查）。"""

    pulse_order: tuple[str, ...]  # 冻结 pulse 顺序（所有 npz 共用）
    sample_interval_s: float  # recording.sample_interval_s，s
    n_time_steps: int  # T = recording.sample_count
    # pulse_id → {b_ext_amplitude_t（已按 schema ×1e-3 转 T）, direction,
    # duration_s}。
    pulses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMeta:
    """dataset_meta.yaml 的内存表示（prepare 产出；schema 见模块 docstring）。"""

    dataset_name: str
    pulse_order: tuple[str, ...]  # 冻结 pulse 顺序
    n_time_steps: int  # T = 冻结 sample_count
    labels: Mapping[str, tuple[float, float]]  # psid → (alpha, ku_j_per_m3)
    members: tuple[str, ...]  # 本清单成员 psid 列表（split 并集须恰等于它）
    # 协议摘要：{"sample_interval_s": float, "pulses": {pid: {...}}}。
    protocol: Mapping[str, Any] | None = None
    # 以下为轻量溯源（可选；写 meta 时填充）。
    config_sha256: Mapping[str, str] | None = None  # psid → config.yaml sha256
    source_index_relpaths: Mapping[str, str] | None = None  # psid → 源 index 相对路径
    generated_at: str | None = None  # ISO-8601 UTC 生成时间


@dataclass(frozen=True)
class SplitDefinition:
    """split.yaml 的内存表示；split 单位 = parameter_set_id（按 ID 分组）。"""

    seed: int
    ratios: SplitRatios
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def members_of(self, split_name: str) -> tuple[str, ...]:
        """按名称取成员（train/val/test；未知名称报错）。"""
        members: dict[str, tuple[str, ...]] = dict(
            zip(_SPLIT_NAMES, (self.train, self.val, self.test), strict=True)
        )
        try:
            return members[split_name]
        except KeyError:
            msg = f"未知 split 名称 {split_name!r}；允许 {list(_SPLIT_NAMES)}"
            raise DataError(msg) from None


def sha256_file(path: Path) -> str:
    """文件 sha256（小写 hex；meta 溯源与 split/ckpt 绑定共用）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrajectoryDataset(Dataset[SampleItem]):
    """按 split 显式成员构造的参数组级数据集。

    元素类型 ``SampleItem``：raw、未标准化；``__getitem__`` 返回
    ``(x_tensor float32 CPU, y_tensor float32 CPU, psid)``，标准化在
    preprocessing 由编排层施加。仅加载成员清单内的 npz：目录中遗留的旧
    npz（不在清单）不被加载。每个 npz **恰好读盘一次**并缓存为
    ``Sample``（``samples`` 属性/``validate_contract`` 均纯内存访问）。
    """

    def __init__(
        self,
        samples_dir: Path,
        meta: DatasetMeta,
        psids: Sequence[str],
        *,
        contract: InputContract | None = None,
    ) -> None:
        """按显式成员列表加载并校验全部 npz（成员必须存在于 meta.members）。

        Args:
            samples_dir: npz 所在目录。
            meta: 数据集清单（成员校验基准）。
            psids: 本 split 的显式成员列表。
            contract: 非 None 时逐样本在加载时校验输入契约（一次读盘即
                完成验证，不回磁盘二次读）。
        """
        members = set(meta.members)
        unknown = [psid for psid in psids if psid not in members]
        if unknown:
            _fail("psids", f"成员不在 dataset_meta 清单中: {unknown}")
        if len(set(psids)) != len(psids):
            _fail("psids", f"重复的 psid: {list(psids)}")
        self._samples_dir = samples_dir
        self._meta = meta
        self._psids = tuple(psids)
        self._samples = tuple(
            _read_sample_npz(samples_dir / f"{psid}.npz", psid, contract) for psid in psids
        )

    @property
    def samples(self) -> tuple[Sample, ...]:
        """缓存的 ``Sample`` 元组（按成员顺序；访问不触发磁盘读取）。"""
        return self._samples

    def validate_contract(self, contract: InputContract) -> None:
        """对全部缓存样本做输入契约校验（纯内存，不回磁盘）。

        用于"契约自首个缓存样本冻结后校验其余成员"的场景；与
        ``load_sample``/加载期校验走同一检查（``_check_sample_contract``）。
        """
        for sample in self._samples:
            _check_sample_contract(sample, contract, f"psid={sample.parameter_set_id}")

    def __len__(self) -> int:
        """成员数。"""
        return len(self._samples)

    def __getitem__(self, index: int) -> SampleItem:
        """返回 raw ``(x_tensor float32[P,T,3], y_tensor float32[2], psid)``。"""
        sample = self._samples[index]
        return torch.from_numpy(sample.x), torch.from_numpy(sample.y), sample.parameter_set_id


def load_protocol_snapshot(
    raw_dataset_dir: Path,
    psids: Sequence[str],
    pulse_order: Sequence[str] | None,
) -> ProtocolSummary:
    """读取协议快照：所选 psid 升序排序后的首份 config.yaml。

    safe_load 只提取 recording（sample_interval_s、sample_count → T）与
    pulses；旧快照缺 numerics 合法；mT→T 换算按 schema 字段
    b_ext_amplitude_mT ×1e-3。``pulse_order`` 非 None 时须为快照 pulse
    集合的无重复排列（缺失/多出/重复均报错）；None = 快照内声明顺序。

    Raises:
        DataError: 快照缺失/结构违反/顺序不一致。
        ConfigError: 数值 schema 违反（复用 mumax3_config 校验器）。
    """
    if not psids:
        _fail("psids", "所选参数组列表为空")
    ordered = sorted(psids)
    config_path = raw_dataset_dir / _require_safe_path_segment(ordered[0], "psid") / "config.yaml"
    if not config_path.is_file():
        _fail("config.yaml", f"协议快照缺失: {config_path}")
    snapshot = _load_yaml(config_path, f"{config_path}:root")

    recording_raw = snapshot.get("recording")
    if not isinstance(recording_raw, dict):
        _fail(f"{config_path}:recording", "快照缺少 recording 节")
    _check_keys(recording_raw, f"{config_path}:recording", _RECORDING_KEYS, _RECORDING_KEYS)
    sample_interval_s = _require_positive_number(
        recording_raw["sample_interval_s"], f"{config_path}:recording.sample_interval_s"
    )
    n_time_steps = _require_positive_int(
        recording_raw["sample_count"], f"{config_path}:recording.sample_count"
    )

    pulses_raw = snapshot.get("pulses")
    if not isinstance(pulses_raw, list) or not pulses_raw:
        _fail(f"{config_path}:pulses", "必须为非空 pulse 列表")
    snapshot_order: list[str] = []
    pulses: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(pulses_raw):
        pulse_field = f"{config_path}:pulses[{index}]"
        pulse_raw = _require_mapping(item, pulse_field)
        _check_keys(pulse_raw, pulse_field, _PULSE_KEYS, _PULSE_KEYS)
        pulse_id = _require_safe_path_segment(pulse_raw["pulse_id"], f"{pulse_field}.pulse_id")
        if pulse_id in pulses:
            _fail(f"{pulse_field}.pulse_id", f"重复的 pulse_id {pulse_id!r}")
        amplitude_millitesla = _require_non_negative_number(
            pulse_raw["b_ext_amplitude_mT"], f"{pulse_field}.b_ext_amplitude_mT"
        )
        direction = _require_vector3(pulse_raw["direction"], f"{pulse_field}.direction")
        duration_s = _require_positive_number(pulse_raw["duration_s"], f"{pulse_field}.duration_s")
        snapshot_order.append(pulse_id)
        # 单位边界：mT 只存在于 YAML，在此 ×1e-3 存为 T（同 mumax3_config）。
        pulses[pulse_id] = {
            "b_ext_amplitude_t": amplitude_millitesla * 1e-3,
            "direction": direction,
            "duration_s": duration_s,
        }

    frozen_order: tuple[str, ...]
    if pulse_order is not None:
        if len(set(pulse_order)) != len(pulse_order):
            _fail("data.pulse_order", f"重复的 pulse_id: {list(pulse_order)}")
        if set(pulse_order) != set(snapshot_order):
            _fail(
                "data.pulse_order",
                f"须为快照 pulse 集合的排列；快照 {sorted(snapshot_order)}，"
                f"配置 {sorted(pulse_order)}",
            )
        frozen_order = tuple(pulse_order)
    else:
        frozen_order = tuple(snapshot_order)

    return ProtocolSummary(
        pulse_order=frozen_order,
        sample_interval_s=sample_interval_s,
        n_time_steps=n_time_steps,
        pulses=pulses,
    )


def read_parameter_group(
    raw_dataset_dir: Path,
    psid: str,
    pulse_order: Sequence[str],
    n_time_steps: int,
) -> tuple[Sample, tuple[float, float]]:
    """读取单个参数组并组装为 ``(Sample, (alpha, ku_j_per_m3))``。

    标签为 index.csv 的原值（float64 精度，供 dataset_meta 溯源）；npz 内
    的 ``y`` 为其 float32 形态。

    - index.csv：必须存在、固定列齐全；行 pulse_id 集合须与 ``pulse_order``
      恰好一致（缺失/多出/重复均报错）；(alpha, ku_j_per_m3) 同组一致。
    - trajectory.csv：``trajectory_path`` 须为 psid 目录内的相对路径且存在；
      按 ``usecols=(2,3,4), ndmin=2`` 读磁化 → 形状 ``(T,3)``（T =
      ``n_time_steps``）；t_s 取 ``pulse_order[0]`` 轨迹（``ndmin=1``）。
    - 不物理 QC：磁化数值本身不校验（生成侧 parse_table 已查）。
    """
    psid_dir = raw_dataset_dir / _require_safe_path_segment(psid, "parameter_set_id")
    if not psid_dir.is_dir():
        _fail("parameter_set_dir", f"参数组目录不存在: {psid_dir}")

    rows = _read_index_rows(psid_dir / "index.csv")
    pulse_field = f"{psid_dir / 'index.csv'}:pulse_id"
    row_by_pulse: dict[str, dict[str, str]] = {}
    for row in rows:
        pulse_id = (row.get("pulse_id") or "").strip()
        if not pulse_id:
            continue
        if pulse_id in row_by_pulse:
            _fail(pulse_field, f"重复的 pulse_id {pulse_id!r}")
        if pulse_id not in set(pulse_order):
            _fail(pulse_field, f"多出的 pulse {pulse_id!r}（冻结顺序 {list(pulse_order)}）")
        row_by_pulse[pulse_id] = row
    missing = [pid for pid in pulse_order if pid not in row_by_pulse]
    if missing:
        _fail(pulse_field, f"缺失 pulse {missing}（冻结顺序 {list(pulse_order)}）")

    alpha, ku = _group_labels(psid, [row_by_pulse[pid] for pid in pulse_order])

    magnetization: list[np.ndarray] = []
    t_s: np.ndarray | None = None
    for position, pulse_id in enumerate(pulse_order):
        trajectory_path = _resolve_trajectory_path(psid_dir, row_by_pulse[pulse_id])
        magnetization.append(
            _load_trajectory(
                trajectory_path, TRAJECTORY_USECOLS, ndmin=2, expected_shape=(n_time_steps, 3)
            )
        )
        if position == 0:
            t_s = _load_trajectory(
                trajectory_path, T_S_USECOLS, ndmin=1, expected_shape=(n_time_steps,)
            )
    assert t_s is not None  # pulse_order 非空（协议快照保证）
    x = np.stack(magnetization).astype(np.float32)
    y = np.array([alpha, ku], dtype=np.float32)
    sample = Sample(parameter_set_id=psid, x=x, y=y, t_s=t_s, pulse_ids=tuple(pulse_order))
    return sample, (alpha, ku)


def _read_index_rows(index_path: Path) -> list[dict[str, str]]:
    """读 index.csv：文件存在、固定列齐全（多余列容忍），返回行字典列表。"""
    if not index_path.is_file():
        _fail("index.csv", f"文件不存在: {index_path}")
    with index_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [column for column in INDEX_COLUMNS if column not in fieldnames]
        if missing:
            _fail(f"{index_path}:header", f"缺失固定列 {missing}")
        return list(reader)


def _group_labels(psid: str, rows: Sequence[Mapping[str, str]]) -> tuple[float, float]:
    """(alpha, ku_j_per_m3)：解析自 index 行，同组各行必须一致。"""
    alpha: float | None = None
    ku: float | None = None
    for row in rows:
        row_alpha = _parse_index_number(row["alpha"], f"{psid}:alpha")
        row_ku = _parse_index_number(row["ku_j_per_m3"], f"{psid}:ku_j_per_m3")
        if alpha is None:
            alpha, ku = row_alpha, row_ku
        elif alpha != row_alpha or ku != row_ku:
            _fail(f"{psid}:labels", f"组内标签不一致: ({alpha}, {ku}) vs ({row_alpha}, {row_ku})")
    assert alpha is not None and ku is not None  # pulse_order 非空保证至少一行
    return alpha, ku


def _parse_index_number(value: str, field: str) -> float:
    """index 数值列：有限数（Ku 可 0/负，alpha 非负性由生成侧保证）。"""
    try:
        number = float(value)
    except ValueError as exc:
        _fail(field, f"必须为数值 (got {value!r}): {exc}")
    if not math.isfinite(number):
        _fail(field, f"必须为有限数值 (got {value!r})")
    return number


def _resolve_trajectory_path(psid_dir: Path, row: Mapping[str, str]) -> Path:
    """trajectory_path：须为 psid 目录内的相对路径且存在（防路径逃逸）。"""
    relative = (row.get("trajectory_path") or "").strip()
    if not relative:
        _fail(f"{psid_dir / 'index.csv'}:trajectory_path", "trajectory_path 为空")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        _fail(f"{psid_dir / 'index.csv'}:trajectory_path", f"不接受绝对路径: {relative!r}")
    candidate = psid_dir / relative_path
    try:
        candidate.resolve().relative_to(psid_dir.resolve())
    except ValueError:
        _fail(f"{psid_dir / 'index.csv'}:trajectory_path", f"路径逃逸出参数组目录: {relative!r}")
    if not candidate.is_file():
        _fail("trajectory.csv", f"文件不存在: {candidate}")
    return candidate


def _load_trajectory(
    path: Path,
    usecols: tuple[int, ...],
    ndmin: Literal[0, 1, 2],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """读 trajectory.csv 指定列；``ndmin`` 防止单行/单列被降维；形状必需检查。"""
    try:
        array = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols, ndmin=ndmin)
    except (OSError, ValueError) as exc:
        _fail("trajectory.csv", f"读取失败 ({path}): {exc}")
    if array.shape != expected_shape:
        _fail("trajectory.csv", f"形状不符 ({path}): got {array.shape}, expected {expected_shape}")
    return array


def make_split(psids: Sequence[str], config: SplitConfig) -> SplitDefinition:
    """确定性划分（train.md 第 3 节）：最大余数法，tie 固定 train→val→test。

    步骤：psid 去重升序 → ``np.random.default_rng(seed).permutation`` 打乱
    → 各组先取 ``floor(n·ratio)``，剩余名额按小数余数降序逐个 +1（余数并列
    固定 train→val→test 优先）→ 检查 ``min_per_split``，任一组不足即报错
    （不硬凑、不静默合并、不二次重切）。ratios 在此复核（非负、和为 1）。

    split.yaml 生成后即快照：raw/参数集合后续变化不保证旧成员划分不变，
    旧训练 run 绑定其 run 内保存的 split 副本。
    """
    ratios = (config.ratios.train, config.ratios.val, config.ratios.test)
    if (
        any(not math.isfinite(r) or r < 0 for r in ratios)
        or abs(sum(ratios) - 1.0) > _RATIO_TOLERANCE
    ):
        _fail("split.ratios", f"必须非负且和为 1（容差 {_RATIO_TOLERANCE}）(got {ratios})")
    ordered = sorted(psids)
    if len(set(ordered)) != len(ordered):
        _fail("psids", f"重复的 psid: {ordered}")
    if not ordered:
        _fail("psids", "所选参数组列表为空")

    n = len(ordered)
    permuted = [str(p) for p in np.random.default_rng(config.seed).permutation(ordered)]
    floors = [math.floor(n * r) for r in ratios]
    remainders = [n * r - math.floor(n * r) for r in ratios]
    # 余数降序；并列时 index 顺序即 train→val→test 优先。
    bonus_order = sorted(range(3), key=lambda i: (-remainders[i], i))
    counts = floors[:]
    for group_index in bonus_order[: n - sum(floors)]:
        counts[group_index] += 1

    mins = (config.min_per_split.train, config.min_per_split.val, config.min_per_split.test)
    deficits = [
        f"{_SPLIT_NAMES[i]} 需要 {mins[i]}, 实际 {counts[i]}"
        for i in range(3)
        if counts[i] < mins[i]
    ]
    if deficits:
        _fail("split.min_per_split", f"样本不足: {'; '.join(deficits)}（不硬凑、不重切）")

    bounds = (counts[0], counts[0] + counts[1])
    return SplitDefinition(
        seed=config.seed,
        ratios=config.ratios,
        train=tuple(permuted[: bounds[0]]),
        val=tuple(permuted[bounds[0] : bounds[1]]),
        test=tuple(permuted[bounds[1] :]),
    )


def load_dataset_meta(samples_dir: Path, *, meta_path: Path | None = None) -> DatasetMeta:
    """读取 dataset_meta.yaml（schema 见模块 docstring）。

    Args:
        samples_dir: 样本目录（缺省 meta 位置 ``<samples_dir>/dataset_meta.yaml``）。
        meta_path: 显式 meta 文件路径（如评估侧 SHA 已核对的 ckpt 锚定
            副本）；None = 缺省位置。解析走本函数同一 schema/校验，不复制。
    """
    source = meta_path if meta_path is not None else samples_dir / "dataset_meta.yaml"
    raw = _load_yaml(source, "dataset_meta")
    _check_keys(raw, "dataset_meta", _META_REQUIRED_KEYS | _META_OPTIONAL_KEYS, _META_REQUIRED_KEYS)
    pulse_order = _parse_pulse_id_list(raw["pulse_order"], "dataset_meta.pulse_order")
    labels = _parse_labels(raw["labels"])
    members = _parse_pulse_id_list(raw["members"], "dataset_meta.members")
    return DatasetMeta(
        dataset_name=_require_safe_path_segment(raw["dataset_name"], "dataset_meta.dataset_name"),
        pulse_order=pulse_order,
        n_time_steps=_require_positive_int(raw["n_time_steps"], "dataset_meta.n_time_steps"),
        labels=labels,
        members=members,
        protocol=raw.get("protocol"),
        config_sha256=_parse_str_mapping(raw.get("config_sha256"), "dataset_meta.config_sha256"),
        source_index_relpaths=_parse_str_mapping(
            raw.get("source_index_relpaths"), "dataset_meta.source_index_relpaths"
        ),
        generated_at=raw.get("generated_at"),
    )


def load_split(
    samples_dir: Path,
    meta: DatasetMeta,
    *,
    split_path: Path | None = None,
) -> SplitDefinition:
    """读取 split.yaml 并校验（成员边界见模块 docstring）。

    Args:
        samples_dir: 真实 npz 所在目录（成员 npz 存在性始终在此校验）。
        meta: 数据集清单（并集须恰等于 ``meta.members``）。
        split_path: 显式 split.yaml 路径（如训练 run 内保存的权威副本）；
            None = ``<samples_dir>/split.yaml``。校验走本函数同一路径，
            不复制逻辑；npz 存在性仍针对 ``samples_dir``。

    校验：三组互斥、并集恰等于 ``meta.members``；每个成员的 npz 存在。
    """
    source = split_path if split_path is not None else samples_dir / "split.yaml"
    raw = _load_yaml(source, "split")
    _check_keys(raw, "split", _SPLIT_YAML_KEYS, _SPLIT_YAML_KEYS)
    ratios_raw = _require_mapping(raw["ratios"], "split.ratios")
    _check_keys(ratios_raw, "split.ratios", _RATIO_KEYS, _RATIO_KEYS)
    ratios = SplitRatios(
        train=_require_non_negative_number(ratios_raw["train"], "split.ratios.train"),
        val=_require_non_negative_number(ratios_raw["val"], "split.ratios.val"),
        test=_require_non_negative_number(ratios_raw["test"], "split.ratios.test"),
    )
    members = {name: _parse_split_members(raw[name], f"split.{name}") for name in _SPLIT_NAMES}
    train_set, val_set, test_set = (set(members[name]) for name in _SPLIT_NAMES)
    overlap = sorted((train_set & val_set) | (train_set & test_set) | (val_set & test_set))
    if overlap:
        _fail("split", f"三组不互斥: {overlap}")
    union = train_set | val_set | test_set
    meta_members = set(meta.members)
    if union != meta_members:
        _fail(
            "split",
            f"并集与清单成员不一致: 多出 {sorted(union - meta_members)}, "
            f"缺失 {sorted(meta_members - union)}",
        )
    for name in _SPLIT_NAMES:
        for psid in members[name]:
            npz_path = samples_dir / f"{psid}.npz"
            if not npz_path.is_file():
                _fail(f"split.{name}", f"成员 npz 不存在: {npz_path}")
    return SplitDefinition(
        seed=_require_non_negative_int(raw["seed"], "split.seed"),
        ratios=ratios,
        train=members["train"],
        val=members["val"],
        test=members["test"],
    )


def build_datasets(
    samples_dir: Path,
    meta: DatasetMeta,
    split: SplitDefinition,
) -> tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
    """按 split 显式成员构造 (train, val, test) 三个 Dataset。"""
    return (
        TrajectoryDataset(samples_dir, meta, split.train),
        TrajectoryDataset(samples_dir, meta, split.val),
        TrajectoryDataset(samples_dir, meta, split.test),
    )


def load_sample(samples_dir: Path, psid: str, contract: InputContract | None = None) -> Sample:
    """加载单个成员 npz；``contract`` 非 None 时做输入契约校验。

    契约校验：``x.shape == contract.input_shape``（P/T/通道数）、
    ``pulse_ids == contract.pulse_order``、``t_s`` 与契约**逐位相等**
    （npz roundtrip 对 float64 是精确的，任何不符都是真实数据集变化，
    因此不做容差放行）。不符 → DataError 含路径。
    """
    npz_path = samples_dir / f"{_require_safe_path_segment(psid, 'psid')}.npz"
    if not npz_path.is_file():
        _fail("npz", f"文件不存在: {npz_path}")
    return _read_sample_npz(npz_path, psid, contract)


def _read_sample_npz(npz_path: Path, psid: str, contract: InputContract | None) -> Sample:
    """读 npz 并按 schema/契约校验（``allow_pickle=False``）。"""
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"x", "y", "parameter_set_id", "t_s", "pulse_ids"}
        missing = required.difference(archive.files)
        if missing:
            _fail(f"{npz_path}:keys", f"缺失数组 {sorted(missing)}")
        x = archive["x"]
        y = archive["y"]
        t_s = archive["t_s"]
        pulse_ids = archive["pulse_ids"]
        psid_scalar = archive["parameter_set_id"]
    _fail_field = f"{npz_path}"
    if x.dtype != np.dtype(X_DTYPE) or x.ndim != 3 or x.shape[2] != 3:
        _fail(_fail_field, f"x 须为 {X_DTYPE} [P,T,3] (got dtype={x.dtype}, shape={x.shape})")
    if y.dtype != np.dtype(Y_DTYPE) or y.shape != (2,):
        _fail(_fail_field, f"y 须为 {Y_DTYPE} [2] (got dtype={y.dtype}, shape={y.shape})")
    if t_s.dtype != np.dtype(T_S_DTYPE) or t_s.ndim != 1:
        _fail(_fail_field, f"t_s 须为 {T_S_DTYPE} [T] (got dtype={t_s.dtype}, shape={t_s.shape})")
    if pulse_ids.dtype.kind != "U" or pulse_ids.ndim != 1:
        _fail(_fail_field, f"pulse_ids 须为 unicode [P] (got dtype={pulse_ids.dtype})")
    if psid_scalar.shape != () or psid_scalar.dtype.kind != "U":
        _fail(_fail_field, f"parameter_set_id 须为 unicode 标量 (got {psid_scalar!r})")
    if str(psid_scalar.item()) != psid:
        _fail(_fail_field, f"parameter_set_id {str(psid_scalar.item())!r} 与请求 {psid!r} 不符")
    n_pulses, n_time_steps, _ = x.shape
    if n_pulses < 1 or len(pulse_ids) != n_pulses:
        _fail(_fail_field, f"pulse_ids 长度 {len(pulse_ids)} != P {n_pulses}")
    if len(t_s) != n_time_steps:
        _fail(_fail_field, f"t_s 长度 {len(t_s)} != T {n_time_steps}")
    ids = tuple(str(v) for v in pulse_ids.tolist())
    if len(set(ids)) != len(ids):
        _fail(_fail_field, f"重复的 pulse_id: {ids}")
    sample = Sample(parameter_set_id=psid, x=x, y=y, t_s=t_s, pulse_ids=ids)
    if contract is not None:
        _check_sample_contract(sample, contract, _fail_field)
    return sample


def _check_sample_contract(sample: Sample, contract: InputContract, context: str) -> None:
    """输入契约校验（加载期与内存缓存校验共用的唯一实现）。

    校验 ``x.shape == contract.input_shape``（P/T/通道数）、
    ``pulse_ids == contract.pulse_order``、``t_s`` 逐位相等（npz roundtrip
    对 float64 精确，任何不符都是真实数据集变化，不做容差放行）。
    """
    if sample.x.shape != contract.input_shape:
        _fail(context, f"x 形状 {sample.x.shape} 与契约 {contract.input_shape} 不符")
    if sample.pulse_ids != tuple(contract.pulse_order):
        _fail(context, f"pulse 顺序 {sample.pulse_ids} 与契约 {list(contract.pulse_order)} 不符")
    if not np.array_equal(sample.t_s, contract.t_s):
        _fail(context, "t_s 与契约不一致（须逐位相等）")


def write_prepared_dataset(
    samples_dir: Path,
    samples: Sequence[Sample],
    meta: DatasetMeta,
    split: SplitDefinition,
) -> None:
    """写出全部生成物：npz + dataset_meta.yaml + split.yaml。

    步骤：写前预检任一目标（各 ``<psid>.npz``、dataset_meta.yaml、
    split.yaml）已存在 → 在写任何样本之前报错拒绝（不隐式覆盖）；随后
    按序写出（逐组 ``np.savez_compressed`` → dataset_meta.yaml →
    split.yaml）。无事务：中途失败可能留下部分产物，下次运行因预检拒绝
    覆盖，须人工清理后重跑。
    """
    if not samples:
        _fail("samples", "样本列表为空")
    if len({sample.parameter_set_id for sample in samples}) != len(samples):
        _fail("samples", "重复的 psid")
    targets = [samples_dir / f"{sample.parameter_set_id}.npz" for sample in samples]
    targets += [samples_dir / "dataset_meta.yaml", samples_dir / "split.yaml"]
    existing = [target for target in targets if target.exists()]
    if existing:
        _fail("precheck", f"目标已存在，拒绝覆盖: {existing}")
    samples_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        arrays: dict[str, Any] = _npz_arrays(sample)
        np.savez_compressed(samples_dir / f"{sample.parameter_set_id}.npz", **arrays)
    (samples_dir / "dataset_meta.yaml").write_text(
        yaml.safe_dump(_meta_mapping(meta), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (samples_dir / "split.yaml").write_text(
        yaml.safe_dump(_split_mapping(split), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _npz_arrays(sample: Sample) -> dict[str, np.ndarray]:
    """npz 数组集（dtype 契约见模块 docstring；绝无 object 数组）。"""
    return {
        "x": np.ascontiguousarray(sample.x, dtype=np.float32),
        "y": np.ascontiguousarray(sample.y, dtype=np.float32),
        "parameter_set_id": np.array(sample.parameter_set_id, dtype=PSID_DTYPE),
        "t_s": np.ascontiguousarray(sample.t_s, dtype=np.float64),
        "pulse_ids": np.array(sample.pulse_ids, dtype=np.str_),
    }


def _meta_mapping(meta: DatasetMeta) -> dict[str, Any]:
    """DatasetMeta → YAML 映射（schema 见模块 docstring；键序确定）。"""
    return {
        "dataset_name": meta.dataset_name,
        "generated_at": meta.generated_at,
        "n_time_steps": meta.n_time_steps,
        "pulse_order": list(meta.pulse_order),
        "protocol": dict(meta.protocol) if meta.protocol is not None else None,
        "labels": {psid: list(values) for psid, values in sorted(meta.labels.items())},
        "members": list(meta.members),
        "source_index_relpaths": (
            dict(meta.source_index_relpaths) if meta.source_index_relpaths is not None else None
        ),
        "config_sha256": dict(meta.config_sha256) if meta.config_sha256 is not None else None,
    }


def _split_mapping(split: SplitDefinition) -> dict[str, Any]:
    """SplitDefinition → YAML 映射（schema 见模块 docstring）。"""
    return {
        "seed": split.seed,
        "ratios": {
            "train": split.ratios.train,
            "val": split.ratios.val,
            "test": split.ratios.test,
        },
        "train": list(split.train),
        "val": list(split.val),
        "test": list(split.test),
    }


def _parse_pulse_id_list(value: object, field: str) -> tuple[str, ...]:
    """非空、元素为安全路径段且无重复的 pulse/psid 列表。"""
    if not isinstance(value, list) or not value:
        _fail(field, f"必须为非空字符串列表 (got {value!r})")
    ids = tuple(_require_safe_path_segment(item, f"{field}[{i}]") for i, item in enumerate(value))
    if len(set(ids)) != len(ids):
        _fail(field, f"重复的 id: {ids}")
    return ids


def _parse_split_members(value: object, field: str) -> tuple[str, ...]:
    """split 成员列表：**允许空组**（min_per_split 可为 0）；元素仍须为安全
    路径段且组内无重复。互斥/并集校验由 load_split 负责。
    """
    if not isinstance(value, list):
        _fail(field, f"必须为字符串列表 (got {value!r})")
    ids = tuple(_require_safe_path_segment(item, f"{field}[{i}]") for i, item in enumerate(value))
    if len(set(ids)) != len(ids):
        _fail(field, f"重复的 id: {ids}")
    return ids


def _parse_labels(value: object) -> dict[str, tuple[float, float]]:
    """labels：psid → [alpha, ku_j_per_m3]（两个有限数）。"""
    if not isinstance(value, dict) or not value:
        _fail("dataset_meta.labels", f"必须为非空映射 (got {value!r})")
    labels: dict[str, tuple[float, float]] = {}
    for psid, pair in value.items():
        _require_safe_path_segment(psid, f"dataset_meta.labels[{psid!r}] key")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            _fail(f"dataset_meta.labels[{psid!r}]", f"必须为 [alpha, ku] (got {pair!r})")
        alpha = _require_finite(pair[0], f"dataset_meta.labels[{psid!r}].alpha")
        ku = _require_finite(pair[1], f"dataset_meta.labels[{psid!r}].ku_j_per_m3")
        labels[psid] = (alpha, ku)
    return labels


def _parse_str_mapping(value: object, field: str) -> dict[str, str] | None:
    """可选的 psid → str 映射（溯源字段）。"""
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail(field, f"必须为映射或 null (got {value!r})")
    return {str(key): str(item) for key, item in value.items()}
