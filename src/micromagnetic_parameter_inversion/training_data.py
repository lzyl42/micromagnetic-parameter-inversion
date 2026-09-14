"""Training-sample data layer: npz schema, protocol snapshot, splits, and Dataset.

Core contracts:

- npz schema (one ``<psid>.npz`` per parameter set): ``x: float32 [P, T, 3]``
  (stacked in frozen pulse order, channel order mx,my,mz), ``y: float32 [2]``
  (alpha, ku_j_per_m3 in raw physical units), ``parameter_set_id: <U16``
  scalar (source psid, split unit), ``t_s: float64 [T]``, ``pulse_ids: <U… [P]``;
  written with ``np.savez_compressed`` (never object arrays), always read with
  ``np.load(..., allow_pickle=False)``.
- Membership boundary: Dataset/split are constructed and validated only from
  explicit members (the three splits are disjoint and their union equals the
  manifest members); no intersection filtering, no scanning for stale npz
  leftovers in the directory, no runtime re-splitting.
- Performs only read-required checks and contract checks: no physical QC, no
  per-group protocol review.
- Protocol snapshot: the config.yaml of the first psid in ascending order;
  safe_load extracts only recording/pulses (older snapshots without numerics
  are valid); mT→T conversion is ×1e-3 per the mumax3_config schema fields.

dataset_meta.yaml serialization schema (written by ``write_prepared_dataset``,
read back by ``load_dataset_meta``; optional null keys use defaults)::

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

split.yaml serialization schema::

    seed: int
    ratios: {train: float, val: float, test: float}
    train: [psid, ...]
    val: [psid, ...]
    test: [psid, ...]

Error types: schema value/field violations raise ConfigError (reusing the
mumax3_config validators); layout/membership/contract violations raise
DataError (messages include the full path).

Dependency direction: training_config → (this module) → preprocessing/training.
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

# npz array dtype contracts.
X_DTYPE = "float32"  # x: [P, T, 3], raw time-domain trajectory, channel order (mx, my, mz)
Y_DTYPE = "float32"  # y: [2] = (alpha, ku_j_per_m3), raw physical units
T_S_DTYPE = "float64"  # t_s: [T], actual time grid (first trajectory's time column)
PSID_DTYPE = "<U16"  # parameter_set_id scalar: first 16 chars of SHA-256

# Trajectory CSV layout contract (verified on the generation side; the header is
# checked by the generation side's parse_table): here we only take columns by
# position and run the required shape checks.
TRAJECTORY_USECOLS = (2, 3, 4)  # take only m_x, m_y, m_z (sample_index/t_s stay out of x)
T_S_USECOLS = (1,)  # t_s column of the first trajectory

# index.csv fixed columns (verified in mumax3_pipeline.py).
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

# Schema required by the protocol snapshot (aligned with mumax3_config; the
# numerics section is not read).
_RECORDING_KEYS = frozenset({"sample_interval_s", "sample_count"})
_PULSE_KEYS = frozenset({"pulse_id", "b_ext_amplitude_mT", "direction", "duration_s"})

# dataset_meta.yaml / split.yaml schema.
_META_REQUIRED_KEYS = frozenset(
    {"dataset_name", "pulse_order", "n_time_steps", "labels", "members"}
)
_META_OPTIONAL_KEYS = frozenset(
    {"protocol", "config_sha256", "source_index_relpaths", "generated_at"}
)
_SPLIT_YAML_KEYS = frozenset({"seed", "ratios", "train", "val", "test"})
_RATIO_KEYS = frozenset({"train", "val", "test"})
_SPLIT_NAMES = ("train", "val", "test")
# Tolerance for the ratios sum (consistent with training_config).
_RATIO_TOLERANCE = 1e-9

# Dataset element type: unnormalized raw sample (x float32[P,T,3], y float32[2], psid);
# also the DataLoader batch item (preprocessing applied by the training/eval side).
type SampleItem = tuple[Tensor, Tensor, str]


class DataError(ValueError):
    """Sample/manifest/split layout or membership contract violation (full path included)."""


def _fail(field: str, problem: str) -> NoReturn:
    """Raise a DataError carrying the field path to locate the exact file/field."""
    raise DataError(f"{field}: {problem}")


def _check_keys(
    mapping: dict[Any, Any], field: str, known: frozenset[str], required: frozenset[str]
) -> None:
    """Strict schema: reject unknown fields; only the ``required`` subset is mandatory."""
    unknown = sorted((key for key in mapping if key not in known), key=repr)
    if unknown:
        _fail(field, f"unknown fields {unknown!r}; allowed fields {sorted(known)}")
    missing = sorted(required.difference(mapping))
    if missing:
        _fail(field, f"missing required fields {missing!r}")


def _load_yaml(path: Path, field: str) -> dict[Any, Any]:
    """safe_load + duplicate-key rejection; a non-mapping is a DataError."""
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        _fail(field, f"failed to parse YAML ({path}): {exc}")
    except OSError as exc:
        _fail(field, f"failed to read ({path}): {exc}")
    return _require_mapping(raw, field)


@dataclass(frozen=True, eq=False)
class InputContract:
    """Input contract: carried by the ckpt; evaluate validates each npz against it
    (contract validation, not physical QC).

    Contains ndarray fields and uses ``eq=False`` to avoid element-wise comparison
    semantics; consistency is checked explicitly by field value (pulse order / T /
    shape / t_s bitwise equality).
    """

    pulse_order: tuple[str, ...]  # frozen pulse_id sequence (P dimension order of x)
    n_time_steps: int  # T: time steps per pulse (frozen sample_count)
    t_s: np.ndarray  # float64 [T], actual time grid
    n_channels: int = 3  # number of magnetization components, always 3
    component_order: tuple[str, ...] = ("mx", "my", "mz")  # channel order

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """``(P, T, 3)``: shape argument for MLPRegressor(input_shape=...)."""
        return (len(self.pulse_order), self.n_time_steps, self.n_channels)


@dataclass(frozen=True, eq=False)
class Sample:
    """Normalized sample of one parameter set (one npz in memory; ndarray fields, eq=False)."""

    parameter_set_id: str  # source psid (split unit; determined only by (alpha, Ku))
    x: np.ndarray  # float32 [P, T, 3], stacked in frozen pulse order
    y: np.ndarray  # float32 [2] = (alpha, ku_j_per_m3), raw units
    t_s: np.ndarray  # float64 [T]
    pulse_ids: tuple[str, ...]  # [P], same order as the P dimension of x


@dataclass(frozen=True)
class ProtocolSummary:
    """Protocol summary frozen from the first config.yaml snapshot (prepare side reads only)."""

    pulse_order: tuple[str, ...]  # frozen pulse order (shared by all npz)
    sample_interval_s: float  # recording.sample_interval_s, s
    n_time_steps: int  # T = recording.sample_count
    # pulse_id → {b_ext_amplitude_t (converted to T via schema ×1e-3), direction,
    # duration_s}.
    pulses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMeta:
    """In-memory representation of dataset_meta.yaml (prepare output; see module docstring)."""

    dataset_name: str
    pulse_order: tuple[str, ...]  # frozen pulse order
    n_time_steps: int  # T = frozen sample_count
    labels: Mapping[str, tuple[float, float]]  # psid → (alpha, ku_j_per_m3)
    members: tuple[str, ...]  # manifest member psids (the split union must equal exactly this)
    # Protocol summary: {"sample_interval_s": float, "pulses": {pid: {...}}}.
    protocol: Mapping[str, Any] | None = None
    # Lightweight provenance below (optional; filled when writing meta).
    config_sha256: Mapping[str, str] | None = None  # psid → config.yaml sha256
    source_index_relpaths: Mapping[str, str] | None = None  # psid → source index relative path
    generated_at: str | None = None  # ISO-8601 UTC generation time


@dataclass(frozen=True)
class SplitDefinition:
    """In-memory representation of split.yaml; split unit = parameter_set_id (grouped by ID)."""

    seed: int
    ratios: SplitRatios
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def members_of(self, split_name: str) -> tuple[str, ...]:
        """Return members by name (train/val/test; an unknown name raises)."""
        members: dict[str, tuple[str, ...]] = dict(
            zip(_SPLIT_NAMES, (self.train, self.val, self.test), strict=True)
        )
        try:
            return members[split_name]
        except KeyError:
            msg = f"unknown split name {split_name!r}; allowed {list(_SPLIT_NAMES)}"
            raise DataError(msg) from None


def sha256_file(path: Path) -> str:
    """File sha256 (lowercase hex; shared by meta provenance and split/ckpt binding)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrajectoryDataset(Dataset[SampleItem]):
    """Parameter-set-level dataset built from explicit split members.

    Element type ``SampleItem``: raw, unnormalized; ``__getitem__`` returns
    ``(x_tensor float32 CPU, y_tensor float32 CPU, psid)``, and normalization is
    applied by the orchestration layer in preprocessing. Only npz files listed in
    the member manifest are loaded: stale npz files left in the directory (not in
    the manifest) are not loaded. Each npz is read from disk **exactly once** and
    cached as ``Sample`` (both the ``samples`` property and ``validate_contract``
    are pure in-memory access).
    """

    def __init__(
        self,
        samples_dir: Path,
        meta: DatasetMeta,
        psids: Sequence[str],
        *,
        contract: InputContract | None = None,
    ) -> None:
        """Load and validate all npz files for the explicit member list (members must
        exist in meta.members).

        Args:
            samples_dir: directory containing the npz files.
            meta: dataset manifest (baseline for member validation).
            psids: explicit member list of this split.
            contract: when not None, validate the input contract per sample at load
                time (validation completes during the single disk read, with no
                second read).
        """
        members = set(meta.members)
        unknown = [psid for psid in psids if psid not in members]
        if unknown:
            _fail("psids", f"members not present in the dataset_meta manifest: {unknown}")
        if len(set(psids)) != len(psids):
            _fail("psids", f"duplicate psid: {list(psids)}")
        self._samples_dir = samples_dir
        self._meta = meta
        self._psids = tuple(psids)
        self._samples = tuple(
            _read_sample_npz(samples_dir / f"{psid}.npz", psid, contract) for psid in psids
        )

    @property
    def samples(self) -> tuple[Sample, ...]:
        """Cached ``Sample`` tuple (in member order; access does not trigger disk reads)."""
        return self._samples

    def validate_contract(self, contract: InputContract) -> None:
        """Validate the input contract for all cached samples (pure in-memory, no disk).

        Used for the "validate the remaining members after freezing the contract
        from the first cached sample" scenario; uses the same check as
        ``load_sample``/load-time validation (``_check_sample_contract``).
        """
        for sample in self._samples:
            _check_sample_contract(sample, contract, f"psid={sample.parameter_set_id}")

    def __len__(self) -> int:
        """Number of members."""
        return len(self._samples)

    def __getitem__(self, index: int) -> SampleItem:
        """Return raw ``(x_tensor float32[P,T,3], y_tensor float32[2], psid)``."""
        sample = self._samples[index]
        return torch.from_numpy(sample.x), torch.from_numpy(sample.y), sample.parameter_set_id


def load_protocol_snapshot(
    raw_dataset_dir: Path,
    psids: Sequence[str],
    pulse_order: Sequence[str] | None,
) -> ProtocolSummary:
    """Read the protocol snapshot: the first config.yaml of the selected psids in
    ascending order.

    safe_load extracts only recording (sample_interval_s, sample_count → T) and
    pulses; older snapshots without numerics are valid; mT→T conversion follows
    the schema field b_ext_amplitude_mT ×1e-3. When ``pulse_order`` is not None it
    must be a duplicate-free permutation of the snapshot pulse set (missing/extra/
    duplicate entries all raise); None = declaration order in the snapshot.

    Raises:
        DataError: snapshot missing / structure violation / order mismatch.
        ConfigError: numeric schema violation (reusing the mumax3_config validators).
    """
    if not psids:
        _fail("psids", "selected parameter set list is empty")
    ordered = sorted(psids)
    config_path = raw_dataset_dir / _require_safe_path_segment(ordered[0], "psid") / "config.yaml"
    if not config_path.is_file():
        _fail("config.yaml", f"protocol snapshot missing: {config_path}")
    snapshot = _load_yaml(config_path, f"{config_path}:root")

    recording_raw = snapshot.get("recording")
    if not isinstance(recording_raw, dict):
        _fail(f"{config_path}:recording", "snapshot is missing the recording section")
    _check_keys(recording_raw, f"{config_path}:recording", _RECORDING_KEYS, _RECORDING_KEYS)
    sample_interval_s = _require_positive_number(
        recording_raw["sample_interval_s"], f"{config_path}:recording.sample_interval_s"
    )
    n_time_steps = _require_positive_int(
        recording_raw["sample_count"], f"{config_path}:recording.sample_count"
    )

    pulses_raw = snapshot.get("pulses")
    if not isinstance(pulses_raw, list) or not pulses_raw:
        _fail(f"{config_path}:pulses", "must be a non-empty pulse list")
    snapshot_order: list[str] = []
    pulses: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(pulses_raw):
        pulse_field = f"{config_path}:pulses[{index}]"
        pulse_raw = _require_mapping(item, pulse_field)
        _check_keys(pulse_raw, pulse_field, _PULSE_KEYS, _PULSE_KEYS)
        pulse_id = _require_safe_path_segment(pulse_raw["pulse_id"], f"{pulse_field}.pulse_id")
        if pulse_id in pulses:
            _fail(f"{pulse_field}.pulse_id", f"duplicate pulse_id {pulse_id!r}")
        amplitude_millitesla = _require_non_negative_number(
            pulse_raw["b_ext_amplitude_mT"], f"{pulse_field}.b_ext_amplitude_mT"
        )
        direction = _require_vector3(pulse_raw["direction"], f"{pulse_field}.direction")
        duration_s = _require_positive_number(pulse_raw["duration_s"], f"{pulse_field}.duration_s")
        snapshot_order.append(pulse_id)
        # Unit boundary: mT exists only in YAML; stored here as T via ×1e-3 (same as mumax3_config).
        pulses[pulse_id] = {
            "b_ext_amplitude_t": amplitude_millitesla * 1e-3,
            "direction": direction,
            "duration_s": duration_s,
        }

    frozen_order: tuple[str, ...]
    if pulse_order is not None:
        if len(set(pulse_order)) != len(pulse_order):
            _fail("data.pulse_order", f"duplicate pulse_id: {list(pulse_order)}")
        if set(pulse_order) != set(snapshot_order):
            _fail(
                "data.pulse_order",
                f"must be a permutation of the snapshot pulse set; snapshot "
                f"{sorted(snapshot_order)}, config {sorted(pulse_order)}",
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
    """Read one parameter set and assemble ``(Sample, (alpha, ku_j_per_m3))``.

    Labels are the original index.csv values (float64 precision, for
    dataset_meta provenance); the ``y`` in the npz is their float32 form.

    - index.csv: must exist and contain all fixed columns; the row pulse_id set
      must exactly match ``pulse_order`` (missing/extra/duplicate all raise);
      (alpha, ku_j_per_m3) must be consistent within the group.
    - trajectory.csv: ``trajectory_path`` must be a relative path inside the psid
      directory and must exist; magnetization is read with
      ``usecols=(2,3,4), ndmin=2`` → shape ``(T,3)`` (T = ``n_time_steps``);
      t_s is taken from the ``pulse_order[0]`` trajectory (``ndmin=1``).
    - No physical QC: the magnetization values themselves are not checked (the
      generation side's parse_table already does).
    """
    psid_dir = raw_dataset_dir / _require_safe_path_segment(psid, "parameter_set_id")
    if not psid_dir.is_dir():
        _fail("parameter_set_dir", f"parameter set directory does not exist: {psid_dir}")

    rows = _read_index_rows(psid_dir / "index.csv")
    pulse_field = f"{psid_dir / 'index.csv'}:pulse_id"
    row_by_pulse: dict[str, dict[str, str]] = {}
    for row in rows:
        pulse_id = (row.get("pulse_id") or "").strip()
        if not pulse_id:
            continue
        if pulse_id in row_by_pulse:
            _fail(pulse_field, f"duplicate pulse_id {pulse_id!r}")
        if pulse_id not in set(pulse_order):
            _fail(pulse_field, f"extra pulse {pulse_id!r} (frozen order {list(pulse_order)})")
        row_by_pulse[pulse_id] = row
    missing = [pid for pid in pulse_order if pid not in row_by_pulse]
    if missing:
        _fail(pulse_field, f"missing pulse {missing} (frozen order {list(pulse_order)})")

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
    assert t_s is not None  # pulse_order is non-empty (guaranteed by the protocol snapshot)
    x = np.stack(magnetization).astype(np.float32)
    y = np.array([alpha, ku], dtype=np.float32)
    sample = Sample(parameter_set_id=psid, x=x, y=y, t_s=t_s, pulse_ids=tuple(pulse_order))
    return sample, (alpha, ku)


def _read_index_rows(index_path: Path) -> list[dict[str, str]]:
    """Read index.csv; require existence and all fixed columns (extra columns tolerated).

    Returns the row dicts as a list.
    """
    if not index_path.is_file():
        _fail("index.csv", f"file does not exist: {index_path}")
    with index_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [column for column in INDEX_COLUMNS if column not in fieldnames]
        if missing:
            _fail(f"{index_path}:header", f"missing fixed columns {missing}")
        return list(reader)


def _group_labels(psid: str, rows: Sequence[Mapping[str, str]]) -> tuple[float, float]:
    """(alpha, ku_j_per_m3): parsed from index rows; all rows in a group must agree."""
    alpha: float | None = None
    ku: float | None = None
    for row in rows:
        row_alpha = _parse_index_number(row["alpha"], f"{psid}:alpha")
        row_ku = _parse_index_number(row["ku_j_per_m3"], f"{psid}:ku_j_per_m3")
        if alpha is None:
            alpha, ku = row_alpha, row_ku
        elif alpha != row_alpha or ku != row_ku:
            _fail(
                f"{psid}:labels",
                f"label mismatch within group: ({alpha}, {ku}) vs ({row_alpha}, {row_ku})",
            )
    assert alpha is not None and ku is not None  # non-empty pulse_order guarantees at least one row
    return alpha, ku


def _parse_index_number(value: str, field: str) -> float:
    """index numeric column: must be finite (Ku may be 0/negative; alpha
    non-negativity is guaranteed by the generation side)."""
    try:
        number = float(value)
    except ValueError as exc:
        _fail(field, f"must be a number (got {value!r}): {exc}")
    if not math.isfinite(number):
        _fail(field, f"must be a finite number (got {value!r})")
    return number


def _resolve_trajectory_path(psid_dir: Path, row: Mapping[str, str]) -> Path:
    """trajectory_path: must be an existing relative path inside the psid directory
    (prevents path escape)."""
    relative = (row.get("trajectory_path") or "").strip()
    if not relative:
        _fail(f"{psid_dir / 'index.csv'}:trajectory_path", "trajectory_path is empty")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        _fail(
            f"{psid_dir / 'index.csv'}:trajectory_path",
            f"absolute paths are not accepted: {relative!r}",
        )
    candidate = psid_dir / relative_path
    try:
        candidate.resolve().relative_to(psid_dir.resolve())
    except ValueError:
        _fail(
            f"{psid_dir / 'index.csv'}:trajectory_path",
            f"path escapes the parameter set directory: {relative!r}",
        )
    if not candidate.is_file():
        _fail("trajectory.csv", f"file does not exist: {candidate}")
    return candidate


def _load_trajectory(
    path: Path,
    usecols: tuple[int, ...],
    ndmin: Literal[0, 1, 2],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Read the selected columns of trajectory.csv; ``ndmin`` prevents a single row
    or column from being squeezed; the shape is required to match."""
    try:
        array = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols, ndmin=ndmin)
    except (OSError, ValueError) as exc:
        _fail("trajectory.csv", f"failed to read ({path}): {exc}")
    if array.shape != expected_shape:
        _fail(
            "trajectory.csv",
            f"shape mismatch ({path}): got {array.shape}, expected {expected_shape}",
        )
    return array


def make_split(psids: Sequence[str], config: SplitConfig) -> SplitDefinition:
    """Deterministic split: largest remainder method, ties fixed train→val→test.

    Steps: deduplicate and sort psids → shuffle via
    ``np.random.default_rng(seed).permutation`` → each split first takes
    ``floor(n·ratio)``, then leftover seats are assigned one by one in descending
    fractional remainders (ties prioritized train→val→test) → check
    ``min_per_split`` and raise if any split is short (no padding, no silent
    merge, no re-splitting). Ratios are re-checked here (non-negative, sum to 1).

    split.yaml is a snapshot once written: later changes to the raw data or
    parameter set do not guarantee the old member split stays valid; an old
    training run binds the split copy stored inside its run directory.
    """
    ratios = (config.ratios.train, config.ratios.val, config.ratios.test)
    if (
        any(not math.isfinite(r) or r < 0 for r in ratios)
        or abs(sum(ratios) - 1.0) > _RATIO_TOLERANCE
    ):
        _fail(
            "split.ratios",
            f"ratios must be non-negative and sum to 1 (tolerance {_RATIO_TOLERANCE}) "
            f"(got {ratios})",
        )
    ordered = sorted(psids)
    if len(set(ordered)) != len(ordered):
        _fail("psids", f"duplicate psid: {ordered}")
    if not ordered:
        _fail("psids", "selected parameter set list is empty")

    n = len(ordered)
    permuted = [str(p) for p in np.random.default_rng(config.seed).permutation(ordered)]
    floors = [math.floor(n * r) for r in ratios]
    remainders = [n * r - math.floor(n * r) for r in ratios]
    # Descending remainders; on ties, index order gives train→val→test priority.
    bonus_order = sorted(range(3), key=lambda i: (-remainders[i], i))
    counts = floors[:]
    for group_index in bonus_order[: n - sum(floors)]:
        counts[group_index] += 1

    mins = (config.min_per_split.train, config.min_per_split.val, config.min_per_split.test)
    deficits = [
        f"{_SPLIT_NAMES[i]} requires {mins[i]}, actual {counts[i]}"
        for i in range(3)
        if counts[i] < mins[i]
    ]
    if deficits:
        _fail(
            "split.min_per_split",
            f"insufficient samples: {'; '.join(deficits)} (no padding, no re-splitting)",
        )

    bounds = (counts[0], counts[0] + counts[1])
    return SplitDefinition(
        seed=config.seed,
        ratios=config.ratios,
        train=tuple(permuted[: bounds[0]]),
        val=tuple(permuted[bounds[0] : bounds[1]]),
        test=tuple(permuted[bounds[1] :]),
    )


def load_dataset_meta(samples_dir: Path, *, meta_path: Path | None = None) -> DatasetMeta:
    """Read dataset_meta.yaml (schema in the module docstring).

    Args:
        samples_dir: samples directory (default meta location
            ``<samples_dir>/dataset_meta.yaml``).
        meta_path: explicit meta file path (e.g. the ckpt-anchored copy whose SHA
            the evaluation side has verified); None = default location. Parsing
            uses the same schema/validation as this function, without duplication.
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
    """Read and validate split.yaml (membership boundary in the module docstring).

    Args:
        samples_dir: directory containing the real npz files (member npz existence
            is always validated here).
        meta: dataset manifest (the union must equal ``meta.members`` exactly).
        split_path: explicit split.yaml path (e.g. the authoritative copy stored
            inside a training run); None = ``<samples_dir>/split.yaml``.
            Validation uses the same code path as this function, without
            duplication; npz existence is still resolved against ``samples_dir``.

    Validation: the three splits are disjoint and their union equals
    ``meta.members`` exactly; each member npz exists.
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
        _fail("split", f"the three splits are not disjoint: {overlap}")
    union = train_set | val_set | test_set
    meta_members = set(meta.members)
    if union != meta_members:
        _fail(
            "split",
            f"union does not match manifest members: extra {sorted(union - meta_members)}, "
            f"missing {sorted(meta_members - union)}",
        )
    for name in _SPLIT_NAMES:
        for psid in members[name]:
            npz_path = samples_dir / f"{psid}.npz"
            if not npz_path.is_file():
                _fail(f"split.{name}", f"member npz does not exist: {npz_path}")
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
    """Build the three Datasets (train, val, test) from the explicit split members."""
    return (
        TrajectoryDataset(samples_dir, meta, split.train),
        TrajectoryDataset(samples_dir, meta, split.val),
        TrajectoryDataset(samples_dir, meta, split.test),
    )


def load_sample(samples_dir: Path, psid: str, contract: InputContract | None = None) -> Sample:
    """Load a single member npz; when ``contract`` is not None, validate the input contract.

    Contract validation: ``x.shape == contract.input_shape`` (P/T/channel count),
    ``pulse_ids == contract.pulse_order``, and ``t_s`` **bitwise equal** to the
    contract (the npz roundtrip is exact for float64, so any difference is a real
    dataset change and no tolerance is allowed). A mismatch → DataError with the path.
    """
    npz_path = samples_dir / f"{_require_safe_path_segment(psid, 'psid')}.npz"
    if not npz_path.is_file():
        _fail("npz", f"file does not exist: {npz_path}")
    return _read_sample_npz(npz_path, psid, contract)


def _read_sample_npz(npz_path: Path, psid: str, contract: InputContract | None) -> Sample:
    """Read npz and validate schema/contract (``allow_pickle=False``)."""
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"x", "y", "parameter_set_id", "t_s", "pulse_ids"}
        missing = required.difference(archive.files)
        if missing:
            _fail(f"{npz_path}:keys", f"missing arrays {sorted(missing)}")
        x = archive["x"]
        y = archive["y"]
        t_s = archive["t_s"]
        pulse_ids = archive["pulse_ids"]
        psid_scalar = archive["parameter_set_id"]
    _fail_field = f"{npz_path}"
    if x.dtype != np.dtype(X_DTYPE) or x.ndim != 3 or x.shape[2] != 3:
        _fail(_fail_field, f"x must be {X_DTYPE} [P,T,3] (got dtype={x.dtype}, shape={x.shape})")
    if y.dtype != np.dtype(Y_DTYPE) or y.shape != (2,):
        _fail(_fail_field, f"y must be {Y_DTYPE} [2] (got dtype={y.dtype}, shape={y.shape})")
    if t_s.dtype != np.dtype(T_S_DTYPE) or t_s.ndim != 1:
        _fail(
            _fail_field, f"t_s must be {T_S_DTYPE} [T] (got dtype={t_s.dtype}, shape={t_s.shape})"
        )
    if pulse_ids.dtype.kind != "U" or pulse_ids.ndim != 1:
        _fail(_fail_field, f"pulse_ids must be unicode [P] (got dtype={pulse_ids.dtype})")
    if psid_scalar.shape != () or psid_scalar.dtype.kind != "U":
        _fail(_fail_field, f"parameter_set_id must be a unicode scalar (got {psid_scalar!r})")
    if str(psid_scalar.item()) != psid:
        _fail(
            _fail_field,
            f"parameter_set_id {str(psid_scalar.item())!r} does not match requested {psid!r}",
        )
    n_pulses, n_time_steps, _ = x.shape
    if n_pulses < 1 or len(pulse_ids) != n_pulses:
        _fail(_fail_field, f"pulse_ids length {len(pulse_ids)} != P {n_pulses}")
    if len(t_s) != n_time_steps:
        _fail(_fail_field, f"t_s length {len(t_s)} != T {n_time_steps}")
    ids = tuple(str(v) for v in pulse_ids.tolist())
    if len(set(ids)) != len(ids):
        _fail(_fail_field, f"duplicate pulse_id: {ids}")
    sample = Sample(parameter_set_id=psid, x=x, y=y, t_s=t_s, pulse_ids=ids)
    if contract is not None:
        _check_sample_contract(sample, contract, _fail_field)
    return sample


def _check_sample_contract(sample: Sample, contract: InputContract, context: str) -> None:
    """Input contract validation (single implementation for load-time and cached checks).

    Validates ``x.shape == contract.input_shape`` (P/T/channel count),
    ``pulse_ids == contract.pulse_order``, and ``t_s`` bitwise equality (the npz
    roundtrip is exact for float64, so any difference is a real dataset change
    and no tolerance is allowed).
    """
    if sample.x.shape != contract.input_shape:
        _fail(context, f"x shape {sample.x.shape} does not match contract {contract.input_shape}")
    if sample.pulse_ids != tuple(contract.pulse_order):
        _fail(
            context,
            f"pulse order {sample.pulse_ids} does not match contract {list(contract.pulse_order)}",
        )
    if not np.array_equal(sample.t_s, contract.t_s):
        _fail(context, "t_s mismatch with contract (bitwise equality required)")


def write_prepared_dataset(
    samples_dir: Path,
    samples: Sequence[Sample],
    meta: DatasetMeta,
    split: SplitDefinition,
) -> None:
    """Write all artifacts: npz + dataset_meta.yaml + split.yaml.

    Steps: pre-check before writing that no target (each ``<psid>.npz``,
    dataset_meta.yaml, split.yaml) already exists → refuse before writing any
    sample (no implicit overwrite); then write in order (per-group
    ``np.savez_compressed`` → dataset_meta.yaml → split.yaml). No transaction: a
    mid-way failure may leave partial artifacts; the next run refuses to overwrite
    due to the pre-check, so manual cleanup followed by a rerun is required.
    """
    if not samples:
        _fail("samples", "sample list is empty")
    if len({sample.parameter_set_id for sample in samples}) != len(samples):
        _fail("samples", "duplicate psid")
    targets = [samples_dir / f"{sample.parameter_set_id}.npz" for sample in samples]
    targets += [samples_dir / "dataset_meta.yaml", samples_dir / "split.yaml"]
    existing = [target for target in targets if target.exists()]
    if existing:
        _fail("precheck", f"target already exists, refusing to overwrite: {existing}")
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
    """npz array set (dtype contracts in the module docstring; never object arrays)."""
    return {
        "x": np.ascontiguousarray(sample.x, dtype=np.float32),
        "y": np.ascontiguousarray(sample.y, dtype=np.float32),
        "parameter_set_id": np.array(sample.parameter_set_id, dtype=PSID_DTYPE),
        "t_s": np.ascontiguousarray(sample.t_s, dtype=np.float64),
        "pulse_ids": np.array(sample.pulse_ids, dtype=np.str_),
    }


def _meta_mapping(meta: DatasetMeta) -> dict[str, Any]:
    """DatasetMeta → YAML mapping (schema in the module docstring; deterministic key order)."""
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
    """SplitDefinition → YAML mapping (schema in the module docstring)."""
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
    """Non-empty pulse/psid list whose elements are safe path segments with no duplicates."""
    if not isinstance(value, list) or not value:
        _fail(field, f"must be a non-empty string list (got {value!r})")
    ids = tuple(_require_safe_path_segment(item, f"{field}[{i}]") for i, item in enumerate(value))
    if len(set(ids)) != len(ids):
        _fail(field, f"duplicate id: {ids}")
    return ids


def _parse_split_members(value: object, field: str) -> tuple[str, ...]:
    """split member list: **empty groups are allowed** (min_per_split may be 0);
    elements must still be safe path segments with no duplicates within the group.
    Disjointness/union checks are handled by load_split.
    """
    if not isinstance(value, list):
        _fail(field, f"must be a string list (got {value!r})")
    ids = tuple(_require_safe_path_segment(item, f"{field}[{i}]") for i, item in enumerate(value))
    if len(set(ids)) != len(ids):
        _fail(field, f"duplicate id: {ids}")
    return ids


def _parse_labels(value: object) -> dict[str, tuple[float, float]]:
    """labels: psid → [alpha, ku_j_per_m3] (two finite numbers)."""
    if not isinstance(value, dict) or not value:
        _fail("dataset_meta.labels", f"must be a non-empty mapping (got {value!r})")
    labels: dict[str, tuple[float, float]] = {}
    for psid, pair in value.items():
        _require_safe_path_segment(psid, f"dataset_meta.labels[{psid!r}] key")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            _fail(f"dataset_meta.labels[{psid!r}]", f"must be [alpha, ku] (got {pair!r})")
        alpha = _require_finite(pair[0], f"dataset_meta.labels[{psid!r}].alpha")
        ku = _require_finite(pair[1], f"dataset_meta.labels[{psid!r}].ku_j_per_m3")
        labels[psid] = (alpha, ku)
    return labels


def _parse_str_mapping(value: object, field: str) -> dict[str, str] | None:
    """Optional psid → str mapping (provenance fields)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail(field, f"must be a mapping or null (got {value!r})")
    return {str(key): str(item) for key, item in value.items()}
