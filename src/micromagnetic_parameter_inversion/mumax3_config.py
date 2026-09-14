"""MuMax3 simulation config: frozen dataclasses, YAML loading, and derived quantities.

One YAML = the multi-pulse simulation input for one (alpha, Ku) parameter set; all
research values come from configs/experiments/*.yaml, and this module carries no
default values. load_config is the only validation and unit-conversion boundary;
afterwards the pipeline assumes a valid config and does not defend again.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import yaml

type Vector3 = tuple[float, float, float]
type Cells3 = tuple[int, int, int]

_TOP_LEVEL_KEYS = frozenset(
    {"dataset_name", "material", "geometry", "initial_m", "recording", "numerics", "pulses"}
)
_MATERIAL_KEYS = frozenset({"ms_a_per_m", "aex_j_per_m", "alpha", "ku_j_per_m3", "anisotropy_axis"})
_GEOMETRY_KEYS = frozenset({"size_m", "cells"})
_RECORDING_KEYS = frozenset({"sample_interval_s", "sample_count"})
_NUMERICS_KEYS = frozenset(
    {
        "edge_smooth",
        "solver",
        "max_err",
        "max_dt_s",
        "gamma_ll_rad_per_t_s",
        "relax_torque_threshold_t",
    }
)
_PULSE_KEYS = frozenset({"pulse_id", "b_ext_amplitude_mT", "direction", "duration_s"})

# Unit-vector norm tolerance: tolerates floating-point rounding in hand-written YAML
# values (e.g. 1/sqrt(3)).
_UNIT_VECTOR_TOLERANCE = 1e-6


class ConfigError(ValueError):
    """YAML config violates the schema/value constraints.

    Messages carry the field path (e.g. material.alpha).
    """


@dataclass(frozen=True)
class MaterialConfig:
    """SI parameters of one uniform material; alpha and Ku are the inversion targets."""

    ms_a_per_m: float  # Saturation magnetization, A/m
    aex_j_per_m: float  # Exchange stiffness constant, J/m
    alpha: float  # Gilbert damping coefficient, dimensionless, >= 0 (inversion target)
    ku_j_per_m3: float  # Uniaxial anisotropy constant, J/m^3; finite only (inversion target)
    anisotropy_axis: Vector3  # Easy-axis unit vector, dimensionless


@dataclass(frozen=True)
class GeometryConfig:
    """Bounding-box size and grid cell counts.

    cell_size is derived from size_m/cells (single source of truth).
    """

    size_m: Vector3  # Bounding-box size [x, y, z], m
    cells: Cells3  # Grid cell counts [nx, ny, nz]


@dataclass(frozen=True)
class RecordingConfig:
    """Fixed sampling of the free decay after field switch-off."""

    sample_interval_s: float  # Sampling interval, s
    sample_count: int  # Number of sample points (determines the trajectory row count)


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical protocol, explicitly rendered into both templates.

    Changing it changes the protocol.
    """

    edge_smooth: int  # EdgeSmooth, non-negative integer (0 = hard step boundary); before SetGeom
    solver: int  # Solver ID for SetSolver(...), positive integer
    max_err: float  # MaxErr, positive, dimensionless
    max_dt_s: float  # MaxDt, positive, s
    gamma_ll_rad_per_t_s: float  # GammaLL, positive, rad/(T*s)
    relax_torque_threshold_t: float  # RelaxTorqueThreshold, T; finite only (-1 = official default)


@dataclass(frozen=True)
class PulseConfig:
    """A single short rectangular pulse excitation (one independent run per entry)."""

    pulse_id: str  # Unique pulse identifier, safe single path segment
    b_ext_amplitude_t: float  # Pulse amplitude in runtime SI units, T (load_config converts, >= 0)
    direction: Vector3  # Pulse direction unit vector, dimensionless
    duration_s: float  # Pulse duration, s; B_ext is identically 0 after switch-off


@dataclass(frozen=True)
class SimulationConfig:
    """One config = all simulation inputs for one parameter set (aggregate root)."""

    # Safe single path segment; represents the fixed material/geometry/simulation protocol.
    dataset_name: str
    material: MaterialConfig
    geometry: GeometryConfig
    initial_m: Vector3  # Uniform initial-state direction, dimensionless unit vector
    recording: RecordingConfig
    numerics: NumericsConfig
    pulses: tuple[PulseConfig, ...]


def _fail(field: str, problem: str) -> NoReturn:
    """Raise a ConfigError carrying the field path, for locating the exact YAML field."""
    raise ConfigError(f"{field}: {problem}")


def _check_mapping_keys(mapping: dict[Any, Any], field: str, known: frozenset[str]) -> None:
    """Strict schema: reject unknown fields and missing required keys (applies at every level)."""
    unknown = sorted((key for key in mapping if key not in known), key=repr)
    if unknown:
        _fail(field, f"unknown fields {unknown!r}; allowed fields {sorted(known)}")
    missing = sorted(known.difference(mapping))
    if missing:
        _fail(field, f"missing required keys {missing!r}")


def _require_mapping(value: object, field: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        _fail(field, f"must be a mapping (got {value!r})")
    return value


def _require_finite(value: object, field: str) -> float:
    """Numeric component: accepts int/float, rejects bool masquerading as a number.

    Non-finite values are also rejected.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, f"must be a number (got {value!r})")
    number = float(value)
    if not math.isfinite(number):
        _fail(field, f"must be a finite number (got non-finite {value!r})")
    return number


def _require_positive_number(value: object, field: str) -> float:
    """Finite number that must be positive by semantics.

    0, negatives, nan/inf, and bool are all rejected.
    """
    number = _require_finite(value, field)
    if number <= 0:
        _fail(field, f"must be positive (got {value!r})")
    return number


def _require_positive_int(value: object, field: str) -> int:
    """Positive integer: rejects bool, floats (including integral floats such as 3.0).

    Non-positive values are also rejected.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(field, f"must be a positive integer (got {value!r})")
    return value


def _require_non_negative_int(value: object, field: str) -> int:
    """Non-negative integer (0 allowed): rejects bool, floats, and non-negativity violations."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(field, f"must be a non-negative integer (got {value!r})")
    return value


def _require_non_negative_number(value: object, field: str) -> float:
    """Finite number that must be non-negative by semantics.

    Negatives, nan/inf, and bool are all rejected.
    """
    number = _require_finite(value, field)
    if number < 0:
        _fail(field, f"must be non-negative (got {value!r})")
    return number


def _require_length3(value: object, field: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        _fail(field, f"must be a 3-component sequence (got {value!r})")
    if len(value) != 3:
        _fail(field, f"must have exactly 3 components (got {len(value)})")
    return value


def _require_vector3(
    value: object,
    field: str,
    component: Callable[[object, str], float] = _require_finite,
) -> Vector3:
    """Vector with exactly 3 finite components; component sets the extra constraint."""
    items = _require_length3(value, field)
    return (
        component(items[0], f"{field}[0]"),
        component(items[1], f"{field}[1]"),
        component(items[2], f"{field}[2]"),
    )


def _require_cells3(value: object, field: str) -> Cells3:
    items = _require_length3(value, field)
    return (
        _require_positive_int(items[0], f"{field}[0]"),
        _require_positive_int(items[1], f"{field}[1]"),
        _require_positive_int(items[2], f"{field}[2]"),
    )


def _require_unit_vector(value: object, field: str) -> Vector3:
    """Vector with exactly 3 finite components and norm 1 (implicitly a non-zero vector)."""
    vector = _require_vector3(value, field)
    norm = math.sqrt(sum(x * x for x in vector))
    if not math.isclose(norm, 1.0, rel_tol=_UNIT_VECTOR_TOLERANCE, abs_tol=_UNIT_VECTOR_TOLERANCE):
        _fail(field, f"must be a unit vector (got {vector!r}, |v| = {norm!r})")
    return vector


def _require_safe_path_segment(value: object, field: str) -> str:
    """Safe single path segment: non-empty, not "."/"..", no "/", "\\", or NUL.

    This prevents path escape.
    """
    if not isinstance(value, str) or not value:
        _fail(field, f"must be a non-empty string (got {value!r})")
    if value in {".", ".."}:
        _fail(field, f"{value!r} is not accepted as a path segment")
    if "/" in value or "\\" in value:
        _fail(field, f"must not contain '/' or '\\' (got {value!r})")
    if "\x00" in value:
        _fail(field, f"must not contain a NUL character (got {value!r})")
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate YAML keys.

    Research values (e.g. alpha) must not be silently overwritten.
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: list[object] = []
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if any(type(key) is type(other) and key == other for other in seen):
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate YAML key: {key!r}", key_node.start_mark
                )
            seen.append(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path: Path) -> SimulationConfig:
    """Read the YAML, run the single-entry validation, and build a SimulationConfig.

    Validation boundary (afterwards the pipeline assumes a valid config and does not
    defend again): strict schema at every level, rejecting missing/unknown fields,
    null, bool masquerading as a number, and non-finite values; positive/non-negative
    numbers and positive/non-negative integers are constrained per field semantics
    (Ku and relax_torque_threshold_t only need to be finite, allowing 0/negative and
    -1); 3D vectors must have exactly 3 finite components, and
    anisotropy_axis/initial_m/direction must be unit vectors (norm tolerance 1e-6);
    pulses must be non-empty with unique pulse_id; dataset_name/pulse_id must be safe
    single path segments.

    Unit boundary: b_ext_amplitude_mT is multiplied by 1e-3 here and stored as
    runtime T; mT never enters the runtime model.
    """
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: failed to parse YAML: {exc}") from exc

    root = _require_mapping(raw, str(path))
    _check_mapping_keys(root, str(path), _TOP_LEVEL_KEYS)

    dataset_name = _require_safe_path_segment(root["dataset_name"], "dataset_name")

    material_raw = _require_mapping(root["material"], "material")
    _check_mapping_keys(material_raw, "material", _MATERIAL_KEYS)
    material = MaterialConfig(
        ms_a_per_m=_require_positive_number(material_raw["ms_a_per_m"], "material.ms_a_per_m"),
        aex_j_per_m=_require_positive_number(material_raw["aex_j_per_m"], "material.aex_j_per_m"),
        alpha=_require_non_negative_number(material_raw["alpha"], "material.alpha"),
        ku_j_per_m3=_require_finite(material_raw["ku_j_per_m3"], "material.ku_j_per_m3"),
        anisotropy_axis=_require_unit_vector(
            material_raw["anisotropy_axis"], "material.anisotropy_axis"
        ),
    )

    geometry_raw = _require_mapping(root["geometry"], "geometry")
    _check_mapping_keys(geometry_raw, "geometry", _GEOMETRY_KEYS)
    geometry = GeometryConfig(
        size_m=_require_vector3(
            geometry_raw["size_m"], "geometry.size_m", component=_require_positive_number
        ),
        cells=_require_cells3(geometry_raw["cells"], "geometry.cells"),
    )

    initial_m = _require_unit_vector(root["initial_m"], "initial_m")

    recording_raw = _require_mapping(root["recording"], "recording")
    _check_mapping_keys(recording_raw, "recording", _RECORDING_KEYS)
    recording = RecordingConfig(
        sample_interval_s=_require_positive_number(
            recording_raw["sample_interval_s"], "recording.sample_interval_s"
        ),
        sample_count=_require_positive_int(recording_raw["sample_count"], "recording.sample_count"),
    )

    numerics_raw = _require_mapping(root["numerics"], "numerics")
    _check_mapping_keys(numerics_raw, "numerics", _NUMERICS_KEYS)
    numerics = NumericsConfig(
        edge_smooth=_require_non_negative_int(numerics_raw["edge_smooth"], "numerics.edge_smooth"),
        solver=_require_positive_int(numerics_raw["solver"], "numerics.solver"),
        max_err=_require_positive_number(numerics_raw["max_err"], "numerics.max_err"),
        max_dt_s=_require_positive_number(numerics_raw["max_dt_s"], "numerics.max_dt_s"),
        gamma_ll_rad_per_t_s=_require_positive_number(
            numerics_raw["gamma_ll_rad_per_t_s"], "numerics.gamma_ll_rad_per_t_s"
        ),
        relax_torque_threshold_t=_require_finite(
            numerics_raw["relax_torque_threshold_t"], "numerics.relax_torque_threshold_t"
        ),
    )

    pulses_raw = root["pulses"]
    if not isinstance(pulses_raw, list) or not pulses_raw:
        _fail("pulses", "must be a non-empty pulse list")

    pulses: list[PulseConfig] = []
    seen_pulse_ids: set[str] = set()
    for index, item in enumerate(pulses_raw):
        field = f"pulses[{index}]"
        pulse_raw = _require_mapping(item, field)
        _check_mapping_keys(pulse_raw, field, _PULSE_KEYS)
        pulse_id = _require_safe_path_segment(pulse_raw["pulse_id"], f"{field}.pulse_id")
        if pulse_id in seen_pulse_ids:
            _fail(f"{field}.pulse_id", f"duplicate pulse_id {pulse_id!r}")
        seen_pulse_ids.add(pulse_id)
        amplitude_millitesla = _require_non_negative_number(
            pulse_raw["b_ext_amplitude_mT"], f"{field}.b_ext_amplitude_mT"
        )
        direction = _require_unit_vector(pulse_raw["direction"], f"{field}.direction")
        duration_s = _require_positive_number(pulse_raw["duration_s"], f"{field}.duration_s")
        pulses.append(
            PulseConfig(
                pulse_id=pulse_id,
                # Unit boundary: mT exists only in YAML; multiply by 1e-3 and store as runtime T.
                b_ext_amplitude_t=amplitude_millitesla * 1e-3,
                direction=direction,
                duration_s=duration_s,
            )
        )

    return SimulationConfig(
        dataset_name=dataset_name,
        material=material,
        geometry=geometry,
        initial_m=initial_m,
        recording=recording,
        numerics=numerics,
        pulses=tuple(pulses),
    )


def derive_cell_size_m(geometry: GeometryConfig) -> Vector3:
    """Derive the grid cell size: cell_size[i] = size_m[i] / cells[i] (single source of truth)."""
    return (
        geometry.size_m[0] / geometry.cells[0],
        geometry.size_m[1] / geometry.cells[1],
        geometry.size_m[2] / geometry.cells[2],
    )


def parameter_set_id(alpha: float, ku_j_per_m3: float) -> str:
    """Generate the parameter-set identity from (alpha, Ku); the unique split grouping key.

    SHA-256 of the canonical ``.17g`` text, first 16 lowercase hex characters; depends
    only on alpha and Ku.
    """
    _require_finite(alpha, "alpha")
    _require_finite(ku_j_per_m3, "ku_j_per_m3")
    canonical = f"alpha={float(alpha):.17g}\nku_j_per_m3={float(ku_j_per_m3):.17g}\n".encode()
    return hashlib.sha256(canonical).hexdigest()[:16]
