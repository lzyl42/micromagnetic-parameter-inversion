"""MuMax3 .mx3 template rendering.

Template originals live in simulations/mumax3/ (equilibrium.mx3.in and
simulation.mx3.in) and are read-only; reading template files, writing rendered
output, and hashing are handled by mumax3_pipeline.
"""

from __future__ import annotations

import re

from micromagnetic_parameter_inversion.mumax3_config import (
    PulseConfig,
    SimulationConfig,
    Vector3,
    derive_cell_size_m,
)

# Established placeholder shape: an uppercase identifier wrapped in double braces
# (e.g. {{MODEL_SETUP}}).
_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z0-9_]+\}\}")


class TemplateRenderError(ValueError):
    """Template placeholder contract violated.

    Missing, duplicated, or residual placeholders after rendering.
    """


def _fmt_number(value: float) -> str:
    """Deterministic numeric text: ``.17g`` decimal/scientific notation.

    Directly parseable by MuMax3 (Go syntax). The same float value always yields the
    same text (byte-reproducible); -0 and 0 are both rendered as 0.
    """
    number = float(value)
    if number == 0.0:
        return "0"
    return format(number, ".17g")


def _fmt_vector3(components: Vector3) -> str:
    """Render 3 components as an ``x, y, z`` argument list for uniform/vector."""
    return ", ".join(_fmt_number(component) for component in components)


def _replace_once(template_text: str, placeholder: str, replacement: str) -> str:
    """The placeholder must appear exactly once and be replaced whole.

    Violation raises TemplateRenderError.
    """
    parts = template_text.split(placeholder)
    occurrences = len(parts) - 1
    if occurrences != 1:
        raise TemplateRenderError(
            f"placeholder {placeholder} appears {occurrences} times, must appear exactly once"
        )
    return replacement.join((parts[0], parts[1]))


def _reject_residual_placeholders(rendered: str) -> None:
    """Any residual {{...}} in the rendered result breaks the template contract."""
    residual = _PLACEHOLDER_PATTERN.findall(rendered)
    if residual:
        raise TemplateRenderError(f"residual placeholders after rendering: {residual!r}")


def _render_model_setup(config: SimulationConfig) -> str:
    """The only renderer for the shared model section {{MODEL_SETUP}}.

    Prevents drift between the two scripts.

    Fixed order: numerical protocol (EdgeSmooth must be set before SetGeom to affect
    geometry voxelization; SetSolver/MaxErr/MaxDt/GammaLL shared integration controls,
    explicitly identical for equilibrium/simulation), grid, cell size (derived from
    size_m/cells, single source of truth), PBC (open boundaries), ellipsoid geometry
    (SetGeom always takes the three full diameters from size_m, with no conditional
    branch on nz: with nz=1 the single-layer voxel discretization naturally behaves as
    a constant-thickness elliptical-section slab, and with nz>1 the ellipsoid z
    surface is resolved layer by layer), demag, material parameters
    (Msat/Aex/Ku1/easy axis). alpha and RelaxTorqueThreshold are per-run/per-script
    parameters and not part of the shared section; the output contains no paths or
    placeholders.
    """
    material = config.material
    geometry = config.geometry
    numerics = config.numerics
    size_x, size_y, size_z = geometry.size_m
    geom_line = f"SetGeom(Ellipsoid({_fmt_vector3((size_x, size_y, size_z))}))"
    return "\n".join(
        (
            "// Numerical protocol (YAML numerics block; identical for equilibrium/simulation)",
            "// EdgeSmooth affects geometry voxelization; set before SetGeom (0 = hard step)",
            f"EdgeSmooth = {numerics.edge_smooth}",
            f"SetSolver({numerics.solver})",
            f"MaxErr = {_fmt_number(numerics.max_err)}",
            f"MaxDt = {_fmt_number(numerics.max_dt_s)}",
            f"GammaLL = {_fmt_number(numerics.gamma_ll_rad_per_t_s)}",
            "// Grid and cell size: cell_size = size_m / cells (single source of truth), units m",
            f"SetGridSize({geometry.cells[0]}, {geometry.cells[1]}, {geometry.cells[2]})",
            f"SetCellSize({_fmt_vector3(derive_cell_size_m(geometry))})",
            "// Open boundaries: periodic images not enabled",
            "SetPBC(0, 0, 0)",
            "// Flat-ellipsoid thin nano-magnet: full diameters dx, dy, dz all come "
            "from size_m (= bounding-box size)",
            geom_line,
            "// Demagnetizing field enabled",
            "EnableDemag = true",
            "// Single uniform material (SI units: Msat A/m; Aex J/m; Ku1 J/m^3) and "
            "easy-axis unit vector",
            f"Msat = {_fmt_number(material.ms_a_per_m)}",
            f"Aex = {_fmt_number(material.aex_j_per_m)}",
            f"Ku1 = {_fmt_number(material.ku_j_per_m3)}",
            f"anisU = vector({_fmt_vector3(material.anisotropy_axis)})",
        )
    )


def render_equilibrium_script(config: SimulationConfig, template_text: str) -> str:
    """Render config into the equilibrium template text (executed once per parameter set)."""
    rendered = _replace_once(template_text, "{{MODEL_SETUP}}", _render_model_setup(config))
    rendered = _replace_once(rendered, "{{INIT_M}}", _fmt_vector3(config.initial_m))
    rendered = _replace_once(
        rendered,
        "{{RELAX_TORQUE_THRESHOLD_T}}",
        _fmt_number(config.numerics.relax_torque_threshold_t),
    )
    _reject_residual_placeholders(rendered)
    return rendered


def render_simulation_script(
    config: SimulationConfig, pulse: PulseConfig, template_text: str
) -> str:
    """Render config and a single pulse into the simulation template text (once per pulse).

    The three external-field components are computed inside the renderer
    (amplitude_t * direction[i], in T; the conversion already happened at the
    load_config boundary); the same config + same pulse render byte-identically.
    """
    amplitude_t = pulse.b_ext_amplitude_t
    direction_x, direction_y, direction_z = pulse.direction
    rendered = template_text
    for placeholder, replacement in (
        ("{{MODEL_SETUP}}", _render_model_setup(config)),
        ("{{ALPHA}}", _fmt_number(config.material.alpha)),
        ("{{B_EXT_X_T}}", _fmt_number(amplitude_t * direction_x)),
        ("{{B_EXT_Y_T}}", _fmt_number(amplitude_t * direction_y)),
        ("{{B_EXT_Z_T}}", _fmt_number(amplitude_t * direction_z)),
        ("{{PULSE_DURATION_S}}", _fmt_number(pulse.duration_s)),
        ("{{SAMPLE_COUNT}}", str(config.recording.sample_count)),
        ("{{SAMPLE_INTERVAL_S}}", _fmt_number(config.recording.sample_interval_s)),
    ):
        rendered = _replace_once(rendered, placeholder, replacement)
    _reject_residual_placeholders(rendered)
    return rendered
