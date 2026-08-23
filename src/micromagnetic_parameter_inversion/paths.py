"""Project path resolution.

Data and output roots come from environment variables when set, otherwise
fall back to project-local ``data/`` and ``artifacts/`` directories. No
absolute machine-specific paths are hard-coded here.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = "MICROMAG_DATA_ROOT"
OUTPUT_ROOT_ENV = "MICROMAG_OUTPUT_ROOT"


def data_root() -> Path:
    """Root for raw/processed data (env override or project-local ``data/``)."""
    return _resolve(DATA_ROOT_ENV, PROJECT_ROOT / "data")


def output_root() -> Path:
    """Root for artifacts/checkpoints (env override or project-local ``artifacts/``)."""
    return _resolve(OUTPUT_ROOT_ENV, PROJECT_ROOT / "artifacts")


def ensure_dirs() -> tuple[Path, Path]:
    """Create and return ``(data_root, output_root)``."""
    data = data_root()
    out = output_root()
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return data, out


def _resolve(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()
