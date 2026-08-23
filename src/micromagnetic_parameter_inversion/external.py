"""MuMax3 discovery and invocation.

MuMax3 is installed per-machine and is never part of this repository.
Locate it via the ``MUMAX3_BIN`` environment variable (default ``mumax3``
on Linux, ``mumax3.exe`` on Windows) using ``shutil.which``. All
invocations use argument lists via ``subprocess`` — never ``shell=True``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def default_mumax3_name() -> str:
    """Return the platform-appropriate default MuMax3 executable name."""
    return "mumax3.exe" if sys.platform == "win32" else "mumax3"


def find_mumax3() -> str | None:
    """Return the path to the MuMax3 executable, or ``None`` if not found.

    ``MUMAX3_BIN`` wins when set; otherwise the platform default name is
    searched on ``PATH``. ``MUMAX3_BIN`` may also point directly at a file
    that is not on ``PATH``.
    """
    configured = os.environ.get("MUMAX3_BIN")
    candidates: list[str] = []
    if configured:
        candidates.append(configured)
    candidates.append(default_mumax3_name())
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if configured and candidate == configured:
            path = Path(candidate)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    return None


@dataclass(frozen=True)
class Mumax3Status:
    """Availability report for MuMax3."""

    available: bool
    executable: str | None
    message: str


def mumax3_status() -> Mumax3Status:
    """Report whether MuMax3 is available, with a diagnostic message."""
    exe = find_mumax3()
    if exe is None:
        return Mumax3Status(
            available=False,
            executable=None,
            message=(
                "MuMax3 not found. Install it on this machine and either add it to "
                "PATH or set MUMAX3_BIN to the executable path (e.g. "
                "MUMAX3_BIN=/opt/mumax3/mumax3 on Linux, "
                "MUMAX3_BIN=C:\\mumax3\\mumax3.exe on Windows)."
            ),
        )
    return Mumax3Status(available=True, executable=exe, message=f"MuMax3 found at {exe}")


def run_mumax3(
    script: str | Path,
    *,
    workdir: str | Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a MuMax3 script with an argument list (never a shell).

    Raises ``FileNotFoundError`` with a diagnostic message when MuMax3 is
    not installed. Output is captured; callers decide how to handle a
    non-zero return code.
    """
    exe = find_mumax3()
    if exe is None:
        raise FileNotFoundError(mumax3_status().message)
    return subprocess.run(
        [exe, str(script)],
        cwd=str(workdir) if workdir is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
