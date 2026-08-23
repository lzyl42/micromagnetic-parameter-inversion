"""Compute-device selection and runtime diagnostics.

Selects CUDA when available, otherwise CPU. Never requires a GPU to be
present: everything in this module must work on a CPU-only machine (the
CUDA 12.8 PyTorch build runs fine on CPU; CUDA activates once NVIDIA
drivers exist on a GPU machine).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class RuntimeInfo:
    """Diagnostic snapshot of the compute runtime."""

    device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    torch_version: str
    torch_cuda_version: str | None
    python_version: str
    platform: str
    extra: dict[str, Any] = field(default_factory=dict)


def select_device(prefer: str = "auto") -> str:
    """Return the compute device string to use.

    ``prefer`` is one of:

    - ``"auto"``: CUDA when available, otherwise CPU;
    - ``"cuda"``: raise ``RuntimeError`` if CUDA is unavailable;
    - ``"cpu"``: always CPU.
    """
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return "cuda"
    if prefer != "auto":
        raise ValueError(f"Unknown device preference: {prefer!r}")
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_runtime_info() -> RuntimeInfo:
    """Collect environment information without requiring a GPU."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = None
    if cuda_available and device_count > 0:
        device_name = torch.cuda.get_device_name(0)
    return RuntimeInfo(
        device=select_device("auto"),
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        cuda_device_name=device_name,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
    )


def smoke_test() -> str:
    """Run a tiny tensor op on the selected device; returns the device used.

    Raises if the op does not actually execute and produce the expected
    result, so a CPU-only machine still gets a real computation check.
    """
    device = select_device("auto")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = (x * 2.0 + 1.0).sum().item()
    if y != 15.0:
        raise AssertionError(f"Unexpected smoke-test result: {y}")
    return device
