#!/usr/bin/env python3
"""Print a diagnostic summary of the runtime environment.

Exits 0 even when there is no GPU and/or no MuMax3; missing pieces are
reported explicitly, so the script is safe to run on a fresh VM or in CI.
"""

from __future__ import annotations

import platform
import sys

import numpy
import pandas
import sklearn

from micromagnetic_parameter_inversion import external, paths, runtime


def _fmt_version(module: object) -> str:
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    info = runtime.get_runtime_info()
    mumax = external.mumax3_status()
    data = paths.data_root()
    out = paths.output_root()

    print("=" * 62)
    print("micromagnetic-parameter-inversion environment check")
    print("=" * 62)
    print(f"Python             : {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform           : {platform.system()} {platform.machine()}")
    print(f"PyTorch            : {info.torch_version}")
    print(f"torch.version.cuda : {info.torch_cuda_version or 'None (CPU-only build)'}")
    print(f"CUDA available     : {info.cuda_available}")
    print(f"CUDA devices       : {info.cuda_device_count}")
    if info.cuda_device_name:
        print(f"CUDA device        : {info.cuda_device_name}")
    print(f"Selected device    : {info.device}")
    print(f"numpy              : {_fmt_version(numpy)}")
    print(f"pandas             : {_fmt_version(pandas)}")
    print(f"scikit-learn       : {_fmt_version(sklearn)}")
    mumax_line = (
        f"MuMax3             : available at {mumax.executable}"
        if mumax.available
        else "MuMax3             : NOT INSTALLED"
    )
    print(mumax_line)
    print(f"Data root          : {data}")
    print(f"Output root        : {out}")

    # CPU fallback smoke test: a real tensor op must execute successfully.
    device = runtime.smoke_test()
    print(f"Smoke test         : OK on device '{device}' (tensor op executed)")

    if not info.cuda_available:
        print()
        print("NOTE: CUDA unavailable - expected on this VM. The cu128 build is installed;")
        print("      CUDA becomes active once NVIDIA drivers are present on a GPU machine.")
    if not mumax.available:
        print()
        print("NOTE: MuMax3 not installed - expected on this VM. Install per-machine and")
        print("      set MUMAX3_BIN or add it to PATH when needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
