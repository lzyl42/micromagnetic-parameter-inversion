# MuMax3 simulations

MuMax3 is an external per-machine dependency: it is **not** installed by
this project and its binaries are **not** stored in Git.

- Install MuMax3 yourself on each machine (Linux/Windows) and either add it
  to `PATH` or set `MUMAX3_BIN` to the executable.
- Scripts (`.mx3`) and simulation manifests can be tracked here; raw output
  goes to `data/raw/` (ignored by Git).
- Invocation is handled by `src/micromagnetic_parameter_inversion/external.py`
  (argument lists only, never `shell=True`).