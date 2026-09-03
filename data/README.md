# Data policy

Full datasets (raw MuMax3 outputs, processed tensors) are **not** stored in
this Git repository. The repository keeps only:

- `README.md` (this file) — data layout and provenance notes;
- `samples/` — small sample files for smoke tests and demos;
- schemas and train/val/test split definitions (when they exist).

Large data and best models are intended for Git LFS or a separate release
channel (e.g. a data server or archive), not for the normal Git history.

## Layout

```
data/
  raw/        # MuMax3 simulation outputs (ignored by Git)
  processed/  # cleaned/featurized tensors (ignored by Git)
  samples/    # small tracked samples
```

Actual roots can be redirected with `MICROMAG_DATA_ROOT` /
`MICROMAG_OUTPUT_ROOT` (see `src/micromagnetic_parameter_inversion/paths.py`).