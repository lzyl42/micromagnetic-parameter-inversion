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
  samples/    # prepared training samples; generated dataset dirs untracked
```

Actual roots can be redirected with `MICROMAG_DATA_ROOT` /
`MICROMAG_OUTPUT_ROOT` (see `src/micromagnetic_parameter_inversion/paths.py`).

## Training samples (train.md §2)

`scripts/prepare_training_samples.py` normalizes raw MuMax3 outputs into
`data/samples/<dataset_name>/`:

- `<psid>.npz` — one file per parameter set: `x float32 [P,T,3]` (frozen
  pulse order, channels mx,my,mz), `y float32 [2]` (alpha,
  ku_j_per_m3 in physical units), `t_s float64 [T]`,
  `parameter_set_id <U16` scalar, `pulse_ids <U… [P]`;
- `dataset_meta.yaml` — protocol summary, frozen pulse order,
  psid→(alpha, Ku) labels, member list, provenance (source index paths,
  per-group config sha256, generation time);
- `split.yaml` — group-level train/val/test member lists (largest-remainder
  allocation, ties fixed train→val→test, min-count check).

Generated dataset directories are **not tracked** (`.gitignore`:
`data/samples/*`). The whitelist permits curated small files under
`data/samples/`, but Git cannot track an empty directory and no sample
files are currently tracked; curated files, if ever needed, are added
explicitly with `git add -f` (no extra README or LFS is required by
policy). Writing is pre-checked and refuses to overwrite existing outputs;
a failed run may leave partial artifacts that must be cleaned manually.
Full-data distribution (LFS/release) remains a research decision
(train.md §10).

## Archiving a runnable run

A run does not embed a full copy of the sample metadata: the run keeps
only its own `split.yaml` copy, and checkpoints reference the external
`dataset_meta.yaml` by anchor-relative path + sha256 (plus the input
contract). Keeping a run evaluable therefore requires archiving both the
samples directory (`dataset_meta.yaml` + `<psid>.npz`) and the run
directory; both roots are configurable via `MICROMAG_DATA_ROOT` /
`MICROMAG_OUTPUT_ROOT` (see `paths.py`).