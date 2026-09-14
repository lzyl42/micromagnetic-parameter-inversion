# micromagnetic-parameter-inversion

Invert the Gilbert damping coefficient `alpha` and the uniaxial anisotropy constant
`Ku` from MuMax3 magnetization dynamics. The current implementation is a **single-
excitation baseline**: a 2 mT / 50 ps short pulse along y, recording the spatially
averaged magnetization trajectory `(m_x, m_y, m_z)` after the field is switched off,
and regressing `(alpha, Ku)` with an MLP; multiple excitations (pulses in different
directions to reduce parameter ambiguity) are a long-term project goal and are not
enabled yet.

**Project status**: all four pipeline stages -- MuMax3 simulation generation, sample
preparation, MLP training, and independent test evaluation -- are implemented (entry
points and usage below). Formal data generation is in progress; a single-excitation
MLP baseline has been trained on the synthetic benchmark, and its validation-set
results are reported in [results/mlp.md](results/mlp.md); those results make no claim
of independent-test or real-device inversion performance. Follow-up models such as
1D CNN / Transformer are outside the current implementation scope.

## Physical model and fixed protocol

MuMax3 solves the Landau-Lifshitz-Gilbert equation at 0 K (explicit Gilbert form,
consistent with the MuMax3 kernel convention):

```
dm/dt = -γ_LL/(1+α²) · [m × B_eff + α m × (m × B_eff)]
```

Here `α` is the `alpha` to be inverted, `γ_LL` (`GammaLL`) uses the MuMax3 default
positive-value convention `1.7595e11 rad/(T·s)`, and `B_eff` is the effective field
(in T) with four terms: exchange, uniaxial anisotropy, demagnetizing, and external
field; 0 K, with no thermal-noise term. A single uniform effective medium, synthetic
CoFeB-inspired baseline; no claim of reproducing any specific material stack; no
DMI / STT / static bias field, open boundaries (`SetPBC(0, 0, 0)`, `EnableDemag = true`).

- Geometry: flat triaxial ellipsoid (not a constant-thickness elliptical cylinder),
  with the three full diameters of `SetGeom(Ellipsoid(dx, dy, dz))` equal to
  `100 nm × 50 nm × 2 nm` (i.e. the `size_m` bounding-box size), and
  `cells = [40, 20, 4]`; the easy axis and the initial magnetization are both along `+x`.
- Fixed material/numerical parameters: `Ms = 1.25e6 A/m`, `Aex = 15e-12 J/m`,
  `EdgeSmooth = 12` (set before `SetGeom`), `solver = 5`, `MaxErr = 1e-5`,
  `MaxDt = 1e-11 s`, `RelaxTorqueThreshold = -1` (official default).
- Excitation and sampling: for each `(alpha, Ku)` parameter set, first run
  [equilibrium.mx3.in](simulations/mumax3/equilibrium.mx3.in) once (zero field,
  `Relax()`, producing the equilibrium state shared by all pulses); then
  [simulation.mx3.in](simulations/mumax3/simulation.mx3.in) loads that equilibrium
  state, sets the true `alpha`, applies a 2 mT / 50 ps rectangular pulse along y, and
  switches the field off exactly (`B_ext = 0`), recording the spatially averaged
  magnetization `(m_x, m_y, m_z)` every 10 ps from the switch-off instant, 401 points
  in total (0..4 ns after switch-off).
- Inversion targets: `alpha ∈ [0.004, 0.020]` (log-space sampling),
  `Ku ∈ [2000, 30000] J/m^3` (linear-space sampling), 1024 Sobol points
  (`scramble=True, rng=42`); `Ku = 0` serves only as a physical control and is
  excluded from the main-domain error.
- The above are **fixed discretization conventions**: neither mesh convergence nor
  real-device validity has been verified, and they do not represent convergence
  conclusions.

The single source of truth for protocol values is
[scripts/generate_dataset.py](scripts/generate_dataset.py) (`FIXED_CONFIGS` /
`PARAMETERS`); the YAML validation and mT→T unit-conversion boundary are in
[src/micromagnetic_parameter_inversion/mumax3_config.py](src/micromagnetic_parameter_inversion/mumax3_config.py);
template rendering (the only renderer for the shared model section and the ellipsoid
geometry/material parameters) is in
[src/micromagnetic_parameter_inversion/mumax3_script.py](src/micromagnetic_parameter_inversion/mumax3_script.py);
simulation orchestration and trajectory export are in
[src/micromagnetic_parameter_inversion/mumax3_pipeline.py](src/micromagnetic_parameter_inversion/mumax3_pipeline.py)
and
[src/micromagnetic_parameter_inversion/mumax3_results.py](src/micromagnetic_parameter_inversion/mumax3_results.py);
the protocol template and field descriptions are in
[configs/experiments/mumax3_simulation.yaml](configs/experiments/mumax3_simulation.yaml).

## MLP architecture and training (highlights)

The baseline model is a pure MLP regressor
([src/micromagnetic_parameter_inversion/models/mlp.py](src/micromagnetic_parameter_inversion/models/mlp.py)):

- Input contract: one sample is the raw time-domain trajectory `[P, T, 3]` of all
  pulses for a parameter set (batch `[N, P, T, 3]`, channels mx/my/mz); no per-pulse
  splitting, hand-crafted statistical features, or downsampling. The current
  protocol has `P = 1` (only `pulse_A2`) and `T = 401`, so the flattened dimension is
  `D = P·T·3 = 1203`.
- Network: `Flatten` followed by a hidden-layer sequence (default `64 → 32 → 32`,
  each layer `Linear + ReLU`), with a final `Linear` layer producing two outputs
  `(alpha, Ku)` in **standardized label space** (z-score space, not physical units).
  `hidden_dims` is configured in
  [configs/training/mlp.yaml](configs/training/mlp.yaml) as an engineering candidate,
  not a validated research parameter.
- Preprocessing
  ([src/micromagnetic_parameter_inversion/preprocessing.py](src/micromagnetic_parameter_inversion/preprocessing.py)):
  input statistics are fit on the train split only and aggregated over the sample and
  time axes into per-pulse-position, per-magnetization-component mean/std (shape
  `[P, 1, 3]`, broadcast over the full trajectory; positions with
  `std <= std_eps` (default `1e-8`) use a divisor of 1). The label transform defaults
  to `identity`; the optional `logalpha` takes **log10** of the alpha column only
  (computed in float64, requires alpha > 0; not the natural logarithm), followed by
  per-output z-scoring.
- Training
  ([src/micromagnetic_parameter_inversion/training.py](src/micromagnetic_parameter_inversion/training.py)):
  Adam (defaults `lr = 1e-3`, `weight_decay = 0`), MSE loss in standardized label
  space, `batch_size = 32`, `max_epochs = 500`, `seed = 42`; early stopping
  (`patience = 50`, `min_delta = 0`) uses an independent reference, separated from the
  absolute best; the absolute-best val-loss weights are written to `best.pt`, and the
  weights of the last completed epoch to `final.pt`; non-finite loss/grad/pred stops
  immediately and no bad weights are saved.
- Evaluation
  ([src/micromagnetic_parameter_inversion/evaluation.py](src/micromagnetic_parameter_inversion/evaluation.py)):
  network structure, preprocessing, and label transform are all restored from the
  checkpoint (not from the current YAML); predictions are inverse-transformed back to
  physical units before reporting alpha/Ku MAE/RMSE (Ku in J/m^3); the main domain
  excludes the `Ku = 0` control, which is reported separately.

## Installation and prerequisites

- Python 3.13 (`>=3.13,<3.14`), managed by [uv](https://docs.astral.sh/uv/); run
  `uv sync` in the repository root to create `.venv` and install dependencies.
  PyTorch comes from the official CUDA 12.8 wheel index (x86_64 Linux / Windows, see
  `pyproject.toml`).
- A GPU driver and [MuMax3](https://mumax.github.io/) are external prerequisites; this
  project does not install or bundle them: add MuMax3 to `PATH` or set `MUMAX3_BIN`.
- Environment diagnostics:

  ```bash
  uv run python scripts/check_environment.py
  ```

  Reports Python, package versions, CUDA availability and device, MuMax3 status, and
  data/output paths; it still exits 0 when CUDA or MuMax3 is missing, so read the
  report content rather than relying on the exit code.

## Workflow (four entry points)

All commands below are run from the repository root; arguments follow each script's
actual CLI.

### 1. Generate raw data

```bash
uv run python scripts/generate_dataset.py
```

- No CLI arguments: the fixed protocol fields and 1024 Sobol parameter points are
  written in the script (`FIXED_CONFIGS` / `PARAMETERS`), and **no external
  YAML/manifest is read**; `configs/experiments/mumax3_simulation.yaml` is only the
  protocol template and schema reference, not the generation entry point.
- Prerequisite: MuMax3 available; the target output directory
  `data/raw/<dataset_name>/<psid>/` does not exist (an existing one is refused, not
  overwritten or cleaned).
- Concurrency is controlled by the script constant `MAX_WORKERS` (`1` = serial); there
  is no resume/skip; for a single point or a gap-filling rerun, keep only the not-yet-
  run targets in `PARAMETERS` inside the script.
- Generated configs are written under `artifacts/generated_configs/<dataset_name>/`.
  Batch execution occupies the GPU for a long time; assess local GPU resources and
  expected runtime before running.

### 2. Prepare training samples

```bash
uv run python scripts/prepare_training_samples.py \
  --config configs/training/mlp.yaml --parameter-set-ids all
```

- Prerequisite: replace the `dataset_name` placeholder in
  `configs/training/mlp.yaml` with the actual dataset; `--config` and
  `--parameter-set-ids` are both required (the latter is an explicit psid list or
  `all`, and `all` expands only the selected dataset directory).
- Output `data/samples/<dataset_name>/`: one `<psid>.npz` per parameter set, plus
  `dataset_meta.yaml` and `split.yaml`; pre-checks run before writing and existing
  files are refused; a failure may leave partial products that must be cleaned up
  manually before rerunning.

### 3. Training (train/val only)

```bash
uv run python scripts/train_mlp.py --config configs/training/mlp.yaml
```

- Prerequisites: replace the `run_name` placeholder; the output directory must not
  exist.
- Standardization/label statistics are fit on the train split only; `test` takes no
  part in tuning or model selection.
- The default output directory is `artifacts/training/mlp/<dataset>/<run_name>/` and
  contains `best.pt` / `final.pt` / `metrics.json` / `split.yaml`, etc.

### 4. Independent test evaluation

```bash
uv run python scripts/evaluate_model.py --run <RUN_DIR>
```

- `--checkpoint` is optional (default `<RUN_DIR>/best.pt`); the checkpoint is bound by
  SHA to the split copy inside the run.
- Writes `test_metrics.json` (main/control MAE/RMSE in physical units) and
  `test_predictions.csv` into the run directory; existing products are refused.

## Data and outputs

- `data/raw/`: raw MuMax3 output; `data/samples/`: preprocessed samples; `artifacts/`:
  run products and checkpoints. Raw data, samples, and artifacts are not committed to
  Git; the public repository contains only code and documentation.
- Data and output paths can be overridden with environment variables:
  `MICROMAG_DATA_ROOT`, `MICROMAG_OUTPUT_ROOT` (see
  [src/micromagnetic_parameter_inversion/paths.py](src/micromagnetic_parameter_inversion/paths.py));
  the MuMax3 executable uses `MUMAX3_BIN` (see
  [src/micromagnetic_parameter_inversion/external.py](src/micromagnetic_parameter_inversion/external.py)),
  and when unset it is looked up in `PATH` under the platform default name.
- **`.env` is not loaded automatically** (the code does not call `load_dotenv`):
  inject environment variables explicitly from the shell or runtime environment; do
  not assume that placing a `.env` file is sufficient. `.env` is excluded by
  `.gitignore` and never enters the repository.
- No open-source license has been chosen yet (no LICENSE added).

## Research-integrity conventions

- Data splits are grouped by parameter combination `(alpha, Ku)`: all excitation
  trajectories of the same combination must stay in the same split (train/val/test)
  to prevent cross-group leakage.
- Standardization and label statistics are fit on the training split only; test is
  evaluated independently; the training seed comes from `training.seed` in
  `configs/training/mlp.yaml`, the data-generation Sobol seed is fixed in
  `scripts/generate_dataset.py`; dependencies are pinned in `uv.lock`, and parameter
  ranges are not invented.
- Final conclusions require MuMax3 forward re-validation (inverted parameters →
  forward simulation → comparison with observations), which has not been performed
  yet.
- **Not yet verified**: mesh convergence, the Relax convergence threshold and its
  robustness, systematic justification of the EdgeSmooth choice, batch
  reproducibility, physical-level OVF QC, real-device validity, forward
  re-validation, and training effectiveness on formal research data.
