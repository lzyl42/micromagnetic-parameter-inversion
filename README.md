# micromagnetic-parameter-inversion

Invert the Gilbert damping coefficient $\alpha$ and the uniaxial anisotropy constant
$K_u$ from single-pulse magnetization dynamics with CoFeB parameter settings. This
stage compares three regression models — MLP, CNN1D, and Temporal Transformer — on
synthetic single-pulse data, and completes a same-protocol forward re-simulation of
the inverted parameters.

## Physical model and protocol

MuMax3 integrates the Landau–Lifshitz–Gilbert equation at 0 K (explicit Gilbert form):

$$
\frac{\mathrm{d}\mathbf{m}}{\mathrm{d}t}
  = -\frac{\gamma_\mathrm{LL}}{1+\alpha^2}
    \left[\mathbf{m}\times\mathbf{B}_\mathrm{eff}
    + \alpha\,\mathbf{m}\times(\mathbf{m}\times\mathbf{B}_\mathrm{eff})\right]
$$

where $\mathbf{B}\_\mathrm{eff}$ collects exchange, uniaxial anisotropy, demagnetizing,
and external-field terms (no thermal-noise term), and $\gamma\_\mathrm{LL}$ uses the
MuMax3 default value $1.7595 \times 10^{11}\,\mathrm{rad/(T \cdot s)}$.

- Geometry: triaxial ellipsoid, full diameters $100 \times 50 \times 2\,\mathrm{nm}$,
  cells `[40, 20, 4]`, easy axis and initial magnetization along $+x$; open boundaries
  with demagnetization enabled.
- Material: $M_s = 1.25 \times 10^6\,\mathrm{A/m}$,
  $A_\mathrm{ex} = 15 \times 10^{-12}\,\mathrm{J/m}$; `EdgeSmooth = 12`, `solver = 5`,
  `MaxErr = 1e-5`, `MaxDt = 1e-11 s`.
- Excitation: one rectangular $2\,\mathrm{mT}$ / $50\,\mathrm{ps}$ pulse along $+y$,
  after which the field is switched off exactly; the spatially averaged magnetization
  $(m_x, m_y, m_z)$ is sampled every $10\,\mathrm{ps}$ over $0$ – $4\,\mathrm{ns}$
  (401 points).
- Targets: $\alpha \in [0.004, 0.020]$ (logarithmic sampling),
  $K_u \in [2000, 30000]\,\mathrm{J/m^3}$.

## Data

1024 parameter combinations: $\alpha \in [0.004, 0.020]$ on a logarithmic scale and
$K_u \in [2000, 30000]\,\mathrm{J/m^3}$, drawn by fixed-seed Sobol sampling. Each
sample is the magnetization-response trajectory of the single `pulse_A2` excitation
above, covering $0$ – $4\,\mathrm{ns}$ with 401 samples at $10\,\mathrm{ps}$ spacing,
shape `[1, 401, 3]` (mx/my/mz). The parameter combinations are split
train/val/test = 717/154/153, with no combination crossing groups. The data are a
synthetic MuMax3 benchmark, not real experimental data.

Run records are kept in `results/public_release/mlp.yaml`, `cnn1d.yaml`, and
`transformer.yaml`; dataset metadata and the split are in
`results/public_release/data/dataset_meta.yaml` and `split.yaml`. The 1024 prepared
sample files and the model checkpoints are kept outside Git (local artifacts /
release attachments).

## Models

### Architectures

- Common input: the raw time-domain trajectory `[P, T, 3]` (pulses × time ×
  magnetization components); for this protocol `P = 1` and `T = 401`, so the
  flattened dimension is 1203.
- MLP: flatten, hidden layers `64 → 32 → 32` with ReLU, output layer `2`.
- CNN1D: pulses and components folded into channels (`[N, P, 3, T]`), `Conv1d`
  layers `8 → 16` (kernel 5) with ReLU, `AdaptiveAvgPool1d(16)`, head
  `256 → 16 → 2`.
- Transformer: time tokens `[N, T, 3P]`, linear projection to `d_model = 64`, fixed
  sinusoidal positional encoding, 2 pre-LN encoder layers (4 heads, FFN 128,
  dropout 0), final LayerNorm, mean pooling, head `64 → 32 → 2`.

### Test metrics

| Model | Architecture | Parameters | test MAPE $\alpha$ | test MAPE $K_u$ |
|---|---|---:|---:|---:|
| MLP | 1203→64→32→32→2 (ReLU) | 80258 | 0.46% | 0.92% |
| CNN1D | 3→8→16 (k5, pool16), head 256→16→2 | 4930 | 0.19% | 0.41% |
| Transformer | d64/h4/L2/FF128, mean pool, head 64→32→2, dropout 0 | 69474 | 0.20% | 0.40% |

All three models apply log10 to $\alpha$, then standardize both labels by
training-set statistics; the network output is inverse-transformed back to physical
units of $\alpha$ and $K_u$. The input is the single-pulse magnetization trajectory
(401×3), and the preprocessing statistics are fit on the training set only and stored
with the checkpoints. The included models were selected by the validation-set rule;
the table gives each representative checkpoint's metrics on the test set (n=153), not
a 5-seed group mean, and this selection did not use test results. All three test
MAPEs are below 1%; parameter count does not represent inference speed or efficiency. Complete per-model and cross-model val/test results are in [results/README.md](results/README.md).

## Forward validation

The inverted parameters are re-relaxed and re-simulated under the original protocol
and compared with the observed trajectory at every time step. The table lists the
magnetization vector RMSE (dimensionless, lower is better) at four ground-truth
points. The in-domain point is the first parameter combination of the frozen test
list; A varies only $\alpha$, B only $K_u$, and C takes both at 1.1× the respective
training upper limits. Apart from the inverted parameters, all physical and numerical
settings follow the original protocol.

| Point | Ground truth (α, Ku) | MLP | CNN1D | Transformer |
|---|---|---:|---:|---:|
| In-domain | 0.013578, 23618.065 | 3.40e-05 | 1.47e-04 | 1.61e-05 |
| A α upper extrapolation | 0.02200, 16000 | 5.32e-04 | 9.53e-04 | 1.90e-04 |
| B Ku upper extrapolation | 0.01000, 33000 | 1.18e-02 | 6.07e-04 | 5.80e-03 |
| C both upper extrapolation | 0.02200, 33000 | 6.58e-03 | 3.34e-03 | 3.69e-03 |

All three models have a lower in-domain vector RMSE than at their extrapolation
points. The forward runs use simulated synthetic data and the same pulse protocol;
the three extrapolation points A/B/C are not sufficient to represent the models'
general extrapolation capability.

Each section below compares the observed and forward-simulated magnetization
trajectories at the corresponding test point; columns are MLP, CNN1D, and
Transformer.

### In-domain (α=0.013578, Ku=23618.065)

| MLP | CNN1D | Transformer |
|---|---|---|
| ![MLP](results/public_release/figures/in_domain_mlp.png) | ![CNN1D](results/public_release/figures/in_domain_cnn1d.png) | ![Transformer](results/public_release/figures/in_domain_transformer.png) |

### A: α upper extrapolation (α=0.02200, Ku=16000)

| MLP | CNN1D | Transformer |
|---|---|---|
| ![MLP](results/public_release/figures/alpha_upper_mlp.png) | ![CNN1D](results/public_release/figures/alpha_upper_cnn1d.png) | ![Transformer](results/public_release/figures/alpha_upper_transformer.png) |

### B: Ku upper extrapolation (α=0.01000, Ku=33000)

| MLP | CNN1D | Transformer |
|---|---|---|
| ![MLP](results/public_release/figures/ku_upper_mlp.png) | ![CNN1D](results/public_release/figures/ku_upper_cnn1d.png) | ![Transformer](results/public_release/figures/ku_upper_transformer.png) |

### C: both upper extrapolation (α=0.02200, Ku=33000)

| MLP | CNN1D | Transformer |
|---|---|---|
| ![MLP](results/public_release/figures/both_upper_mlp.png) | ![CNN1D](results/public_release/figures/both_upper_cnn1d.png) | ![Transformer](results/public_release/figures/both_upper_transformer.png) |

## How to run

```bash
uv sync
uv run python scripts/check_environment.py
uv run python scripts/generate_dataset.py
uv run python scripts/prepare_training_samples.py --config configs/training/mlp.yaml --parameter-set-ids all
uv run python scripts/train_mlp.py --config configs/training/mlp.yaml
uv run python scripts/train_cnn1d.py --config configs/training/cnn1d.yaml
uv run python scripts/train_transformer.py --config configs/training/transformer.yaml
uv run python scripts/evaluate_model.py --run <RUN_DIR> --split test
uv run python scripts/validate_forward.py --checkpoint <CKPT> --sample-dir data/raw/<DATASET>/<PSID> --output <NEW_DIR>
```

- Python 3.13 with `uv`; MuMax3 and a GPU driver are external prerequisites
  (`MUMAX3_BIN`, or the executable on `PATH`). `check_environment.py` reports
  CUDA/MuMax3 status and still exits 0 when they are missing.
- `generate_dataset.py` takes no arguments (the protocol and parameter points are
  embedded in the script) and is a long GPU batch. `prepare_training_samples.py`
  writes `data/samples/<dataset_name>/`; replace `PLACEHOLDER_DATASET_NAME` and
  `PLACEHOLDER_RUN_NAME` in `configs/training/*.yaml` before use.
- `evaluate_model.py` requires the run directory (default checkpoint
  `<RUN_DIR>/best.pt`) and uses the split copy bound inside the run; the default is
  `--split test`, and `--split val` selects the validation members.
- `validate_forward.py` needs one checkpoint and one raw parameter-group directory
  (`config.yaml`, `index.csv`, trajectories); `--dry-run` writes only
  `prediction.json` and `forward_config.yaml`, without running MuMax3. `<NEW_DIR>`
  must not already exist. Data/output roots can be overridden with
  `MICROMAG_DATA_ROOT` / `MICROMAG_OUTPUT_ROOT`.
