# micromagnetic-parameter-inversion

基于 MuMax3 多激励磁化动力学，用 MLP、1D CNN、Temporal Transformer 反演
Gilbert 阻尼系数 `alpha` 与单轴磁各向异性常数 `Ku` 的研究项目。

项目目标、科研约束与当前实现状态以本仓库 `README.md`、`AGENTS.md`、配置和代码为准。

## 环境要求

- Python 3.13（`>=3.13,<3.14`），由 uv 管理（`.python-version`、`uv.lock`）。
- 支持 x86_64 Linux 与 Windows；PyTorch 固定使用官方 CUDA 12.8 wheel
  （`https://download.pytorch.org/whl/cu128`，经 `[tool.uv.sources]` 的
  platform marker 配置，同一 `uv.lock` 覆盖两个平台）。
- **GPU 驱动是外部前提**：本项目不安装/配置 NVIDIA 驱动或 CUDA Toolkit。
  当前 Ubuntu agent 已直通 Tesla T10 16GB，并以 NVIDIA 580.173.02 驱动和
  PyTorch cu128 完成 CUDA 运算验证；其他机器仍按运行时状态自动选择设备
  （`device: auto`）。
- **MuMax3 按机器单独安装，不入库**：见下方「MuMax3」。

## 快速开始

```bash
# Linux / macOS 风格
uv sync

# Windows (PowerShell)
uv sync
```

`uv sync` 会创建 `.venv`、安装依赖（含 cu128 PyTorch）并生成 `uv.lock`。
`.venv` 不入库。

## VS Code

- 推荐扩展见 `.vscode/extensions.json`（ruff、python、pylance、jupyter）。
- 解释器：依赖 Python 扩展自动发现工作区 `.venv`（跨平台，不硬编码
  `bin/python` 或 `Scripts/python.exe`）；若未自动选中，用
  "Python: Select Interpreter" 手动选择 `.venv`。
- pytest 与 ruff 已配置（`.vscode/settings.json`）。

## 运行检查

```bash
uv run python scripts/check_environment.py
```

输出 Python/平台、关键包版本、torch build CUDA、CUDA 可用性/设备、选择的
计算设备、MuMax3 可用性、数据/输出路径。无 GPU 或无 MuMax3 时仍以 0 退出，
但会明确报告状态。

## 测试与格式

```bash
uv run pytest          # 测试不要求 GPU / MuMax3
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pre-commit run --all-files   # 可选，repo-local hooks
```

## 环境变量

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `MUMAX3_BIN` | MuMax3 可执行文件路径 | Linux `mumax3` / Windows `mumax3.exe`（PATH 搜索） |
| `MICROMAG_DATA_ROOT` | 数据根目录 | `<project>/data` |
| `MICROMAG_OUTPUT_ROOT` | 输出/产物根目录 | `<project>/artifacts` |

示例见 `.env.example`（复制为 `.env`，`.env` 不入库）。运行代码不硬编码
任何机器绝对路径（`src/micromagnetic_parameter_inversion/paths.py`）。

## 数据政策

完整数据（`data/raw/`、`data/processed/`）、checkpoints、TensorBoard
runs/cache 不入库；`data/README.md` 与 `data/samples/` 可跟踪；最终
`results/figures`、`results/tables` 可跟踪。大数据与最佳模型未来走
Git LFS 或独立发布。详见 `data/README.md`。

## MuMax3

- 本项目不打包 MuMax3 二进制；每台机器单独安装后加入 PATH 或设置
  `MUMAX3_BIN`。当前 Ubuntu agent 已用户级安装 MuMax3 3.12（CUDA 12.9
  build），并通过官方 `mumax3 -test`；其他机器仍须独立安装匹配版本。
- 调用统一走 `src/micromagnetic_parameter_inversion/external.py`
  （`shutil.which` + `subprocess` 参数列表，禁止 `shell=True`）。
- `scripts/check_environment.py` 会报告当前机器的 MuMax3、GPU 与 CUDA 状态；
  不得仅凭依赖声明假设外部二进制可用。

## 目录结构

```
src/micromagnetic_parameter_inversion/   # 包（runtime / external / paths）
scripts/check_environment.py             # 环境诊断
tests/                                   # pytest（无需 GPU/MuMax3）
configs/base.yaml                        # 通用设置（seed、device=auto）
data/                                    # 数据（完整数据不入库）
notebooks/                               # 探索性 notebook
simulations/mumax3/                      # MuMax3 脚本/manifest
results/figures, results/tables          # 最终结果（可跟踪）
```
