# micromagnetic-parameter-inversion

基于 MuMax3 多激励磁化动力学，用 MLP、1D CNN、Temporal Transformer 反演
Gilbert 阻尼系数 `alpha` 与单轴磁各向异性常数 `Ku` 的研究项目。

项目目标、科研约束与当前实现状态以本仓库 `README.md`、`AGENTS.md`、配置和代码为准。

## 物理模型概述

本项目计划模拟的物理系统与变化过程如下（详细推导见 `plan.md` 第 5–8 节；几何尺寸等为初始方案，最终以试运行和网格收敛测试为准）。

### 模拟对象

一个扁平的三轴椭球薄纳米磁体（初定三轴全直径 `100 nm × 50 nm × 2 nm`，长轴沿 x 方向），初始磁化近似沿 `+x`：

- 长轴提供明确的形状易轴，磁体接近单畴、又保留少量空间非均匀性；
- 计算网格小，适合 GPU 批量模拟；
- 固定 `Ms`（饱和磁化强度）、`A`（交换常数）、几何与网格，待反演参数只有
  Gilbert 阻尼系数 `alpha` 与单轴磁各向异性常数 `Ku`。

几何语义：`size_m=[dx,dy,dz]` 是椭球三轴全直径（= 包围盒尺寸），公共模型
段恒渲染完整三轴 `SetGeom(Ellipsoid(dx, dy, dz))`；`cells=[nx,ny,nz]` 各分量
为任意正整数——`nz=1` 时单层体素离散自然表现为恒厚椭圆截面薄片，`nz>1`
时才逐层解析 z 方向椭球表面，正式研究须通过网格收敛测试确定 cells。

### 物理模型

微磁学连续介质模型：以归一化磁化场 `m(r, t)`（`|m| = 1`）为变量，系统总能量为

`E = E_ex + E_ani + E_demag + E_Z`

即交换能（使相邻磁化趋于平行）、单轴各向异性能（`Ku` 决定磁化偏离易轴的能量代价，起"回复作用"）、退磁能（源于磁体自身形状与磁化分布，倾向于把磁化保持在薄膜面内）与 Zeeman 能（外场使磁化沿外场方向排列）之和。

磁化随时间的变化由 Landau–Lifshitz–Gilbert（LLG）方程决定，其右侧包含两个作用：

- **进动项**：磁化像陀螺一样绕有效场旋转；
- **Gilbert 阻尼项**：磁化逐渐转向有效场方向，使振荡衰减（`alpha` 越大衰减越快）。

### 模拟的变化过程

每次模拟先用短磁场脉冲把磁化推离平衡位置，脉冲关闭后记录空间平均磁化
`(m_x, m_y, m_z)` 随时间的自由衰减轨迹，典型表现为阻尼振荡：

- `alpha` 主要控制振荡的衰减快慢；
- `Ku` 主要影响回复力矩，从而改变振荡频率与幅度。

### 多激励设计

同一组 (`alpha`, `Ku`) 分别施加三种方向的脉冲，利用不同方向磁场对参数敏感性的差异降低两参数间的混淆：

| 激励 | 脉冲方向 | 目的 |
| --- | --- | --- |
| A | 面内横向（沿 y） | 最简单的单激励基准 |
| B | 面内倾斜（x–y 面内） | 补充不同面内角度的响应 |
| C | 带面外分量（y–z 面内） | 激出面外进动信息 |

研究的核心逆问题：从一条或多条这样的平均磁化轨迹反演 (`alpha`, `Ku`)，再用 MuMax3 正向回代检验预测参数能否重建原始轨迹。

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

## MuMax3 模拟 CLI

```bash
uv run python scripts/run_mumax3_simulation.py --config <validated-experiment.yaml>
```

`--config` 指向一份实验 YAML（模板见
`configs/experiments/mumax3_simulation.yaml`）。该示例配置当前**所有研究值
均为 null 占位**：运行前必须先填写并审查正式研究值（`load_config` 会拒绝
任何仍为 null 的必填研究值）。训练/推理 pipeline 尚未实现。

YAML 中的指数数值请使用带指数符号的形式（如 `8.0e+5`）或直接写十进制
（如 `800000.0`）：`8.0e5` 这类不带符号的指数会被 PyYAML 解析为字符串，
随后被配置校验拒绝。

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
- 有限验证记录（2026-09-02，test-only pilot）：在当前机器上以临时短 pilot
  （8×4×1 网格、1 pulse、3 samples，数值不代表研究参数）验证了整条执行链：
  模板可解析执行、Relax 产生 equilibrium.ovf、simulation 经固定相对路径
  LoadFile 共享 OVF、原生表头 `# t (s) mx () my () mz ()` 与恰 3 行、parser
  时间重锚定、config.yaml 快照与 index.csv 原子写出；`mumax3 -test` 同轮
  通过。正式研究参数、
  最终几何/网格收敛、Relax 收敛阈值/鲁棒性、EdgeSmooth 选择、OVF 物理 QC
  与批量重复性、正向回代验证仍未进行。

## 目录结构

```
src/micromagnetic_parameter_inversion/   # 包（runtime / external / paths）
                                         # + mumax3 vertical slice（config / script / results / pipeline）
scripts/check_environment.py             # 环境诊断
scripts/run_mumax3_simulation.py         # MuMax3 模拟 CLI（薄封装）
configs/base.yaml                        # 通用设置（seed、device=auto）
configs/experiments/mumax3_simulation.yaml
                                         # 实验配置模板（研究值当前全 null）
simulations/mumax3/                      # MuMax3 脚本模板（.mx3.in）与说明
tests/                                   # pytest（无需 GPU/MuMax3）
data/                                    # 数据（完整数据不入库）
notebooks/                               # 探索性 notebook
results/figures, results/tables          # 最终结果（可跟踪）
```
