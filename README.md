# micromagnetic-parameter-inversion

基于 MuMax3 多激励磁化动力学，用 MLP、1D CNN、Temporal Transformer 反演
Gilbert 阻尼系数 `alpha` 与单轴磁各向异性常数 `Ku` 的研究项目。

项目目标、科研约束与当前实现状态以本仓库 `README.md`、`AGENTS.md`、配置和代码为准。

## 当前协议：Protocol B（固定离散 benchmark，训练优先）

**用户决策（2026-09）**：连续模型的网格收敛与真实器件有效性均未证明，
网格扫描停止；先行推进「固定离散 benchmark → 数据 → 训练」闭环。当前
一切模拟与数据生成以本协议为准；历史 ES0/其他网格/其他 pulse 协议数据
一律不得与 Protocol B 数据混用。

基准性质：0 K、无热噪声、单一均匀有效介质的 **synthetic CoFeB-inspired**
benchmark（仅借鉴 CoFeB 典型量级，不声称复现任何具体 stack）。**固定
离散 benchmark，不是网格收敛协议**：cells 与 EdgeSmooth 为固定约定，
不作为网格收敛结论。

### 固定字段全表（`configs/experiments/mumax3_simulation.yaml`）

| 字段 | 固定值 |
| --- | --- |
| `material.ms_a_per_m` | `1.25e+6` A/m |
| `material.aex_j_per_m` | `15.0e-12` J/m |
| `material.anisotropy_axis` | `+x`（`[1.0, 0.0, 0.0]`） |
| `geometry.size_m` | `100.0e-9 × 50.0e-9 × 2.0e-9` m（椭球三轴全直径） |
| `geometry.cells` | `[40, 20, 4]` |
| `initial_m` | `+x`（`[1.0, 0.0, 0.0]`） |
| `recording.sample_interval_s` | `10.0e-12` s（10 ps） |
| `recording.sample_count` | `401`（关场后 0..4 ns） |
| `numerics.edge_smooth` | `12` |
| `numerics.solver` | `5` |
| `numerics.max_err` | `1.0e-5` |
| `numerics.max_dt_s` | `1.0e-11` s |
| `numerics.gamma_ll_rad_per_t_s` | `1.7595e+11` rad/(T·s) |
| `numerics.relax_torque_threshold_t` | `-1.0`（保留官方默认收敛判据） |
| `pulses`（首版仅 1 个） | `pulse_A2`：方向 y、幅值 `2.0` mT、时长 `50.0e-12` s |

待填字段（仅三处，运行前必填，模板中保持 null）：
`dataset_name`、`material.alpha`、`material.ku_j_per_m3`。
`dataset_name` 须按协议命名——如 `cofeb_discrete_a2_v1` **仅为格式示例**，
未填入模板；每套协议须改名后填写。

### 参数主域（采样设计）

- `alpha ∈ [0.004, 0.020]`：在 **log(alpha) 空间**采样；
- `Ku ∈ [2000, 30000] J/m^3`：在**线性空间**采样；
- `Ku = 0`：仅作独立物理 control，**不进入主域误差统计**。

### 激励策略：先单激励 baseline，多激励为后续对照

首版协议只含单一 `pulse_A2`（y 方向 2 mT、50 ps），先把「单激励 →
(`alpha`, `Ku`)」baseline 跑通。模型方法的长远目标仍是多激励降混淆，
但当前只有单激励 baseline；增加激励（如 z 向脉冲）留作后续对照实验，
尚未批准执行。

### 数据使用纪律（同协议闭环）

- 同一 Protocol B 数据同时用于 training/val/test 划分与最终 MuMax3 正向
  回代检验；不与历史 ES0/其他 cells/其他 pulse 协议数据混用。
- 数据划分按参数组合进行：同一 (`alpha`, `Ku`) 的所有轨迹必须同组，
  防止跨组泄漏。
- 标准化/特征统计量只在训练组拟合，验证/测试组不得参与拟合。
- 固定依赖（`uv.lock`）、seed（`configs/base.yaml`）与代码版本；
  不为此新增脚本或 schema 字段。

### 如何运行

模拟执行唯一入口是批量脚本 `scripts/generate_dataset.py`（旧的单份配置
CLI `scripts/run_mumax3_simulation.py` 已删除）。固定字段内置在
`FIXED_CONFIGS`（与上表一致），参数点内置在 `PARAMETERS`——1024 点
Sobol（`scipy.stats.qmc.Sobol(d=2, scramble=True, rng=42).random_base2(m=10)`，
`alpha = 0.004·5**u` 取对数空间、`Ku = 2000 + 28000·v` 取线性空间），
不读取任何外部清单/模板文件。直接运行即对每个 FIXED_CONFIGS ×
PARAMETERS 组合写出实验 YAML（`artifacts/generated_configs/<dataset_name>/`）
并启动模拟：脚本内 `MAX_WORKERS = 2` 为并发上限（可改的正整数，`1` 即
串行），有界线程池直接调用 `mumax3_pipeline.run_parameter_set`，不另起
Python 进程；底层 MuMax3 仍经 `external.py` 以 subprocess 参数列表调用。
控制台日志只有组级进度（时间、组序号、alpha/Ku、配置写出、模拟开始、
成功/失败与耗时、总进度），不含 Relax 或 pulse 内部实时进度；各组详细
run.log 仍由原机制写入。无 CLI 参数、无 plan/resume、无「目录已存在即
跳过」；不承诺不覆盖已生成的 YAML，已有 raw set 目录仍被 pipeline 拒绝。
发生失败：停止提交新任务、等运行中的组结束、保留现场并报错。单点或
补跑：改 `PARAMETERS` 只留尚未执行的目标后运行；失败残留须人工检查后
处置，不要自动删除或换 dataset_name 蒙混。**批量模拟执行前必须单独获得
用户批准**；截至 2026-09-09 该数据集尚未运行任何模拟，两并发在 GPU 上
的性能未实测（不声称占满显卡），输出目录与 `index.csv` 只是约定结构，
不构成科学正确性证明。

### 当前状态声明

- 模拟 pipeline（vertical slice）已实现（批量脚本 `generate_dataset.py`
  为唯一模拟执行入口，单份配置 CLI 已删除）；首版「样本准备 → 训练 →
  独立评估」工程已实现（见 `train.md`），**但未在正式研究数据上训练，
  不存在可靠科研结果或研究结论**。
- **CNN1D 分支**：P1 配置层与 P2 模型已实现（`model.kind` 判别 + CNN 四字段
  `channels`/`kernel_sizes`/`pool_bins`/`head_hidden_dims` 严格校验，YAML 与
  mapping 同一 schema；`CNN1DRegressor` 已实现于
  `src/micromagnetic_parameter_inversion/models/cnn1d.py`，并有 CPU 合成单元测试）。
  但模型**尚未接入** `training.train_model`/模型工厂，独立 checkpoint 与 CNN
  训练入口（`scripts/train_cnn1d.py`）也未实现，因此**不能经项目训练脚本训练
  CNN**；`configs/training/cnn1d.yaml` 仍为全注释占位、不可加载。原 MLP 流程与
  产物不变。CNN 与 MLP 的 checkpoint、训练产物与预处理统计**完全独立**，仅共用
  同一 dataset 与冻结 split，并在自身训练组上拟合统计量。
- 已完成的部分 QC：offline vertical slice 测试（31 passed）、2026-09-02
  test-only pilot 执行链冒烟、Pilot v1 哨兵轮轨迹层检查（历史协议）。
- **未证明**：连续模型网格收敛、Relax 收敛鲁棒性、EdgeSmooth 选择的
  系统论证、批量可复现性、OVF/物理级 QC、真实器件有效性、正向回代验证、
  训练侧科研有效性（未在正式研究数据上训练）。

## 物理模型概述

本项目模拟的物理系统与变化过程如下（详细推导见 `plan.md` 第 5–8 节；
几何与离散已由上文「当前协议」固定为固定离散 benchmark 的约定值——
连续模型网格收敛与真实器件有效性未证明，此为当前状态，不是待办门槛）。

### 模拟对象

一个扁平的三轴椭球薄纳米磁体（三轴全直径 `100 nm × 50 nm × 2 nm`，长轴
沿 x 方向，当前协议固定值），初始磁化沿 `+x`：

- 长轴提供明确的形状易轴，磁体接近单畴、又保留少量空间非均匀性；
- 计算网格小，适合 GPU 批量模拟；
- 固定 `Ms`（饱和磁化强度）、`A`（交换常数）、几何与网格，待反演参数只有
  Gilbert 阻尼系数 `alpha` 与单轴磁各向异性常数 `Ku`。

几何语义：`size_m=[dx,dy,dz]` 是椭球三轴全直径（= 包围盒尺寸），公共模型
段恒渲染完整三轴 `SetGeom(Ellipsoid(dx, dy, dz))`；`cells=[nx,ny,nz]` 各分量
为任意正整数。注意 `nz=1` **并不**使模型变成某个等价的恒厚柱体/薄片：
ES>0 时单层网格仍按平滑后的 Ellipsoid 几何填充（边界单元带部分填充权重），
与连续椭球并非等价替换；当前协议固定 `cells=[40, 20, 4]`（`nz=4`，逐层
解析 z 方向椭球表面）。cells 为固定离散选择，网格收敛未证明。

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

### 多激励设计（目标形态）

同一组 (`alpha`, `Ku`) 施加不同方向的短脉冲，利用磁场方向带来的敏感性差异
降低两参数间的混淆。模型方法的长远目标是多激励，但**当前协议（Protocol B）
只含单一 `pulse_A2`（y 向）baseline**；额外激励（如 B 沿 z）留作后续对照
实验，尚未批准执行。

| 激励 | 脉冲方向 | 当前状态 |
| --- | --- | --- |
| A | y（面内横向） | 当前协议唯一启用 |
| B | z（面外） | 后续对照候选，未批准 |
| C | 待定 | 暂缓 |

研究的核心逆问题：从一条或多条这样的平均磁化轨迹反演 (`alpha`, `Ku`)，再用 MuMax3 正向回代检验预测参数能否重建原始轨迹。

## 历史记录：Pilot v1 与 ES0 哨兵轮（已被 Protocol B 取代）

本节为**历史记录**：Pilot v1 协议选择过程与 ES0 哨兵轮实测数据/结果。
除标注「文献事实」外，数值均为项目选择或实测记录，不是文献结论。
本节内容不构成当前协议门槛；当前协议以上文「Protocol B」为准。

### 基准性质与固定候选（历史沿用至 Protocol B）

- **合成基准，不是复现**：0 K、无热噪声、单一均匀有效介质的
  synthetic CoFeB-inspired benchmark，仅借鉴 CoFeB 的典型量级，
  **不声称精确复现任何具体 CoFeB/GaAs stack**。
- **固定候选（可冻结）**：`Ms = 1.25e+6 A/m`、`Aex = 15.0e-12 J/m`、
  各向异性易轴 `+x`、真三轴椭球全直径 `100 × 50 × 2 nm`、
  `initial_m = +x`；材料抽象与采样坐标（关场后 `t = 0` 重锚定）一并视为
  协议约定。以上候选已被 Protocol B 采纳为固定值。
- 文献事实（量级参考，非取值来源）：椭球退磁因子见 Osborn (1945,
  DOI 10.1103/PhysRev.67.351)；单畴椭球进动频率见 Kittel (1948,
  DOI 10.1103/PhysRev.73.155)；CoFeB 材料量级可参考 Conca et al.
  (JAP 113, 213909 (2013), DOI 10.1063/1.4808462)。`Ms`/`Aex` 的具体取值
  是本项目对齐这些量级的选择；模拟工具见 MuMax3 论文
  (DOI 10.1063/1.4899186)。
- 参数主域与 `Ku = 0` control 约定自 Pilot v1 起沿用至今（见「当前协议」）。

### ES0 哨兵轮（历史数据/结果，非训练数据）

- 协议（历史）：`cells = 40 × 20 × 4` 且 **`EdgeSmooth = 0`**；两个 pulse
  （A 沿 y、B 沿 z），各 `10 mT`、`50 ps`；关场后每 `10 ps` 记录、共
  `1001` 点（0..10 ns）。
- 执行环境：MuMax3 3.12、Tesla T10；7 组 × 2 pulses 全部完整，每条轨迹
  `1001` 点（0..10 ns）。此为 `EdgeSmooth=0`/默认 solver 协议的
  **哨兵数据，不是训练数据**。
- 轨迹层健全性：全部数值有限，`mean|m| ≈ 0.99996–0.99998`（min ≥ 0.9995）。
- alpha/Ku 分离清晰（轨迹层）：dominant f 随 `Ku = 0/2k/16k/30k J/m^3` 约
  `6.79/7.09/8.69/10.19 GHz`，跨 alpha 不变；e-fold 随
  `alpha = .004/.008944/.020` 约 `1.82–1.95/.84–.85/.36–.40 ns`。
- pulse B=z 响应幅值与 normalized RMS 可分性约比 A=y 弱 5–6 倍。
- 历史阻塞（**仅属 ES0 协议，不是当前 Protocol B 的放行门槛**）：实测频率
  相对理想连续椭球宏自旋估算偏高约 2–7%（低 Ku 偏差最大），可能来自 ES0
  阶梯边界/网格/有限振幅等，因此当时未冻结 ES0 协议。

### 决策与历史方案处置

- Pilot v1 的「放行状态/下一步」表、32 点网格扫描方案（4 log-alpha ×
  8 linear-Ku 切片）与低 alpha 尾部分析计划**均已被 Protocol B 决策取代**：
  网格扫描停止，当前不计划任何网格扫描；该轮 ES0 数值协议控制尚未存在，
  其 ES0/默认 solver 记录为历史事实。数值协议（EdgeSmooth/solver/MaxErr/
  MaxDt/GammaLL/RelaxTorqueThreshold）现由 YAML `numerics` 块显式控制，
  Protocol B 固定为 ES12/solver 5（固定离散选择，非网格收敛结论）。
  首版训练/评估工程已实现（见 `train.md`），但未在正式研究数据上训练，
  不存在可靠科研结果。

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

## MuMax3 模拟入口

单份配置 CLI `scripts/run_mumax3_simulation.py --config <yaml>` 已删除，
不再可用；模拟执行唯一入口是 `scripts/generate_dataset.py`（见上文
「如何运行」）。单点或补跑不走命令行：改脚本内 `PARAMETERS` 只留尚未
执行的目标后运行。

`configs/experiments/mumax3_simulation.yaml` 仍是 Protocol B 固定协议的
模板与 schema 参照（**仅 `dataset_name`、`material.alpha`、
`material.ku_j_per_m3` 三处为 null 占位**；`load_config` 会拒绝任何仍为
null 的必填研究值）。批量脚本不读取此 YAML，其内置固定配置已对齐
Protocol B。训练/评估 pipeline 首版工程已实现（见 `train.md`），未做
正式研究训练。

YAML 中的指数数值请使用带指数符号的形式（如 `8.0e+5`）或直接写十进制
（如 `800000.0`）：`8.0e5` 这类不带符号的指数会被 PyYAML 解析为字符串，
随后被配置校验拒绝。

### 远端批量生成（Windows，历史记录）

本节记录旧 per-config CLI 时代的远端流程，曾在 Windows + MuMax3 机器
完成四组两批并行生成（`artifacts/` 下辅助脚本属运行产物，不入库、不作为
版本化入口）。**该流程依赖的 `scripts/run_mumax3_simulation.py --config`
入口已删除，整节不再适用于当前执行**；当前批量执行唯一入口是
`scripts/generate_dataset.py`（内置 `MAX_WORKERS` 并发，见「如何运行」）。
以下 launcher/SSH 机制仅作历史参考：

1. （旧流程）每参数组从模板复制一份运行 YAML 到 `artifacts/run_configs/<run>/`，
   填 `dataset_name`、`material.alpha`、`material.ku_j_per_m3`，经 `load_config` 校验并核对
   parameter_set_id；远端预检版本/依赖/GPU 与目标 set 目录不存在（不覆盖、
   不清理）；远端代码与本地不一致时不得上传覆盖源码。
2. （旧流程）每组独立调用已删除的 per-config CLI——该命令不再存在，
   不得执行；当前没有等价的单组命令行入口。
3. 脱离 SSH 会话用 CIM `Win32_Process.Create` 拉起 launcher（ASCII 脚本，
   `$PSScriptRoot` 定位），launcher 内以 `Start-Process` 并行启动每批两组，
   各自 stdout/stderr 日志、PID 与退出码；上一批结构验收通过后再启动
   下一批（SSH 会话内直接 `start`/`Start-Process` 可能随会话退出被结束）。
4. 启动后核对真实进程命令行、config.yaml 快照与原生输出文件；按 25 分钟
   间隔做有界轮询。失败暂停、保留现场、不自动重试；只按本任务记录的
   PID 停止。
5. 验收：equilibrium 与各 pulse 的 run.log 均为 returncode 0、index.csv 每 pulse 一条数据行、
   trajectory.csv 行数 = sample_count+表头、各 OVF 存在且非空、采样窗 =
   (sample_count-1)×interval。打包下载核对 SHA256；安全解压不覆盖已有，
   归档条目反斜杠规范化并拒绝绝对路径与 `..` 逃逸。

注意：PowerShell 5.1 `powershell -File` 不做表达式解析，逗号分隔不会自动
拆成数组（按标量参数传递）；`Start-Process -ArgumentList` 以空格拼接，
含空格路径需自行加引号；`-PassThru` 进程建议启动后先取一次 `.Handle`
固定句柄、`WaitForExit()` 后再读 `.ExitCode`（实测长等待中不取句柄会得到
空退出码，属单次观察而非普遍保证）。两组并行不保证占满 GPU。SSH 读写
遵循当前 SSH MCP 通道策略：可匹配只读策略用 read 通道，其余命令
经批准走 ask，文件传输走 ask（Windows 上传路径写 `/D:/...`）。结构验收
不等于科学放行：Protocol B 为固定离散 benchmark，网格收敛与真实器件
有效性未证明。

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
runs/cache 不入库；`data/README.md` 可跟踪，`data/samples/` 在白名单内
（Git 无法跟踪空目录，当前其下无已跟踪文件）；最终 `results/figures`、
`results/tables` 可跟踪。大数据与最佳模型未来走 Git LFS 或独立发布。
详见 `data/README.md`。

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
  通过。部分 QC 已完成：offline vertical slice 测试、执行链 pilot 冒烟、
  Pilot v1 哨兵轮轨迹层检查（历史 ES0 协议）；正式研究参数已由 Protocol B
  固定于实验 YAML 模板。**但连续模型网格收敛、Relax 收敛阈值/鲁棒性、
  EdgeSmooth 选择的系统论证、OVF 物理 QC 与批量可复现性、真实器件有效性
  及正向回代验证仍未证明。**

## 目录结构

```
src/micromagnetic_parameter_inversion/   # 包（runtime / external / paths）
                                         # + mumax3 vertical slice（config / script / results / pipeline）
                                         # + 训练/评估（training_config / training_data / preprocessing /
                                         #   models/mlp / training / evaluation）
scripts/check_environment.py             # 环境诊断
scripts/generate_dataset.py              # 唯一模拟执行入口：生成批量实验 YAML 并
                                         # 运行模拟（内置 Protocol B 固定配置与 Sobol
                                         # 1024 点；MAX_WORKERS=2 并发，1=串行；
                                         # 直接运行会启动模拟，执行前须获批准）
scripts/prepare_training_samples.py      # raw → data/samples npz/meta/split（train.md §2）
scripts/train_mlp.py                     # MLP 训练入口（train.md）
scripts/evaluate_model.py                # 独立 test 评估入口（train.md）
configs/base.yaml                        # 通用设置（seed、device=auto）
configs/experiments/mumax3_simulation.yaml
                                         # 实验配置模板（Protocol B 固定值；
                                         # 仅 dataset_name/alpha/Ku 三处待填）
configs/training/mlp.yaml                # 训练配置（dataset/run_name 占位待填）
train.md                                 # 首版训练/评估实现说明
simulations/mumax3/                      # MuMax3 脚本模板（.mx3.in）与说明
tests/                                   # pytest（无需 GPU/MuMax3）
data/                                    # 数据（完整数据不入库）
notebooks/                               # 探索性 notebook
results/figures, results/tables          # 最终结果（可跟踪）
```
