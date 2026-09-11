# micromagnetic-parameter-inversion

基于 MuMax3 磁化动力学反演 Gilbert 阻尼系数 `alpha` 与单轴磁各向异性常数
`Ku`。当前实现为**单激励 baseline**：y 方向 2 mT / 50 ps 短脉冲，关场后记录
空间平均磁化轨迹 `(m_x, m_y, m_z)`，用 MLP 回归 `(alpha, Ku)`；多激励（不同
方向脉冲以降低参数混淆）是项目长期目标，尚未启用。

**工程状态**：MuMax3 模拟生成 → 样本准备 → MLP 训练 → 独立 test 评估四段
pipeline 均已实现（入口与用法见下）。正式数据生成进行中，**尚未在正式研究
数据上训练，仓库内不存在可靠研究结果或结论**。1D CNN / Transformer 等后续
模型不在当前实现范围内。

## 物理模型与固定协议

MuMax3 在 0 K 下求解 Landau–Lifshitz–Gilbert 方程（Gilbert 显式形式，与
MuMax3 内核约定一致）：

```
dm/dt = -γ_LL/(1+α²) · [m × B_eff + α m × (m × B_eff)]
```

其中 `α` 即待反演的 `alpha`，`γ_LL`（`GammaLL`）取 MuMax3 默认正值约定
`1.7595e11 rad/(T·s)`，`B_eff` 为有效场（单位 T），含交换场、单轴各向异性
场、退磁场与外场四项；0 K、无热噪声项。单一均匀有效介质，synthetic
CoFeB-inspired 基准，不声称复现任何具体材料 stack；无 DMI / STT / 静态
偏置场，开放边界（`SetPBC(0, 0, 0)`，`EnableDemag = true`）。

- 几何：扁三轴椭球（非恒厚椭圆柱），`SetGeom(Ellipsoid(dx, dy, dz))` 的
  三轴全直径 `100 nm × 50 nm × 2 nm`（即 `size_m` 包围盒尺寸），
  `cells = [40, 20, 4]`；易轴与初始磁化均沿 `+x`。
- 固定材料/数值参数：`Ms = 1.25e6 A/m`、`Aex = 15e-12 J/m`、
  `EdgeSmooth = 12`（先于 `SetGeom` 设置）、`solver = 5`、`MaxErr = 1e-5`、
  `MaxDt = 1e-11 s`、`RelaxTorqueThreshold = -1`（官方默认）。
- 激励与采样：每个 `(alpha, Ku)` 参数组先运行一次
  [equilibrium.mx3.in](simulations/mumax3/equilibrium.mx3.in)（无外场
  `Relax()`，产出全部 pulse 共享的平衡态）；随后
  [simulation.mx3.in](simulations/mumax3/simulation.mx3.in) 加载该平衡态、
  设入真实 `alpha`，施加 y 方向 2 mT / 50 ps 矩形脉冲后精确关场
  （`B_ext = 0`），自关场时刻起每 10 ps 记录一次空间平均磁化
  `(m_x, m_y, m_z)`，共 401 点（关场后 0..4 ns）。
- 待反演参数：`alpha ∈ [0.004, 0.020]`（对数空间采样）、
  `Ku ∈ [2000, 30000] J/m^3`（线性空间采样），Sobol 1024 点
  （`scramble=True, rng=42`）；`Ku = 0` 仅作物理 control，不计入主域误差。
- 以上为**固定离散约定**：网格收敛与真实器件有效性均未验证，不代表收敛
  结论。

协议数值的唯一来源是
[scripts/generate_dataset.py](scripts/generate_dataset.py)（`FIXED_CONFIGS` /
`PARAMETERS`）；YAML 校验与 mT→T 单位换算边界见
[src/micromagnetic_parameter_inversion/mumax3_config.py](src/micromagnetic_parameter_inversion/mumax3_config.py)；
模板渲染（公共模型段与椭球几何/材料参数的唯一渲染器）见
[src/micromagnetic_parameter_inversion/mumax3_script.py](src/micromagnetic_parameter_inversion/mumax3_script.py)；
模拟编排与轨迹导出见
[src/micromagnetic_parameter_inversion/mumax3_pipeline.py](src/micromagnetic_parameter_inversion/mumax3_pipeline.py)
与
[src/micromagnetic_parameter_inversion/mumax3_results.py](src/micromagnetic_parameter_inversion/mumax3_results.py)；
协议模板与字段说明参照
[configs/experiments/mumax3_simulation.yaml](configs/experiments/mumax3_simulation.yaml)。

## MLP 架构与训练（要点）

baseline 模型为纯 MLP 回归器
（[src/micromagnetic_parameter_inversion/models/mlp.py](src/micromagnetic_parameter_inversion/models/mlp.py)）：

- 输入契约：单样本为某参数组全部 pulse 的原始时域轨迹 `[P, T, 3]`（batch
  `[N, P, T, 3]`，通道为 mx/my/mz）；不做每 pulse 切分、手工统计特征或
  降采样。当前协议 `P = 1`（仅 `pulse_A2`）、`T = 401`，展平维度
  `D = P·T·3 = 1203`。
- 网络：`Flatten` 后接隐层序列（默认 `64 → 32 → 32`，每层 `Linear + ReLU`），
  末层 `Linear` 双输出 `(alpha, Ku)` 的**标准化标签值**（z-score 空间，非
  物理单位）。`hidden_dims` 在
  [configs/training/mlp.yaml](configs/training/mlp.yaml) 配置，属工程候选，
  非已验证科研参数。
- 预处理
  （[src/micromagnetic_parameter_inversion/preprocessing.py](src/micromagnetic_parameter_inversion/preprocessing.py)）：
  输入统计量仅在 train 组拟合，沿样本维与时间轴聚合为每 pulse 位置、每
  磁化分量的 mean/std（形状 `[P, 1, 3]`，广播作用于整条轨迹；
  `std <= std_eps`（默认 `1e-8`）的位置除数取 1）。标签变换默认
  `identity`；可选 `logalpha` 仅对 alpha 列取 **log10**（float64 计算，要求
  alpha > 0，非自然对数），随后逐输出 z-score。
- 训练
  （[src/micromagnetic_parameter_inversion/training.py](src/micromagnetic_parameter_inversion/training.py)）：
  Adam（默认 `lr = 1e-3`、`weight_decay = 0`），标准化标签空间 MSE 损失，
  `batch_size = 32`、`max_epochs = 500`、`seed = 42`；early stopping
  （`patience = 50`、`min_delta = 0`）用独立 reference，与绝对 best 分开；
  val loss 绝对最优权重写 `best.pt`，最后完成 epoch 权重写 `final.pt`；
  非有限 loss/grad/pred 立即停止且不保存坏权重。
- 评估
  （[src/micromagnetic_parameter_inversion/evaluation.py](src/micromagnetic_parameter_inversion/evaluation.py)）：
  网络结构、预处理与标签变换全部从 checkpoint 恢复（不经当前 YAML），预测
  经逆变换还原物理单位后报告 alpha/Ku 的 MAE/RMSE（Ku 单位 J/m^3）；主域
  排除 `Ku = 0` 的 control，control 单独报告。

## 安装与前置条件

- Python 3.13（`>=3.13,<3.14`），由 [uv](https://docs.astral.sh/uv/) 管理；
  在仓库根目录运行 `uv sync` 创建 `.venv` 并安装依赖。PyTorch 来自官方
  CUDA 12.8 wheel 索引（x86_64 Linux / Windows，见 `pyproject.toml`）。
- GPU 驱动与 [MuMax3](https://mumax.github.io/) 均为外部前提，本项目不安装、
  不打包：MuMax3 加入 `PATH` 或设置 `MUMAX3_BIN`。
- 环境诊断：

  ```bash
  uv run python scripts/check_environment.py
  ```

  报告 Python、包版本、CUDA 可用性与设备、MuMax3 状态、数据/输出路径；CUDA
  或 MuMax3 缺失时仍以 0 退出，须阅读报告内容而非依赖退出码。

## 使用流程（四个入口）

以下命令均在仓库根目录运行，参数以各脚本的真实 CLI 为准。

### 1. 生成原始数据

```bash
uv run python scripts/generate_dataset.py
```

- 无 CLI 参数：固定协议字段与 1024 个 Sobol 参数点写在脚本内
  （`FIXED_CONFIGS` / `PARAMETERS`），**不读取外部 YAML/清单**；
  `configs/experiments/mumax3_simulation.yaml` 只是协议模板与 schema 参照，
  不是生成入口。
- 前置：MuMax3 可用；目标输出目录 `data/raw/<dataset_name>/<psid>/` 不存在
  （已存在会被拒绝，不覆盖、不清理）。
- 并发由脚本常量 `MAX_WORKERS` 控制（`1` 为串行），无 resume/跳过；单点或
  补跑改为在脚本内 `PARAMETERS` 中只保留尚未执行的目标。
- 生成配置写在 `artifacts/generated_configs/<dataset_name>/`。批量执行会
  长时间占用 GPU，运行前请评估本机 GPU 资源与预计耗时。

### 2. 准备训练样本

```bash
uv run python scripts/prepare_training_samples.py \
  --config configs/training/mlp.yaml --parameter-set-ids all
```

- 前置：把 `configs/training/mlp.yaml` 的 `dataset_name` 占位符替换为实际
  dataset；`--config` 与 `--parameter-set-ids` 均为必填（后者为显式 psid
  列表或 `all`，`all` 只展开所选 dataset 目录）。
- 输出 `data/samples/<dataset_name>/`：每参数组一个 `<psid>.npz`，外加
  `dataset_meta.yaml` 与 `split.yaml`；写前预检、拒绝覆盖，失败可能留下
  部分产物，须人工清理后重跑。

### 3. 训练（仅 train/val）

```bash
uv run python scripts/train_mlp.py --config configs/training/mlp.yaml
```

- 前置：替换 `run_name` 占位符；输出目录不存在。
- 标准化/标签统计量只在 train 组拟合；`test` 不参与调参或模型选择。
- 输出目录默认 `artifacts/training/mlp/<dataset>/<run_name>/`，含
  `best.pt` / `final.pt` / `metrics.json` / `split.yaml` 等。

### 4. 独立 test 评估

```bash
uv run python scripts/evaluate_model.py --run <RUN_DIR>
```

- `--checkpoint` 可选（缺省 `<RUN_DIR>/best.pt`）；checkpoint 与 run 内 split
  副本 SHA 绑定。
- 在 run 目录写出 `test_metrics.json`（main/control 的 MAE/RMSE，物理单位）
  与 `test_predictions.csv`；产物已存在则拒绝覆盖。

## 数据与输出

- `data/raw/`：MuMax3 原始输出；`data/samples/`：预处理样本；`artifacts/`：
  运行产物与 checkpoint。原始数据、样本与产物均不入 Git，对外公开的仓库只含
  代码与文档。
- 数据与输出路径可用环境变量覆盖：`MICROMAG_DATA_ROOT`、
  `MICROMAG_OUTPUT_ROOT`（见
  [src/micromagnetic_parameter_inversion/paths.py](src/micromagnetic_parameter_inversion/paths.py)）；
  MuMax3 可执行文件用 `MUMAX3_BIN`（见
  [src/micromagnetic_parameter_inversion/external.py](src/micromagnetic_parameter_inversion/external.py)），
  未设置时按平台默认名在 `PATH` 查找。
- **`.env` 不会自动加载**（代码未调用 `load_dotenv`）：需要环境变量时由
  shell 或运行环境显式注入，不要假设放置 `.env` 文件即生效；`.env` 由
  `.gitignore` 排除，不会入库。
- 仓库尚未选择开源许可证（LICENSE 未添加）。

## 科研完整性约定

- 数据划分按参数组合 `(alpha, Ku)` 分组：同一组合的所有激励轨迹必须同组
  （train/val/test），防止跨组泄漏。
- 标准化与标签统计量仅在训练组拟合；test 独立评估；训练 seed 取
  `configs/training/mlp.yaml` 的 `training.seed`，生成数据的 Sobol seed
  固定在 `scripts/generate_dataset.py`；依赖固定于 `uv.lock`，不虚构参数范围。
- 最终结论需 MuMax3 正向回代验证（反演参数 → 正向模拟 → 与观测对比），
  当前尚未执行。
- **尚未验证**：网格收敛、Relax 收敛阈值与鲁棒性、EdgeSmooth 选择的系统
  论证、批量可复现性、OVF 物理级 QC、真实器件有效性、正向回代，以及在
  正式研究数据上的训练有效性。
