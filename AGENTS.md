# 仓库指引（micromagnetic-parameter-inversion）

## 项目范围与目标

- 本仓库是 MuMax3 多激励磁化动力学研究项目：基于多激励磁化动力学反演 Gilbert
  阻尼系数 `alpha` 与单轴磁各向异性常数 `Ku`（模型方向：MLP / 1D CNN /
  Temporal Transformer）。项目目标与设计以仓库内 `README.md` 与代码为准。
- **当前实现状态**：MuMax3 vertical slice 模拟 pipeline 与 CLI 已实现；
  首版「样本准备 → 训练 → 独立评估」工程已实现（见 `train.md`），但未在
  正式研究数据上训练，不存在可靠科研结果或研究结论。不得虚构或声称已有
  正式研究训练或可靠科研结论。可执行脚本包括：
  `scripts/check_environment.py`、`scripts/run_mumax3_simulation.py`、
  `scripts/generate_dataset.py`、`scripts/prepare_training_samples.py`、
  `scripts/train_mlp.py`、`scripts/evaluate_model.py`。
- **工作范围严格限于本仓库**：不得读取/编辑仓库外目录（`external_directory`
  已全局 deny），不引用仓库外研究计划路径；外部资料调研只走官方文档/论文。

## 环境与验证

- Python 3.13（`>=3.13,<3.14`），仅通过 `uv sync` 安装/同步；Linux 与 Windows
  均从 `pyproject.toml` 配置的 CUDA 12.8 索引解析 PyTorch。
- 环境诊断：`uv run python scripts/check_environment.py`。该脚本在 CUDA 或
  MuMax3 不可用时仍以 0 退出，须阅读其报告的状态而非依赖退出码。
- 已核实事实（2026-08）：Tesla T10 16GB passthrough、NVIDIA driver
  580.173.02、PyTorch 2.11.0+cu128、torch CUDA 12.8、CUDA available；用户级
  MuMax3 3.12（CUDA 12.9 build）已通过 `mumax3 -test`。**运行时仍以
  `scripts/check_environment.py` 与 `mumax3 -test` 输出为准**。
- 已核实事实（2026-09-02）：离线 vertical slice 测试 31 passed；在当前
  MuMax3 3.12/CUDA 12.9/Tesla T10 上完成一次临时 test-only 短 pilot
  （8×4×1 网格、1 pulse、3 samples，数值不代表研究参数）：模板可解析执行、
  Relax 产生 equilibrium.ovf、simulation 经固定相对路径 LoadFile 共享
  OVF、原生表头 `# t (s) mx () my () mz ()` 与恰 3 行、parser 时间重锚定、
  m_t0/m_tfinal、config.yaml 快照与 index.csv；`mumax3 -test` 同轮通过。
  **此为执行链冒烟验证，不是正式科研验证**：最终几何/网格收敛、Relax 收敛阈值/
  鲁棒性、EdgeSmooth 选择、物理 QC、批量可复现性、正向回代仍未验证。
- 测试：`uv run pytest`（无 GPU/MuMax3 也可运行）；聚焦文件/用例用标准 pytest
  节点，如 `uv run pytest tests/test_paths.py` 或
  `uv run pytest tests/test_paths.py::test_env_override_data_root`。
- 静态检查：`uv run ruff check .`、`uv run ruff format --check .`、
  `uv run pyright`；`uv run pre-commit run --all-files` 仅包装上述两个 Ruff
  检查（repo-local hooks）。
- `.env` 当前**不会自动加载**（代码未调用 `load_dotenv`）：需要环境变量时由
  shell 或 VS Code 显式注入，不要假设 `.env` 生效。

## 架构与代码组织

- `src/micromagnetic_parameter_inversion/paths.py` 负责数据/输出根目录解析，
  `runtime.py` 负责设备选择与诊断，`external.py` 负责 MuMax3 发现与执行。
- MuMax3 vertical slice 模块：`mumax3_config.py` 负责 frozen dataclass 与
  YAML 加载/校验（拒绝 null 研究值、mT→T 边界转换）；`mumax3_script.py`
  负责 `.mx3.in` 模板渲染；`mumax3_results.py` 负责 table 解析与轨迹 CSV
  导出；`mumax3_pipeline.py` 负责编排（equilibrium → 各 pulse、原字节
  config.yaml 快照与最终 index.csv 原子写出）。模拟入口：
  `scripts/run_mumax3_simulation.py --config <实验 YAML>`。
- 训练/评估首版模块（见 `train.md`）：`training_config.py`（严格 YAML
  加载/校验）、`training_data.py`（npz/dataset_meta/split/Dataset）、
  `preprocessing.py`、`models/mlp.py`、`training.py`（训练循环/early
  stopping/ckpt 读写）、`evaluation.py`（独立 test 评估）。
- 可复用代码放 `src/micromagnetic_parameter_inversion/`；notebook 仅用于探索
  （notebook 纪律：用项目 `.venv` 内核、不提交大数据/checkpoint 输出、不出现
  机器绝对路径）。
- 不硬编码机器路径：用 `MICROMAG_DATA_ROOT` / `MICROMAG_OUTPUT_ROOT`；
  MuMax3 通过 `MUMAX3_BIN` 或 `PATH` 选择。
- MuMax3 只经 `external.py` 调用，使用 subprocess 参数列表，禁止 `shell=True`。

## 科研完整性（硬约束）

- 不虚构 `alpha`/`Ku`/几何/激励范围：`configs/base.yaml` 只含已核实的
  机器无关默认值；研究参数来自研究计划或显式实验配置，**配置先于实验**。
- 固定 seed、依赖与代码版本（`configs/base.yaml` 的 `seed`、`uv.lock`、Git
  历史），保证可复现。
- 数据划分按参数组合进行，同一参数组合的所有激励必须同组（train/val/test），
  防止跨组泄漏。
- 防止标准化/特征拟合泄漏：统计量只在训练组上计算，验证/测试组不得参与拟合。
- 保留单位与物理量纲；报告插值/外推/噪声对结果的影响。
- 最终结论需 MuMax3 正向回代验证（反演参数 → 正向模拟 → 与观测对比）。

## 数据政策

- 完整数据（`data/raw/`、`data/processed/`）、checkpoints、TensorBoard
  runs/cache 不入库；白名单允许 `data/README.md` 与 `data/samples/` 下的
  少量样本文件（Git 无法跟踪空目录，当前 `data/samples/` 下无已跟踪
  文件），以及最终 `results/figures`、`results/tables`。详见
  `data/README.md`。

## 协作与执行纪律

- 使用 OMOS 自带 agent：`explorer` 做代码搜索、`librarian` 做官方
  PyTorch/MuMax3/科学资料调研、`fixer` 做边界明确的实现、`oracle` 做高风险
  架构/实验设计审查；不自行创建项目级 agent。
- **禁止主动创建 Git 分支**；Git commit/push/remote 等操作仅在用户明确要求时
  执行。
- GPU/MuMax3 长任务、批量模拟、训练、系统/驱动配置等操作执行前必须先呈计划
  并获用户批准。
