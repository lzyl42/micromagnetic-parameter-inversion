# MLP 训练反演：首版实现说明（工程完成，非正式研究验证）

**状态**：prepare / train / evaluate 首版代码已实现（含本轮新增的停止
原因记录与 test_metrics provenance），工程验证以聚焦离线测试（tmp 合成
数据、CPU）为准；**未在正式研究数据上训练，不存在可靠科研结果，未做
科研有效性验证**。`configs/training/mlp.yaml` 的 `dataset_name`/`run_name`
仍为占位符，运行前必须替换（`load_config` 拒绝占位值），因此下述命令在
占位状态下不会成功。现有 `data/raw/` 各数据集（哨兵/QC 轮）仅验证过格式
与执行链，不构成科研训练有效性依据。

> **未实现的 CNN1D 分支**：1D CNN 反演目前只有骨架与注释
> （`src/micromagnetic_parameter_inversion/models/cnn1d.py`、
> `scripts/train_cnn1d.py`、`configs/training/cnn1d.yaml` 不可执行、
> `tests/test_cnn1d.py` 无测试函数、pytest 不收集），**不能训练**；所有正式
> 实现待逐步批准。CNN 与 MLP 的 checkpoint、训练产物与预处理统计**完全独立**：
> 只共用同一 dataset 与冻结 split，统计量由 CNN 在自身训练组上拟合，不复用
> MLP 权重/checkpoint/已有预处理统计/训练产物。
>
> **未来 CNN1D 入口与 checkpoint（未实现）**：共享编排未来从现
> `scripts/train_mlp.py` 的 `run()` 抽到既有 `training.py`，保持命令/默认行为、
> 显式 `output_dir`、构造模型前设定 seed、best/final 与数值失败语义不变；
> `train_mlp.py` / `train_cnn1d.py` 为薄入口，各自在读数据、建目录**之前**校验
> `model.kind`。checkpoint 不做统一 v2：MLP 的
> `Checkpoint`/`save_checkpoint`/`load_checkpoint` 与格式原样不动，CNN 用独立的
> `CNNCheckpoint`/`save_cnn_checkpoint`/`load_cnn_checkpoint`（草案名，与核心实现
> 统一）及独立 `cnn1d` 格式，二者互不接受对方文件、损坏 CNN 不回退 MLP。细节
> 大纲见 `tests/test_training_config.py`、`tests/test_training.py`、
> `tests/test_evaluation.py` 的 `TODO(CNN1D-*)` 注释。本文档描述的
> prepare/train/evaluate 流程目前仍只覆盖 MLP。

## 1. 目标与非目标

- 目标：把已生成的 MuMax3 raw 输出规范化为模型可直接读取的 npz 样本，训练
  MLP 反演 `alpha`、`Ku`。网络隐层规模采用本计划的首批工程候选
  `64/32/32`（是否扩大由学习曲线决定）。
- **首版已实现**：prepare（raw → npz/meta/split）、MLP 训练（仅 train/val）、
  独立 evaluate（test）与 checkpoint 读写；工程验证为聚焦离线测试（见第 9
  节）。正式数据实验、超参搜索与科研结论（含 MuMax3 正向回代）不在首版
  范围（见第 10 节待定研究项）。
- 不重复物理 QC：生成侧 `parse_table` 已校验每行 |磁化分量| <= 1+1e-6、
  时间网格与行数；首版不做准入 manifest、不验证数据科学合法性、不逐组
  重查协议一致性——用户显式选定单个 dataset 即由用户保证单协议。

## 2. 数据规范化：raw → data/samples/&lt;dataset_name&gt;/（prepare 脚本）

输入布局（已核实，`mumax3_pipeline.py`）：
`data/raw/<dataset_name>/<parameter_set_id>/{config.yaml, index.csv, equilibrium/, runs/<pulse_id>/trajectory.csv}`；
index 列固定 `parameter_set_id,pulse_id,alpha,ku_j_per_m3,b_ext_x_T,b_ext_y_T,b_ext_z_T,pulse_duration_s,trajectory_path`；
轨迹表头固定 `sample_index,t_s,m_x,m_y,m_z`，t=0 为关场时刻（生成侧已重锚定）。

处理顺序（`scripts/prepare_training_samples.py`）：

1. `--config configs/training/mlp.yaml` 提供 `dataset_name`（必填）；
   `--parameter-set-ids all | <id...>` 显式选择参数组，`all` 只展开该
   dataset 目录下的子目录。**绝不扫描跨 dataset 的全部 raw**；未给出该
   参数即报错；跨 dataset 引用一律拒绝。
2. 每个参数组读其 `index.csv`；只做读入必需校验：文件存在、列名齐全、
   `trajectory_path` 相对路径存在；缺失时给出含完整路径的友好错误并终止。
3. 协议元信息只读不查：「首份」= `--parameter-set-ids` 选中的 psid **升序
   排序后的第一个**参数组的 `config.yaml` 快照；用 `yaml.safe_load` 只提取
   `recording`（sample_interval_s、sample_count → T）与 `pulses`（pulse_id、
   duration、b_ext），**不走** `mumax3_config.load_config`（避免引入研究值
   校验语义）。旧快照缺 `numerics` 等节不视为错误（只读取上述两节）；mT→T
   换算为 `b_ext_T = b_ext_amplitude_mT × 1e-3 × direction`，时长读取
   `duration_s`（秒）。据此冻结协议摘要与 pulse 顺序：配置
   `data.pulse_order` 非 null 时，必须是首份 config pulse 集合的一个**无
   重复排列**，任一单组相对该顺序缺失或多出 pulse 即报错；不逐组做协议
   物理审核。结果记录进 `dataset_meta.yaml`。此后所有 npz 使用同一冻结
   顺序；**不逐文件依赖各自 index 行序**，也不逐组比对 pulses/recording
   是否一致。
4. 每行读 `trajectory.csv`：`np.loadtxt(path, delimiter=",", skiprows=1,
   usecols=(2, 3, 4), ndmin=2)`，即取 `m_x, m_y, m_z` 三列（**不含**
   sample_index 与 t_s；`ndmin=2` 防止单行文件被降为一维）得 `[T,3]`，转
   float32。源 CSV 表头已按上述布局契约检查，此处只做形状必需检查
   `shape == (T, 3)`（T = 冻结 sample_count）。
   冻结 pulse 顺序中某 pulse 的 trajectory.csv 缺失 → 含完整路径的友好
   报错；数据**按冻结 pulse 顺序 stack** 为 `[P, T, 3]`。另读**首条**
   轨迹的 `t_s` 列（`ndmin=1`）保存为该组 `t_s`，无需任何重校验。
5. 标签 `(alpha, ku_j_per_m3)` 取自该组 index（原始物理单位：alpha 无量纲、
   Ku 单位 J/m^3）；同组各行必须一致（否则组标签无法定义，报错）。
   x/y 均不做任何跨数据预标准化（第 4 节按训练组拟合）。
6. 每参数组写一个 npz 到 `data_root()/samples/<dataset_name>/<psid>.npz`：
   - `x: float32 [P, T, 3]`（P = 冻结 pulse 顺序长度，通道序 mx,my,mz，
     全部原始时域轨迹）
   - `y: float32 [2]` = (alpha, ku_j_per_m3) 原始单位
   - `parameter_set_id: <U16` 标量；`t_s: float64 [T]`（实际时间网格，
     取自该组首条轨迹的时间列）；`pulse_ids: <U… [P]`
   - 写出直接用 `np.savez_compressed`，内容**只有数值数组与定长 unicode
     数组，绝无 object 数组**；读取一律 `np.load(..., allow_pickle=False)`。
7. 元信息独立文件 `data_root()/samples/<dataset_name>/dataset_meta.yaml`：
   dataset_name、协议摘要（冻结 config 的 recording/pulses）、冻结的
   pulse 顺序、psid→(alpha, Ku) 表、每组源 index 相对路径与 config.yaml
   的 sha256（轻量溯源）、生成时间、以及**本清单成员 psid 列表**——后续
   Dataset 按 split 显式成员构造（见第 3 节）。
8. 生成前预检（不引入复杂事务）：任一目标 npz、`dataset_meta.yaml` 或
   `split.yaml` 已存在 → **在写任何样本之前**报错拒绝（不隐式覆盖）；
   split 有效性（比例、`min_per_split`）同样在写样本前判定，尽量避免
   半成品。更换参数集合须更换输出位置或先人工清理，防止旧文件混入。
   写出按序进行、无事务：中途失败可能留下部分产物；下次运行因预检
   拒绝覆盖，须人工清理后重跑。
9. 数据政策：完整样本生成物**不入普通 git**。`.gitignore` 与
   `data/README.md` 已随首版实现同步落实：生成的
   `data/samples/<dataset_name>/` 子目录被忽略（`data/samples/*` 白名单
   允许显式添加的少量样本文件；Git 无法跟踪空目录，当前 `data/samples/`
   下无已跟踪文件），小样本文件如需入库用 `git add -f` 显式添加；输出
   位置仍为 `data/samples/<dataset_name>/`。不新增 Git 执行门禁、不强制
   新增 README 或 LFS；完整数据入库渠道留待研究决策（见第 10 节）。

## 3. 分组划分（split）

- split 单位 = `parameter_set_id`（即 npz 文件），**按 ID 分组**划分。
  psid 仅由 (alpha, Ku) 决定（SHA-256 前 16 位），未来同一参数的噪声副本
  或重复运行继承同一 psid，因此与原组同组；这是分组规则约定，并用测试
  验证划分后各组 psid 互斥且并集完备（不声称结构上天然不可能泄漏）。
- 配置块（仅 prepare 脚本使用）：`split.seed: 42`、
  `split.ratios: {train: 0.7, val: 0.15, test: 0.15}`、
  `split.min_per_split: {train: 2, val: 1, test: 1}`，均可配置。ratios
  必须全部非负且和为 1，否则报错。
- 算法：psid 列表排序后用 `np.random.default_rng(seed).permutation` 打乱，
  按 train→val→test 顺序用**最大余数法**分配——各组先取 `floor(n·ratio_i)`，
  剩余名额按小数余数降序逐个 +1，余数并列时固定按 train→val→test 优先
  （确定性）。分配完成后检查 `min_per_split`，任一组不足 → 直接报错退出，
  不硬凑、不静默合并、不为通过检查二次重切。split.yaml 生成后即快照：
  raw/参数集合后续变化**不保证**旧成员划分不变；旧训练 run 一律绑定其
  run 内保存的 split 副本（见第 8 节），不依赖事后重生成的 split.yaml。
- 划分清单独立保存 `data_root()/samples/<dataset_name>/split.yaml`：
  `{seed, ratios, train: [psid...], val: [...], test: [...]}`。训练脚本按
  该 split.yaml、evaluate 按 run 内保存的 split 副本（唯一权威，经
  `load_split` 的 `split_path` 参数加载，见第 8 节）的**显式成员**构造
  Dataset，并校验：每个成员存在于
  dataset_meta 清单（对应 npz 存在）、三组互斥、并集恰等于清单成员；
  任一不符即报错——**不做交集过滤**（交集会静默丢失成员），也**不允许
  运行时自行重切**。
- test 不参与任何调参与模型选择；early stopping 与 best 选择只看 val。

## 4. 输入与标签预处理（training_data.py + preprocessing.py）

- **一个训练样本 = 一个参数组的全部 pulse**：`x[P, T, 3]` 全原始时域轨迹，
  无每 pulse 切分、无手工统计特征、无降采样。
- 输入标准化（仅 train 组拟合）：对训练样本沿**样本维与时间轴**聚合，得到
  每个 pulse 位置、每个磁化分量的 mean/std，形状 `[P, 1, 3]`，广播作用于
  `[N, P, T, 3]`。`std <= eps`（默认 1e-8，可配置）→ 除数取 1；保存的
  `std` 即 **effective scale**（`<= eps` 位置恒为 1.0，原始 std 不保存）。
  注意该 pulse 位置通道在 val/test 上的变换结果**不保证**为 0（统计不
  来自那些组），零方差位置清单记入 preprocessing 元信息。
- 标签：`label.transform: identity`（默认）或 `logalpha`——仅对 alpha 取
  log10，要求 alpha > 0 否则报错退出，**无静默裁剪**；Ku 保持线性。注意：
  log(alpha) 空间采样是设计选择，**不等于**强制标签 log 变换，两者独立。
- 标签标准化：变换后对 (alpha', ku) 逐输出 z-score（train 拟合）；y_std
  同样保存 **effective scale**（`<= eps` 位置为 1.0）并记录零方差输出
  （防 train 内某输出零方差——如同 alpha 场景——在 val/test 上除 0）；
  val/test 复用同一统计量；逆变换所需全部参数随 ckpt 保存。

## 5. 模型与优化（models/mlp.py + training.py）

- 输入维度 `D = P × T × 3`，模型内 `Flatten([P,T,3] → D)` 后接 MLP：
  `Linear D→h1 → ReLU → … → Linear→2`（线性输出，无激活/约束）；
  `model.hidden_dims: [64, 32, 32]` 配置化。参数量参考：[64,32,32] 时为
  `64·D + 3266`（首层 D→64 主导），仅作容量参考。
- 损失：MSE（标准化标签空间）。优化器：Adam。
- **初始超参为工程候选，不是已验证的科研参数**，仅供起步：
  `lr=1e-3, weight_decay=0.0, batch_size=32, max_epochs=500,
  early_stopping={patience: 50, min_delta: 0.0}`。
- 规模感：样本数 = 参数组数量级（组级样本，非 pulse 级）。Dataset 保留在
  CPU，DataLoader 按 batch 取数、每个 batch 传至 `device`，**不假设整个
  数据集能放进 GPU 显存**；batch_size 大于 train 样本数时等价全批量。

## 6. 代码目录与职责（首版已实现，复用现有模块）

```
src/micromagnetic_parameter_inversion/
  training_config.py   # frozen dataclass + YAML 严格加载/校验（风格对齐 mumax3_config）
  training_data.py     # npz 读取/清单成员与形状校验/按 split.yaml 建 torch Dataset
  preprocessing.py     # [P,1,3] 广播标准化、标签变换与逆变换
  models/mlp.py        # MLP 定义（携带输入契约：P、T、C=3、pulse 顺序）
  training.py          # 训练循环、early stopping、ckpt 读写
  evaluation.py        # val/test 评估、物理单位指标、test_predictions 导出
scripts/
  prepare_training_samples.py  # raw → samples/<dataset>/ npz + dataset_meta + split
  train_mlp.py                 # 配置 → 样本/划分 → 训练 → artifacts 产物
  evaluate_model.py            # run/ckpt 定位 → run 内 split 副本的 test 指标
                               # 与 test_predictions.csv
configs/training/mlp.yaml       # 训练配置（见第 7 节）
```

复用：`paths.data_root()/output_root()`（不硬编码机器路径）、
`runtime.select_device(config.training.device)`；协议元信息按第 2 节用
`yaml.safe_load` 只读首份 config.yaml 快照的 recording/pulses（不经
`mumax3_config.load_config`）。依赖只用现有 numpy/torch/pandas/pyyaml，
**不新增依赖**。

## 7. 配置字段与调用流程

`configs/training/mlp.yaml`（示例中 dataset 与 run_name 为**占位符**，运行
前替换为实际值，不虚构研究参数）：

```yaml
dataset_name: PLACEHOLDER_DATASET_NAME   # 占位：替换为 data/raw 下实际 dataset
run_name: PLACEHOLDER_RUN_NAME           # 必填（无默认、不可为 null/空）：
                                         # 输出目录已存在则拒绝启动，不隐式覆盖
data:
  pulse_order: null    # null = 从所选首份 config.yaml 冻结并记录进 dataset_meta；
                       # 非 null = 显式 pulse_id 列表（须与协议一致）
model:
  hidden_dims: [64, 32, 32]
label: {transform: identity}             # identity | logalpha
preprocessing: {std_eps: 1.0e-8}
training:
  seed: 42            # 与 configs/base.yaml 一致
  device: auto        # 交 runtime.select_device
  batch_size: 32
  max_epochs: 500
  learning_rate: 1.0e-3
  weight_decay: 0.0
  early_stopping: {patience: 50, min_delta: 0.0}
split:                # 仅 prepare 脚本使用
  seed: 42
  ratios: {train: 0.7, val: 0.15, test: 0.15}
  min_per_split: {train: 2, val: 1, test: 1}
output_dir: null      # null = output_root()/training/mlp/<dataset_name>/<run_name>
```

调用流程（首版接口；占位符未替换时 `load_config` 拒绝运行，替换后按
实际 dataset 使用）：

```bash
uv run python scripts/prepare_training_samples.py \
  --config configs/training/mlp.yaml --parameter-set-ids all
uv run python scripts/train_mlp.py --config configs/training/mlp.yaml
uv run python scripts/evaluate_model.py --run RUN_DIR
```

evaluate 的 `--checkpoint CKPT` 为可选（默认 `<run>/best.pt`）；无论显式
与否，ckpt 必须与 run 内 split 副本 SHA 绑定一致，结构/预处理全部取自
ckpt，不要求提供当前训练 config 重建模型。

## 8. 训练循环、指标与产物

- seed 入口：模型初始化**之前**设定 `torch.manual_seed(seed)` 与 numpy
  种子（均取 `training.seed`）；DataLoader 使用显式 seeded
  `torch.Generator`（shuffle 可复现）；首版 `num_workers=0` 可用。不承诺
  跨硬件位级一致，可复现性以同机同版本为准。
- 循环：`model.train()` 训练，每 batch 依次 `optimizer.zero_grad()` →
  `loss.backward()` → `optimizer.step()`；`model.eval()` +
  `torch.no_grad()` 评估。每 epoch 计算 train 与 val 的标准化 MSE；
  best = 历史 val MSE 最低的权重（严格小于才更新）。early stopping 与
  best 选择相互独立，使用独立的「显著改善」reference（初始 +inf）：仅当
  当前 val MSE 相对该 reference 的改善**严格 > `min_delta`** 时，
  reference 才推进到当前值并把 patience 计数清零；否则计数 +1，连续
  `patience` 个 epoch 无显著改善即停止。绝对 best（严格更低即更新）独立
  推进，不受 `min_delta` 影响。任一 loss 非有限（NaN/Inf）→ 立即停止
  训练，不把非有限值写入 history/ckpt。训练只使用 train/val；test 不在
  任何自动流程中评估（含 metrics.json）。
- 停止记录：训练结束记录 `stop_reason`（`max_epochs` / `early_stopping` /
  `numerical_failure`）、`stop_epoch` 与 `detail`（记录停止时的 metrics），
  随 run 产物保存。数值失败（非有限 loss/grad/pred）以**非零退出码**
  结束；此时保留的此前有效权重（best/final）**不代表训练成功**。
- 指标（物理单位，alpha 与 Ku **分别**报告）：首版仅 MAE、RMSE；主域排除
  Ku = 0 的物理 control（按 dataset_meta 的 psid→Ku 表识别），control
  单独一行报告、不计入主域指标；相对误差不在首版指标内（未来可选，届时
  再定义零值剔除与计数口径）。每个指标子集同时输出样本数 `n`；某子集为
  空（如 control 为空）时该子集指标输出 `null`（JSON null，**非 NaN、
  非 0**）；不得为凑出非空指标重新切分子集或追加剔除样本。预测含非有限
  值 → 报错，不当作空子集。
- 常规 evaluate（`scripts/evaluate_model.py --run RUN_DIR`）：split 仅
  从该 run 保存的 split 副本定位（接口不接收 split 参数；经
  `load_split(..., split_path=副本)` 走同一校验边界）；checkpoint 保存
  该副本的 SHA-256，加载时核对，不一致即报错（防另选 checkpoint 与 run
  划分错配）；dataset_meta 按锚点 `data_root()/samples/<dataset>/` +
  ckpt 内 relpath 定位（resolve 强制留在锚点内，防逃逸）并核对 SHA。
  网络结构、预处理与 label 变换**全部从 checkpoint 恢复**，当前 YAML
  不得覆盖或重新拟合任何统计量。评估前校验每个 npz 与 ckpt 输入契约
  一致：`x` 形状 `[P,T,3]`、`pulse_ids` 顺序、`t_s` 数组，任一不符即
  报错——这是契约校验，**不是物理 QC**。`run_evaluation` 返回
  `(report, rows)`；推理在 CPU 上按 ckpt 的 batch_size 分批
  （eval/`no_grad`），产物 `test_metrics.json`（main/control 的 n 与
  MAE/RMSE，另含 provenance：实际使用的 checkpoint 路径、checkpoint
  sha256 与 split sha256）与 `test_predictions.csv`（行序按 split.test
  成员顺序）写入 run 目录，各只写一次、拒绝覆盖旧评估。
- 输出目录（`output_root()/training/...`，不入库）：
  `config_resolved.yaml`（生效配置快照）、split.yaml 副本（唯一权威
  split；**样本元信息不完整复制进 run**——run/ckpt 以锚点
  `data_root()/samples/<dataset>/` 内的相对路径 + sha256 引用外部
  `dataset_meta.yaml`）、`preprocessing.yaml`（[P,1,3] mean/effective
  scale std、eps、零方差位置清单、label transform、y mean/std）、
  `metrics.json`（逐 epoch train/val MSE history + best val；**无 test
  评估**——test 只由独立 evaluate 触发）、`best.pt`、`final.pt`。
- ckpt 磁盘形态：`torch.save` 字典仅含 Python primitives/list/dict +
  CPU Tensor（dataclass/numpy 对象一律展开为纯容器）；读取显式
  `torch.load(weights_only=True, map_location="cpu")`，校验格式版本与
  关键字段/契约形状后重建嵌套 dataclass（不经临时文件）。
- ckpt 内容包含：
  - `ckpt_format_version`（ckpt schema 版本号）
  - `model_state_dict`
  - 模型结构显式字段：`hidden_dims`（如 [64,32,32]）与激活函数配置
    （不只藏在 config 副本里）
  - 输入契约：P、T、C=3、pulse 顺序（冻结 pulse_id 序列）、磁化分量序
    （mx,my,mz）、实际时间网格（t_s 数组）；dataset_meta 引用（锚点内
    相对路径 + sha256）——**不内嵌协议摘要**，协议元信息仍以外部
    `dataset_meta.yaml` 为准（评估时按锚点定位并核对 SHA）
  - 预处理状态：x 的 [P,1,3] mean 与 effective scale std、eps、零方差
    清单、label transform、y mean/effective scale std 及逆变换所需全部
  - seed、代码版本（可得时记录 git SHA + dirty 标记）、torch/numpy 版本、
    config 副本、best val loss
  目标：仅凭 ckpt + run 内 split 副本 + 锚点下经 SHA 核对的
  dataset_meta/npz 即可独立推理。resume 断点续训**首版不必须**。

## 9. 验证范围（tests/，聚焦离线测试，无需 GPU/MuMax3）

已实现的聚焦离线测试（tmp 合成数据，不依赖 data/raw；通过状态以
`uv run pytest` 当前结果为准）：(a) 最小
合成 raw 布局——
`config.yaml` + `index.csv` + `runs/<pulse>/trajectory.csv`；(b) 直接合成
小型 npz（形状正确、数值任意）与 npz/meta/split 直写。用例：

1. npz 读写往返：schema/dtype/形状、无 object 数组、`allow_pickle=False`
   读取成功；缺文件、行数 != T 的报错消息含路径与期望形状。
2. 列选取与组装：x 仅取 m_x,m_y,m_z 三列（sample_index/t_s 不进 x）；t_s
   取自首条轨迹时间列；x 按冻结 pulse 顺序 stack 为 [P,T,3]；缺 pulse 报
   错含路径；覆盖 T=1 的形状保持与缺 numerics 的历史快照读取。
3. split：psid 各组互斥、并集完备、同 seed 结果确定、比例生效、样本不足
   报错；split.yaml 成员缺失于 meta、三组不互斥或未覆盖清单成员时报错。
4. Dataset 按显式成员构造：目录中遗留的旧 npz（不在清单）不被加载。
5. preprocessing：仅 train 拟合（val/test 统计不影响输出）；[P,1,3] 广播
   形状正确；x/y 零方差位置除数均取 1；y 变换-逆变换往返恢复原值；
   logalpha 遇 alpha <= 0 报错。
6. 模型：输入 [N,P,T,3] 展平 D=P*T*3，前向输出 [N,2]；hidden_dims 配置
   生效。
7. CPU smoke：微型合成集跑 1–2 epoch 全流程（zero_grad/backward/step 后
   权重发生有限更新），产物齐全、ckpt 重载一致、evaluate 输出每参数组合
   一行、两次重载推理结果一致。
8. 常规 evaluate：run 内 split 副本与 ckpt.split SHA 不一致报错；npz 与
   ckpt 输入契约错配（T、pulse_ids 顺序、t_s 不一致）报错；空指标子集
   输出 null 而非 NaN/0；预测非有限报错。
9. 预检：目标 npz/meta/split 已存在或部分残留时，拒绝且不写入新产物。
10. 不做：物理 QC、协议逐字段重查、正向回代、科研有效性验证。

## 10. 已实现模块清单与待定研究项

已实现模块清单（工程验证以 ruff/pyright 与聚焦离线 pytest 为准；非
科研验证）：

- 配置/数据：`training_config.py`（严格 YAML 加载）、`training_data.py`
  （npz/协议快照/split/Dataset）、`configs/training/mlp.yaml`
- 预处理/模型：`preprocessing.py`（train-only 拟合与变换）、
  `models/mlp.py`（MLPRegressor）
- 训练/评估：`training.py`（训练循环/early stopping/ckpt 读写）、
  `evaluation.py`（绑定校验/指标）；入口
  `scripts/prepare_training_samples.py`、`scripts/train_mlp.py`、
  `scripts/evaluate_model.py`；测试 `tests/test_training_config.py`、
  `tests/test_training_data.py`、`tests/test_preprocessing.py`、
  `tests/test_mlp.py`、`tests/test_training.py`、
  `tests/test_evaluation.py`

**以上为工程实现与离线验证：未在正式研究数据上训练，未做科研有效性
验证，不存在任何研究结论。**

待定研究项（不阻塞工程实现，需研究决策/数据后定）：
- 正式协议的 pulse 集合与时间网格冻结；
- 标签变换（identity vs logalpha）；
- split 比例与评估目标口径，以及 Ku=0 control 是否参与训练/选模；
  随机 holdout 的指标**不自动**证明插值/外推能力，需另行设计评估；
- 超参搜索（当前仅为初始工程候选）；
- 完整样本入库渠道（是否 LFS/发布，待研究决策，不强制 LFS）；
- MuMax3 正向回代验证（最终结论前置条件，不在本计划范围内）。
