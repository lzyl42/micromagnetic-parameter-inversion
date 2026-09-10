# 训练首版实施日志（2026-09-09）

> 性质：工程实施与工程验证日志。**未在正式研究数据上训练，不存在可靠
> 科研结果或研究结论**；本文不构成科研有效性证明。设计细节见 `train.md`，
> MuMax3 前向数据生成与 QC 见 `progress.md`。

## 1. 日期与实施范围

- 日期：2026-09-09（首版训练/评估工程实施完成）。
- 范围：从模块骨架到首版可运行实现——
  - `training_config.py`：frozen dataclass + 严格 YAML 加载/校验（拒绝占位值）；
  - `training_data.py`：npz 读写（无 object 数组）、dataset_meta/split
    加载校验、按显式成员构造 Dataset、输入契约；
  - `preprocessing.py`：仅 train 组拟合的 [P,1,3] 广播标准化与标签
    变换/逆变换（effective scale 语义）；
  - `models/mlp.py`：MLPRegressor（携带输入契约）；
  - `training.py`：训练循环、early stopping（独立显著改善 reference）、
    ckpt 安全读写；
  - `evaluation.py`：run/ckpt 绑定校验、物理单位指标、test_predictions
    导出；
  - 入口脚本：`scripts/prepare_training_samples.py`、
    `scripts/train_mlp.py`、`scripts/evaluate_model.py`；
  - 配置：`configs/training/mlp.yaml`（dataset/run_name 仍为占位符）。

## 2. 两轮 review 修复（要点）

- 数值失败停止状态与 CLI 非零退出：训练结束记录 `stop_reason`
  （`max_epochs` / `early_stopping` / `numerical_failure`）、`stop_epoch`
  与 `detail`（停止时 metrics）；数值失败时保留上一完整 epoch 的有效
  快照，但 CLI 以非零退出码结束、不打印训练完成——保留权重不代表训练
  成功。
- ckpt 安全落盘/读取：磁盘字典仅含 Python primitives/list/dict + CPU
  Tensor；读取显式 `torch.load(weights_only=True, map_location="cpu")`，
  校验格式版本与关键字段/契约形状后重建 dataclass。
- 评估 provenance：`test_metrics.json` 新增 `provenance`（实际加载的
  checkpoint 路径、checkpoint sha256、split sha256）；输出按 split.test
  成员顺序。
- 防泄漏：预处理统计仅 train 组拟合；split 按参数组合（psid）分组划分，
  同一参数组合的全部激励不跨 train/val/test。
- 空 split 与数值精度：train/val 为空即报错；损失聚合升 float64、
  effective scale（std <= eps → 1）、空指标子集输出 JSON null（非
  NaN/0）。
- mapping 与 I/O 去重：配置、预处理状态的序列化由所属模块共用，train/val
  样本不再完整读取两遍；产物预检拒绝覆盖
  （npz/meta/split、run 目录、ckpt、评估产物各只写一次）。

## 3. 提交前复验记录

- 提交前复验：**107 个聚焦离线 tests 通过**（tmp 合成数据、CPU，不依赖
  GPU/MuMax3；覆盖 npz 往返、split、preprocessing、模型、CPU smoke、
  evaluate 绑定/契约/空子集、预检拒绝覆盖等，范围见 `train.md` 第 9 节）。
- 复验命令：`uv run --no-sync pytest tests/test_training_config.py
  tests/test_training_data.py tests/test_preprocessing.py tests/test_mlp.py
  tests/test_training.py tests/test_evaluation.py`。
- 本轮相关 Python 文件的 Ruff 检查、Ruff 格式检查及定向 Pyright 检查
  均通过；提交前未重新运行 GPU 实验。

## 4. 真实既有数据工程审计摘要（本机 Tesla T10）

数据：仓库既有 4 参数组 `cofeb_protocol_b_four_corners_test_v1`（alpha/Ku
四角：PL(0.004, 2000)、PH(0.004, 30000)、PD(0.02, 2000)、PDH(0.02,
30000)），单 pulse（pulse_A2）、T=401（10 ps 采样）。原始 raw 以只读
符号链接挂入隔离 data_root：未复制、未覆盖、未重跑 MuMax3。

- 环境：PyTorch 2.11.0+cu128、CUDA available、Tesla T10、`device: auto`
  → cuda；被测为脏工作区实现（源码 SHA-256 清单 26 文件 + git HEAD +
  dirty 标记留档于审计目录 `env/`）。
- prepare/split：seed 43、ratios 0.5/0.25/0.25 → train={PD, PH}、
  val={PL}、test={PDH}（三组互斥、并集完备，未为结果调整划分）；npz 与
  源 CSV float32 逐位一致；预处理统计仅用 train 独立复算（y 均值
  alpha 0.012 / Ku 16000）。
- 训练：batch_size=1、MLP [64,32,32]、Adam lr=1e-3；early_stopping @
  epoch 51（patience=50），best_epoch=1（best_val_loss≈0.847）；train
  MSE 1.056→3.6e-5（2 个 train 样本记忆化），val 0.847→3.17（过拟合）。
  best.pt 仅含第一轮 2 次参数更新。
- 独立评估（best.pt，test=PDH，n=1）：alpha 预测 0.012560 vs 真值 0.02
  （MAE=RMSE=0.00744）；Ku 预测 14090.2 vs 真值 30000（MAE=RMSE=15909.8
  J/m³）；main n=1、control n=0（空子集指标 null）；两次独立重载预测
  逐位一致。
- 参照（简述）：train 组均值基线（alpha 0.012 / Ku 16000）在 test 上的
  MAE 约 0.008 / 14000，与上述模型误差同量级——n=1 下不构成模型优于
  基线的证据。

## 5. 精度审计覆盖与局限

- 已覆盖：首个 batch 的独立数值审计（forward / MSE float64 复算 /
  backward / zero_grad / Adam 首步公式 / 中心差分 9 个平滑坐标，均在
  float32 精度内一致）；第一轮两个 batch 的逐位复现核对（首步参数与
  审计快照、第二步参数与 best.pt 逐位一致）。
- 未覆盖：第 2–51 轮共 100 次参数更新**未逐步独立数值审计**（全部 51
  轮共 102 次更新；后续轮次仅有生产
  侧有限性检查、损失轨迹与完成状态证据）；差分抽查未覆盖第二隐层；val
  MSE 未独立复算；batch_size=1、单 pulse，不能替代多样本 batch 与多
  激励验证；未验证非零 weight_decay 行为。
- 未做：未新跑 `generate_dataset`、未重跑 MuMax3、未做正向回代（最终
  结论前置条件，不在本轮范围）。

## 6. 结论边界

- 本轮全部为**工程验证**：执行链、契约绑定与数值实现正确性的冒烟证据；
  **无科研可靠结论**。4 参数组（train=2 / val=1 / test=1）的指标仅有
  描述意义，不支持总体性能或统计显著性判断，不作泛化断言。
- test 组参数组合位于训练参数凸包之外；本次误差不能归结为几何上的
  必然结果，**不是**物理不可识别性的证明，也不证明插值/外推能力。
- 正式研究训练未做；`configs/training/mlp.yaml` 的 dataset/run_name 仍
  为占位符。未来以新数据覆盖既有配置/数据前必须先获用户批准（配置先于
  实验）。

## 7. 产物与可复跑

- 原始审计产物（QC/审计 JSON、日志、run 产物、ckpt、环境与源码清单）
  全部位于被 `.gitignore` 忽略的 `artifacts/train_audit_20260909/`（相对
  路径），仅本地留存，**不随提交公开**；仓库内以本日志摘要为准。本文
  不复制 raw 数据/checkpoint，不记录机器绝对路径或用户名等敏感信息；
  源码哈希以审计目录内清单文件存在为准（不在此粘贴）。
- 重跑要求：保留 `data/samples/`（npz + dataset_meta.yaml）与训练 run
  目录，并以 `MICROMAG_DATA_ROOT` / `MICROMAG_OUTPUT_ROOT` 指向相应根
  目录（run/ckpt 以锚点相对路径 + sha256 绑定外部 meta，见 `train.md`
  第 8 节与 `data/README.md`）。
