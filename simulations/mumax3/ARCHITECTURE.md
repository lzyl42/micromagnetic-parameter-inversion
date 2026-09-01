# MuMax3 单 case 多脉冲模拟 —— 架构审查文档

> **状态：架构审查骨架。** 本分支上的十个新文件全部是设计注释，不含任何可执行
> 代码。当前仓库不可执行任何模拟、训练或推理；本文档描述的是"将要构建的
> 东西"，不是"已完成的东西"。

## 1. 目标 / 非目标

**目标（首版）**

- 一份配置 = **一个物理 case**（单组材料 / 椭圆几何 / 一对 alpha+Ku 反演目标）
  × **多个短脉冲激励**。不是多参数 campaign。
- 实现仓库真实物理流程：
  **initialization（初始磁化）→ equilibration（弛豫/最小化）→ pulse（短矩形
  脉冲）→ field off（精确关场）→ free-decay recording（自由衰减记录）**。
- 每个脉冲激励生成一个独立 run；每个 run 产出轨迹 CSV；实验级产出
  index.csv 与 manifest.json，作为后续数据集层与反演层的输入。

**非目标（首版）**

- 多参数 campaign / alpha-Ku 扫描（后续独立层，以多个单 case 配置为输入）。
- 连续周期场、泛化波形（首版只有短矩形脉冲）。
- ellipse 以外的几何（首版仅 ellipse，禁止 box 泛化）。
- 非均匀初态（首版仅均匀初态）。
- 自动恢复/重试/自动清理（首版 staging 碰撞即失败并保留现场；崩溃保留
  staging 诊断；不自动 resume、不自动删除任何 staging）。
- 训练、推理、反演模型（均不存在，也不在本架构计划内）。

## 2. 文件职责

| 文件 | 职责 |
|---|---|
| `configs/experiments/mumax3_simulation.yaml` | 单 case 多脉冲实验配置模板（纯注释；结构校验必然失败、不可用于运行） |
| `src/.../mumax3_config.py` | 配置模型 + 加载（LoadedConfig：原文/哈希/身份）+ 校验 + mT→T 换算 + 身份计算 |
| `src/.../physics.py` | CaseDerivedPhysics（交换长度/mesh 诊断/H_k 诊断）+ RunDerivedPhysics（B_ext 三分量 T、显式采样表、关场边界） |
| `src/.../mumax3_script.py` | Mumax3ScriptContext + 渲染（确定性；固定文件名 script.mx3；不嵌入输出路径） |
| `src/.../mumax3_results.py` | table 解析 + 固定 schema v1 CSV（t_s, m_*, b_ext_*_T）；硬失败政策 |
| `src/.../mumax3_pipeline.py` | RunMode + RunRecord/SimulationRunResult + 编排 + staging + 原子发布 + manifest |
| `scripts/run_mumax3_simulation.py` | CLI 薄封装（委托 pipeline；`--dry-run`/`--data-root`/`--dry-run-root`，无 `--output-root`） |
| `simulations/mumax3/simulation.mx3.in` | .mx3 模板（纯注释五段结构 + 占位符） |
| `tests/test_mumax3_simulation.py` | 未来测试设计（纯注释） |

## 3. 对象关系

```
LoadedConfig (YAML 原文 + SHA256 + 解析配置 + 身份)
  ├─ Mumax3SimulationConfig
  │    ├─ MaterialConfig ────────┐
  │    ├─ EllipseGeometryConfig ─┤
  │    ├─ MeshConfig ────────────┤   case_id = SHA256(规范化 case)
  │    ├─ InitialStateConfig ────┤   （不含激励/执行/输出/run 顺序）
  │    ├─ EquilibrationConfig ───┘
  │    └─ RecordingConfig ──┐
  │                         │
  │    └─ ExcitationConfig（多个短脉冲，不属于 case 身份）
  │                         │
  ├─ parameter_set_id = SHA256(规范化{alpha, Ku})   ← 数据集 split 分组键
  │
  ▼
CaseDerivedPhysics（交换长度 / mesh 诊断 / H_k 诊断）      [case 级，一次]
  ▼
RunDerivedPhysics（B_ext 三分量 T / sample_times_s / pulse_off_index）
  ▼                                        [run 级，每脉冲一次]
Mumax3ScriptContext → script.mx3（staging，固定文件名）
  ▼
external.run_mumax3（workdir=staging, timeout）
  ▼
TrajectoryResult（schema v1 CSV）→ RunRecord（原子更新）
  ▼
index.csv + manifest.json（原子发布，始终发布；仅 RunMode.EXECUTE）
```

RunMode.DRY_RUN 不进入执行与正式发布：仅 plan + render 到 artifacts 检查
目录（见第 5 节）。

## 4. 完整流程

1. **YAML → LoadedConfig**：`mumax3_config.load_config` 读取原 YAML 字节
   （SHA256 留存）→ 解析 → `validate_config` → 计算 `parameter_set_id` 与
   `case_id` → 组装 `LoadedConfig`。
2. **校验**：`validate_config` 执行全部字段级与跨字段校验（含"脉冲时长为
   采样间隔整数倍"的关场边界约束）。
3. **派生物理量**：`physics.compute_case_derived_physics`（交换长度、mesh
   诊断、H_k 诊断、mesh extent）→ 每个激励再算 `compute_run_derived_physics`
   （B_ext 三分量 T、显式采样表 sample_times_s、pulse_off_index）。
4. **渲染**：`mumax3_script.build_context` + `render_script` +
   `write_script` → 每个 run 的 staging 目录写入 `script.mx3`（字节哈希留存）。
5. **external.py 调用**：`external.run_mumax3(script_path, workdir=staging,
   timeout=execution.timeout_s)`。工作目录 = staging，脚本名 script.mx3，
   原生输出 script.out，table = script.out/table.txt。MuMax3 只经此入口
   执行（subprocess 参数列表，禁止 `shell=True`）；未来 `-o`/`-gpu` 等参数
   只能扩展 external.py 的安全参数列表。
6. **table 解析**：`mumax3_results.parse_table(script.out/table.txt, run_id,
   expected_sample_times_s)` → 硬失败政策校验 → `write_trajectory_csv`。
7. **轨迹 CSV / index / manifest**：每 run 产出轨迹 CSV（schema v1）；
   实验级 `write_index_csv`（每 run 一行，含身份字段）与
   `write_manifest_json`（完整 provenance）以临时文件 + `os.replace`
   原子发布——成功、部分失败、全失败均发布。

   以上 3–7 为 RunMode.EXECUTE 流程；RunMode.DRY_RUN 仅执行 plan 与渲染
   （第 4 步），产物写入 artifacts 检查目录，不执行第 5–7 步（见第 5 节）。

## 5. 运行模式与输出隔离（RunMode）

- **RunMode（拟议 enum，定义于 mumax3_pipeline.py）**：`DRY_RUN` | `EXECUTE`。
  pipeline 的显式模式开关；CLI 只委托（`--dry-run` → `DRY_RUN`，缺省 →
  `EXECUTE`），不自行复制编排逻辑。
- **RunMode.DRY_RUN（检查产物，属 artifacts）**：
  - 不创建/占用 `data/raw` 最终目录；不写正式 initial/final manifest 或
    index（`index_csv_path` / `manifest_json_path` 为 `None`）。
  - 仅 plan + render 到 `paths.output_root()/mumax3_dry_runs/<case_id>/
    <normalized_config_hash>/`（CLI `--dry-run-root` 可覆盖检查产物根）。
  - `normalized_config_hash` = 规范化完整有效配置（含激励列表；不含执行
    参数、输出路径与 CLI 覆盖）的 SHA256；仅用于 dry-run 检查目录命名，
    与 `case_id` / `parameter_set_id` 身份规则相互独立。
  - 同名检查目录已存在 → 拒绝覆盖（PipelineError），不自动清理。
- **RunMode.EXECUTE（正式输出，属 data/raw）**：
  - 独占 `data/raw/<experiment_name>/<case_id>/` 与正式 staging、initial/
    final manifest、index（见第 6、8、11 节）。
- **互不阻塞**：dry-run 产物只存在于 artifacts 检查目录；EXECUTE 不读取、
  不依赖、不清理 dry-run 产物——真实运行不被先前 dry-run 阻塞；dry-run
  也绝不触碰 `data/raw`。

## 6. 目录结构（运行时，未来）

```
data/raw/<experiment_name>/<case_id>/        # 仅 RunMode.EXECUTE
├── manifest.json               # 实验 provenance（原子发布，始终发布）
├── index.csv                   # 每 run 一行（原子发布，始终发布）
├── config.snapshot.yaml        # 原 YAML 快照（实验级固定名，始终保留）
├── .staging-<run_id>/          # 每 run 独立 staging（路径唯一且预先不存在；
│   │                           #   碰撞即失败并保留现场；首版不自动清理/删除）
│   ├── script.mx3              # 渲染脚本（run 级固定名）
│   ├── script.out/             # MuMax3 原生输出（run 级固定名）
│   │   └── table.txt           # 原生 table（run 级固定名，始终保留）
│   ├── run.log                 # stdout/stderr 落盘（run 级固定名，始终保留）
│   └── trajectory.csv          # 轨迹 CSV（staging 内固定名；发布为 <run_id>.csv）
├── <run_id>.csv                # 轨迹 CSV（schema v1，发布名，每 run 唯一）
└── ...（每 run 的发布产物）

artifacts/mumax3_dry_runs/<case_id>/<normalized_config_hash>/   # 仅 RunMode.DRY_RUN
└── ...（每 run 渲染的 script.mx3 等检查产物；同名目录已存在则拒绝覆盖）
```

固定文件名契约（与 mumax3_pipeline.py 同步声明）：

- 实验级（最终目录内）：`config.snapshot.yaml`、`manifest.json`、`index.csv`。
- run 级（每 run staging 内）：`script.mx3`、`script.out/table.txt`、
  `run.log`、`trajectory.csv`。
- 轨迹 CSV 发布到最终目录时以 `<run_id>.csv` 命名（每 run 唯一）。
- manifest/index 仅记录相对数据根的相对路径，不写绝对路径。

## 7. 单位政策

- 传给 MuMax3 的一切数值均为 SI 单位。
- 唯一例外：配置层外场幅值以 mT 书写（`b_ext_amplitude_mT`，人类可读），
  经 `mumax3_config.mt_to_t` 唯一换算点转为 `b_ext_amplitude_T` 传给 MuMax3。
- **标量与向量职责区分**：`b_ext_amplitude_T`（config 层）是经单位换算的
  **标量幅值**；`RunDerivedPhysics.b_ext_*_T`（run 层）是该标量与方向分量
  相乘后的 **T 向量**——后者才是模板/求解输入；两者不是重复真值。
- **B/H 严格区分**：`B_ext` 单位 T（模板求解输入）；`H_k` 单位 A/m（仅诊断/
  manifest，不作求解输入）。
- CSV 列名携带单位（`t_s`、`m_x`…`b_ext_*_T`），m 分量单位为 1。

## 8. 失败处理

本节为全文唯一的失败处理权威版本（原重复章节已合并；staging 碰撞与
dry-run 目录冲突一并在此声明）。

| 层级 | 触发 | 行为 |
|---|---|---|
| 配置 | 加载/校验失败 | 退出，不创建任何目录（ConfigLoadError/ConfigValidationError） |
| 实验级 | MuMax3 缺失、模板非法 | experiment_status=error；manifest 记录原因并发布；未开始任何 run |
| 单 run | 超时、非零退出码、table 缺失/解析失败/硬校验失败 | 该 run failed（记录 return_code、日志路径、哈希），继续其余 run |
| 结果校验 | 缺核心列/非有限核心值/时间非严格递增/期望采样点缺失 | run failed（硬失败政策，见 mumax3_results.py） |
| staging | run 的 staging 路径已存在（碰撞） | 该 run 失败并保留现场；不自动清空/清理/复用旧 staging |
| dry-run 目录 | 检查目录已存在 | 拒绝覆盖（PipelineError）；data/raw 不受影响 |
| 发布 | 临时文件 + os.replace | 成功/部分/全失败均发布；崩溃保留 staging 诊断 |

## 9. 数据集层边界（扩展位置）

- **按参数组合划分**：以本流水线产出的 index.csv 为输入，**仅按
  `parameter_set_id` 分组**做 train/val/test 划分——同一参数组的所有激励
  必须同 split，防止跨组泄漏。激励定义不写入参数组合。
- **防泄漏特征化**：标准化/特征统计只在训练组上拟合，属于数据集层职责。
- **campaign / 多 alpha-Ku 扫描**：后续独立层，以多个单 case 配置（或规范
  化 case spec）为输入；每个 case 独立走本流水线，case_id 天然区分。

## 10. 验证边界（test 分支与宏自旋）

- **两类验证必须区分**：
  1. **真实 MuMax3 最小单元 run**：验证"渲染 → 执行 → table 解析"全链路
     （渲染正确性、执行链路、table 解析正确性）。MuMax3 自证其执行链路。
  2. **独立 Python macrospin/LLG 解**：验证物理趋势与单畴极限（如衰减曲线
     符合 LLG 预期）。这是独立参考解，**不能用 MuMax3 自证物理正确性**。
- **test 历史分支**：仅作为开发阶段的独立 Python macrospin/LLG 参考来源；
  不是运行时依赖，也不是稳定数据载体。本文档不绑定具体旧分支名，也不声称
  某分支是唯一基线。
- **验证接受后**：微型真实 table + 配置 + manifest + hash 应迁移到
  `data/samples/` 或 `tests/fixtures/` 并进入主开发历史，作为回归基线。
- **当前状态**：本分支为纯注释骨架，不可执行；上述验证全部待办。

## 11. manifest / provenance 清单

`write_manifest_json` 必须覆盖（全部相对路径；仅 RunMode.EXECUTE 发布正式
manifest——dry-run 不写正式 manifest/index，见第 5 节）：

- manifest schema version
- 原始 YAML SHA256 + 配置快照（原文）
- 规范化有效配置哈希、case_id、parameter_set_id（含规范化规则版本）
- Git commit + dirty 标志；uv.lock 哈希
- 模板哈希；每个 rendered script 的 SHA256
- MuMax3 banner/build 信息与本次使用的安全参数列表（经 external.py）
- GPU/driver 信息
- 平衡态来源及其哈希（EquilibrationConfig provenance）
- 每 run：原始 table / 轨迹 CSV / 日志的 SHA256 与相对路径
- CSV schema version 与 parser version；原生表头映射
- 每 run 状态、warnings、时间戳；experiment_status

## 12. 审查清单

**结构约束（已核对，本次一致性清理确认）：**

- [结构·已核对] 十个文件均为纯注释骨架；.py/YAML 非空行以 `#` 开头，mx3.in 以 `//` 开头
- [结构·已核对] 无 import / 常量 / class / def 实现 / 赋值 / 可执行代码
- [结构·已核对] 无研究参数数值；单位字段只说明必填与单位
- [结构·已核对] 对象归属与命名契约（B_ext=T / H_k=A/m / m 单位 1）在注释层一致
- [结构·已核对] 身份规则（parameter_set_id / case_id）不含激励、执行、输出、run 顺序
- [结构·已核对] 固定文件名契约（实验级 config.snapshot.yaml；run 级 script.mx3、script.out/table.txt、run.log、trajectory.csv）在 pipeline 与本文档同步声明；manifest/index 仅记录相对路径
- [结构·已核对] 模板占位符集合完全枚举（无 `{{EQUIL_*}}` 通配；平衡区段为单一 `{{EQUILIBRATION_BLOCK}}`），placeholder_values 键集合可精确校验
- [结构·已核对] b_ext_amplitude_T（config 层标量幅值）与 RunDerivedPhysics.b_ext_*_T（方向相乘后 T 向量）职责区分明确，无重复真值歧义
- [结构·已核对] CLI 路径覆盖仅 `--data-root`（数据根，等价 MICROMAG_DATA_ROOT 的显式覆盖）与 `--dry-run-root`（dry-run 检查产物根）；无误导性 `--output-root`
- [结构·已核对] RunMode.DRY_RUN / EXECUTE 输出隔离：dry-run 只写 artifacts 检查目录、不触碰 data/raw、不写正式 manifest/index；EXECUTE 独占 data/raw 与正式 staging/manifest/index；互不阻塞
- [结构·已核对] staging 策略：路径唯一且预先不存在；碰撞即失败并保留现场；开始前断言无旧 script.out/table.txt；首版不自动恢复/重试/删除
- [结构·已核对] 失败与原子发布策略在注释层自洽（成功/部分/全失败均发布；失败处理仅第 8 节一个权威版本）
- [结构·已核对] 数据根目录遵从 data/raw 政策（paths.data_root()，非 artifacts）；dry-run 检查产物属 artifacts（paths.output_root()）

**物理与 MuMax3 语义（待验证，实现前必须完成）：**

- [待验证] 派生量公式（交换长度、H_k）以官方文献核实并以测试参考值核对
- [待验证] MuMax3 命令语法：初态/平衡（含恢复真实 alpha）/精确关场/显式采样序列
- [待验证] 真实 table 格式与 schema v1 的映射（宏自旋最小 run 验证）
- [待验证] 显式采样调度与 MuMax3 Run/TableSave 节奏的实际对齐
- [待验证] 宏自旋物理趋势与单畴极限的独立 Python LLG 参考解比对
