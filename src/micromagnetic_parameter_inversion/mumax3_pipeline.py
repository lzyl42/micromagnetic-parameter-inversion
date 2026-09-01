# =============================================================================
# 模块：mumax3_pipeline.py —— 单 case 多脉冲编排（架构审查骨架，纯注释）
# =============================================================================
# 状态声明：
#   - 本文件当前只包含以 # 开头的中文注释与空行；不含 import、常量、类、
#     函数或任何可执行 Python 语句。
#
# 未来职责：
#   1. 定义 RunMode（DRY_RUN | EXECUTE）并编排首版物理流程：加载配置 ->
#      校验 -> case 派生量 -> 逐脉冲 run（staging 准备 -> 渲染 -> 经
#      external.py 执行 -> table 解析 -> 轨迹 CSV）-> 原子发布
#      index.csv + manifest.json（仅 RunMode.EXECUTE）。
#   2. dry-run（RunMode.DRY_RUN）作为本模块的显式模式：仅 plan + render 到
#      artifacts 检查目录，不触碰 data/raw、不写正式 manifest/index
#      （CLI 只委托，不自行复制编排）。
#
# 依赖（未来实现时引入，当前不存在）：
#   - 项目内：mumax3_config（LoadedConfig）、physics、mumax3_script、
#     mumax3_results、external（run_mumax3 —— 唯一执行入口）、paths
#   - 标准库：pathlib、json、csv、hashlib、os、time
#   - 硬约束：MuMax3 只经 external.run_mumax3 调用（subprocess 参数列表，
#     禁止 shell=True）。未来如需 -o/-gpu 等参数，只能扩展 external.py 经
#     白名单/参数列表传递；本模块绝不绕过它直接起进程。
# =============================================================================


# -----------------------------------------------------------------------------
# 输出路径契约（硬约束，基于 external.run_mumax3([exe, script]) 的现实）
# -----------------------------------------------------------------------------
# - 最终目录（仅 RunMode.EXECUTE）：data/raw/<experiment_name>/<case_id>/
#   （经 paths.data_root()；CLI --data-root 为等价的显式覆盖，不写入/不改写
#   任何配置对象字段）；已存在最终目录一律拒绝隐式覆盖（显式清空指令是
#   未来 CLI 的事）。
# - RunMode.DRY_RUN 不创建/占用 data/raw 最终目录，不写正式 initial/final
#   manifest 或 index；仅 plan + render 到检查产物根
#   paths.output_root()/mumax3_dry_runs/<case_id>/<normalized_config_hash>/
#   （CLI --dry-run-root 可覆盖；同名目录已存在则拒绝覆盖）。检查产物属
#   artifacts；EXECUTE 不读取、不依赖、不清理 dry-run 产物——真实运行
#   不被先前 dry-run 阻塞。
# - staging 策略（定案）：每次正式 run 使用唯一且预先不存在的 staging 路径
#   （位于最终目录内，命名实现时固定，如 .staging-<run_id>/）；路径已存在
#   （碰撞）即失败并保留现场，绝不自动"清空/清理/复用"旧 staging；staging
#   开始前断言其中不存在旧 script.out/table.txt（防御式复核；正常流程中
#   因路径预先不存在而天然满足）。
# - 进程工作目录 = 该 run 的 staging 目录；脚本名固定 script.mx3；
#   MuMax3 默认原生目录随之固定为 script.out；table 固定为
#   script.out/table.txt。输出目录不由 .mx3 控制（模板无 {{OUTPUT_DIR}}）。
# - run 成功后产物从 staging 原子发布到最终目录（轨迹 CSV 发布名
#   <run_id>.csv，每 run 唯一）；staging 目录本身（含原始 table、run.log、
#   rendered script）首版一律保留，不自动删除。manifest/index 中只写
#   相对路径（相对数据根）。
# - gpu_index（若提供）未来只经 external.py 白名单参数传递（当前不改）。


# -----------------------------------------------------------------------------
# 对象：RunMode（拟议 enum）
# -----------------------------------------------------------------------------
# 值：DRY_RUN | EXECUTE
# 语义：
#   - DRY_RUN：检查产物模式（属 artifacts）。不创建/占用 data/raw 最终目录；
#     不写正式 initial/final manifest 或 index；仅 plan + render 到
#     paths.output_root()/mumax3_dry_runs/<case_id>/<normalized_config_hash>/
#     （CLI --dry-run-root 可覆盖检查产物根；与 --data-root 互不混用）；
#     同名检查目录已存在则拒绝覆盖（PipelineError），不自动清理。
#     normalized_config_hash = 规范化完整有效配置（含激励列表；不含执行
#     参数、输出路径与 CLI 覆盖）的 SHA256，仅用于 dry-run 检查目录命名，
#     与 case_id/parameter_set_id 身份规则相互独立。
#   - EXECUTE：正式输出模式（属 data/raw）。使用 data/raw 最终目录与正式
#     staging、initial/final manifest、index。不读取、不依赖、不清理
#     dry-run 产物——真实运行不被先前 dry-run 阻塞。


# -----------------------------------------------------------------------------
# 固定文件名契约（与 ARCHITECTURE.md 同步声明）
# -----------------------------------------------------------------------------
# - 实验级（最终目录内，固定名）：config.snapshot.yaml（原 YAML 快照）、
#   manifest.json、index.csv。
# - run 级（每 run staging 内，固定名）：script.mx3、script.out/table.txt、
#   run.log、trajectory.csv。
# - 轨迹 CSV 在 staging 内固定名 trajectory.csv；run 成功后发布到最终目录
#   时以 <run_id>.csv 命名（每 run 唯一，避免同名冲突）。
# - manifest/index 仅记录相对数据根的相对路径，不写绝对路径。


# -----------------------------------------------------------------------------
# 对象：RunRecord（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 职责：自包含的单 run 审计记录——index.csv 一行 + manifest 条目所需的一切。
# 字段（拟议）：
#   - run_id: str
#   - excitation_id: str              # pulse_id
#   - case_id: str                    # 规范化物理 case 身份
#   - parameter_set_id: str           # 规范化 alpha+Ku 身份（数据集 split 键）
#   - alpha: float                    # 反演目标（冗余记录，便于 index 直接消费）
#   - Ku_J_per_m3: float              # 反演目标（冗余记录，命名带单位）
#   - status: str                     # 枚举：success | failed | skipped
#   - error_message: str | None
#   - return_code: int | None         # MuMax3 进程退出码
#   - duration_s: float | None
#   - started_at / finished_at: str | None   # ISO 8601
#   - paths（相对数据根）：script（staging 内固定名 script.mx3）、原生
#     table（script.out/table.txt）、轨迹 CSV（发布名 <run_id>.csv）、
#     stdout/stderr 日志（run.log）——固定文件名契约见上节
#   - hashes：script_sha256、table_sha256、csv_sha256、log_sha256
#   - warnings: tuple[str, ...]       # 仅非关键诊断
# 校验：status=success 时轨迹 CSV 路径与哈希必须存在；failed/skipped 时为空；
#       身份字段（case_id/parameter_set_id/excitation_id/alpha/Ku）恒非空——
#       失败的 run 也必须携带身份，保证 index 可被数据集层分组消费。
# 异常：PipelineError


# -----------------------------------------------------------------------------
# 对象：SimulationRunResult（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 职责：一次实验的完整汇总；足以单独生成 manifest（自包含）。
# 字段（拟议）：
#   - loaded_config: LoadedConfig     # 原文、source_sha256、解析配置、身份
#   - case_derived: CaseDerivedPhysics
#   - records: tuple[RunRecord, ...]  # 全部 run（含失败/跳过）
#   - software_metadata: dict         # Git commit + dirty、uv.lock hash、
#                                     # Python 版本、MuMax3 banner/build 与
#                                     # 安全参数列表、GPU/driver 信息
#   - schema_versions: dict           # manifest/index/csv/parser schema 版本
#   - experiment_status: str          # 枚举：success | partial | failed | error
#                                     # error = 实验级失败（未开始任何 run）
#   - index_csv_path / manifest_json_path: Path | None
#                                     # 相对数据根；RunMode.DRY_RUN 下为 None
#                                     # （不写正式 index/manifest）
#   - started_at / finished_at: str


# -----------------------------------------------------------------------------
# 失败与原子发布（硬约束）
# -----------------------------------------------------------------------------
# - RunMode.EXECUTE：配置一旦有效：立即在最终目录创建 staging 并写入
#   initial manifest（含 run plan：全部预期 run 的身份与路径）。此后无论
#   成功、部分失败还是全部失败，最终 index.csv 与 manifest.json 都会记录
#   所有状态并发布——成功/部分失败/全失败均有最终审计产物。
# - RunMode.DRY_RUN：不创建/占用 data/raw 最终目录，不写正式 initial/final
#   manifest 或 index；仅 plan + render 到检查产物根
#   paths.output_root()/mumax3_dry_runs/<case_id>/<normalized_config_hash>/
#   （或 --dry-run-root 覆盖）；同名检查目录已存在则拒绝覆盖（PipelineError）。
# - 两级失败区分：
#     * 实验级失败（MuMax3 缺失、公共模板非法等）：未开始任何 run，
#       experiment_status=error，manifest 记录原因；同样发布。
#     * 单 run 失败（超时/非零退出码/table 缺失或校验失败）：该 run
#       status=failed，其余继续；experiment_status=partial 或 failed。
# - 原子性：
#     * 每 run 完成后原子写入/更新其持久化 RunRecord、产物 checksum 与状态
#       （同目录临时文件 + os.replace）。
#     * 实验级 index/manifest 发布：同目录临时文件 + os.replace。
#     * 进程崩溃时保留 staging 诊断；staging 路径碰撞即失败并保留现场，
#       绝不自动清空/清理/复用旧 staging；首版不自动恢复、重试或删除
#       （成功/失败/崩溃的 staging 一律保留）；未来 resume 必须先校验
#       配置、模板、脚本与产物哈希一致。
# - 失败政策衔接：mumax3_results 的硬失败（缺核心列/非有限值/时间非严格
#   递增/期望采样点缺失）使 run failed；warnings 仅非关键诊断并入 RunRecord。


# -----------------------------------------------------------------------------
# 拟议函数签名
# -----------------------------------------------------------------------------
# def plan_runs(loaded: LoadedConfig, *, mode: RunMode,
#               data_root: Path | None = None,
#               dry_run_root: Path | None = None) -> list[RunRecord]
#   输入：LoadedConfig、运行模式、可选数据根/检查产物根覆盖
#         （data_root 缺省 = paths.data_root()；dry_run_root 缺省 =
#         paths.output_root()/mumax3_dry_runs）
#   输出：预期 RunRecord 列表（status=skipped 占位，身份/路径已填）
#   职责：纯规划——不创建目录、不执行进程；EXECUTE 校验最终目录不存在
#         （拒绝隐式覆盖）；DRY_RUN 校验检查目录不存在（拒绝覆盖）
#   异常：PipelineError
#
# def run_experiment(loaded: LoadedConfig, *, mode: RunMode,
#                    data_root: Path | None = None,
#                    dry_run_root: Path | None = None) -> SimulationRunResult
#   输入：LoadedConfig；mode 为显式模式开关（RunMode.DRY_RUN | RunMode.EXECUTE）
#   输出：SimulationRunResult
#   职责（mode=RunMode.EXECUTE，完整流程）：
#     1. 防御式 validate_config（LoadedConfig 已校验，此处复核）
#     2. compute_case_derived_physics
#     3. 创建最终目录 + staging，写入 initial manifest（run plan）
#     4. 逐激励：
#        a. compute_run_derived_physics（采样调度、关场边界）
#        b. 取唯一且预先不存在的 staging 路径（碰撞即失败并保留现场）->
#           断言无旧 script.out/table.txt -> mumax3_script 渲染并写
#           script.mx3
#        c. external.run_mumax3(script_path, workdir=staging,
#           timeout=execution.timeout_s)
#        d. mumax3_results.parse_table(script.out/table.txt, run_id,
#           expected_sample_times_s) -> write_trajectory_csv（staging 内
#           固定名 trajectory.csv）
#        e. 原子更新 RunRecord（状态、哈希、相对路径、warnings）
#     5. 发布 index.csv + manifest.json（临时文件 + os.replace）；成功 run
#        的轨迹 CSV 发布为最终目录内 <run_id>.csv
#   职责（mode=RunMode.DRY_RUN，显式模式）：
#     - 不创建/占用 data/raw 最终目录；不写正式 initial/final manifest 或
#       index（index_csv_path/manifest_json_path 为 None）
#     - 仅 plan + render 全部 script.mx3 到检查产物根
#       <dry_run_root>/<case_id>/<normalized_config_hash>/（同名目录已存在
#       则拒绝覆盖）；不调用 external.run_mumax3、不解析 table
#     - 返回 SimulationRunResult（records 标记 skipped，附脚本路径与哈希）
#   异常：实验级失败 -> PipelineError（EXECUTE 下 manifest 仍记录，见失败
#         政策；DRY_RUN 下检查目录冲突同样抛 PipelineError）
#
# def write_index_csv(records: tuple[RunRecord, ...], path: Path) -> Path
#   职责：每 run 一行，列固定含：run_id、excitation_id、case_id、
#         parameter_set_id、alpha、Ku_J_per_m3、status、相对路径、哈希、
#         时间、warnings；UTF-8、确定性别名顺序；临时文件 + os.replace
#
# def write_manifest_json(result: SimulationRunResult, path: Path) -> Path
#   职责：写入完整 provenance（见下节清单）；临时文件 + os.replace
# =============================================================================


# -----------------------------------------------------------------------------
# manifest/provenance 清单（write_manifest_json 必须覆盖；相对路径；
# 仅 RunMode.EXECUTE 发布正式 manifest——DRY_RUN 不写正式 manifest/index）
# -----------------------------------------------------------------------------
# - manifest schema version
# - 原始 YAML SHA256（LoadedConfig.source_sha256）+ 配置快照（原文）
# - 规范化有效配置哈希、case_id、parameter_set_id（含规范化规则版本）
# - Git commit + dirty 标志；uv.lock 哈希
# - 模板哈希；每个 rendered script 的 SHA256
# - MuMax3 banner/build 信息与本次使用的安全参数列表（经 external.py）
# - GPU/driver 信息
# - 平衡态来源及其哈希（EquilibrationConfig provenance）
# - 每 run：原始 table / 轨迹 CSV / 日志的 SHA256 与相对路径
# - CSV schema version 与 parser version；原生表头映射
# - 每 run 状态、warnings、时间戳；experiment_status
# =============================================================================


# -----------------------------------------------------------------------------
# 边界声明（不在本模块实现）
# -----------------------------------------------------------------------------
# 1. 数据集层：以 index.csv 为输入，仅按 parameter_set_id 分组做
#    train/val/test 划分（同参数组所有激励同 split），防泄漏特征化在此层做。
# 2. campaign / 多 alpha-Ku 扫描：后续独立层，以多个单 case 配置为输入。
# 3. 训练、推理、反演：均不存在，也不在本模块计划中。
# 4. 本模块当前不可执行；仓库中不存在任何已实现的训练、推理或模拟流程。
