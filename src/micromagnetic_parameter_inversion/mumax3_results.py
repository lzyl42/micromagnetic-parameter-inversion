# =============================================================================
# 模块：mumax3_results.py —— table 解析与轨迹 CSV（架构审查骨架，纯注释）
# =============================================================================
# 状态声明：
#   - 本文件当前只包含以 # 开头的中文注释与空行；不含 import、常量、类、
#     函数或任何可执行 Python 语句。
#
# 未来职责：
#   1. 解析单个 run 的 MuMax3 table 文本文件。
#   2. 按固定版本化 schema 校验并导出轨迹 CSV。
#
# 依赖边界（硬约束）：
#   - 本模块不依赖 mumax3_pipeline（避免循环依赖）：不感知任何编排对象
#     （RunRecord/SimulationRunResult 等），只做解析与导出。
#   - 只接受明确输入：table 文件 Path、run_id、列契约/期望采样信息。
#
# 调用顺序（未来）：
#   external.run_mumax3 完成（pipeline 编排）
#   -> 本模块 parse_table -> validate_trajectory -> write_trajectory_csv
#   -> pipeline 原子更新 RunRecord / 发布 index+manifest
# =============================================================================


# -----------------------------------------------------------------------------
# 固定版本化 CSV schema（v1，硬契约）
# -----------------------------------------------------------------------------
# 列（拟议，顺序固定，列名携带单位）：
#   t_s, m_x, m_y, m_z, b_ext_x_T, b_ext_y_T, b_ext_z_T
# 单位：
#   - t_s：秒（s）
#   - m_x/m_y/m_z：磁化分量，单位 1（无量纲，归一化分量）
#   - b_ext_*_T：外场分量，特斯拉（T）——MuMax3 的 B_ext，不是 H_ext（A/m）
# 版本化：
#   - schema_version 与 parser_version 随每个产物记录（CSV 头注释区/
#     manifest），格式演进必须升版本，不静默改列。
# 原始映射保留：
#   - 原始 MuMax3 表头（列名 + 原始单位）与到本 schema 的映射关系保存在
#     RunRecord/manifest 中（审计：可从 CSV 追溯到原生 table 列）。
#   - MuMax3 table 的确切分隔符/列名/单位实现前以真实输出核实，不预先虚构。


# -----------------------------------------------------------------------------
# 对象：TrajectoryResult（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - run_id: str
#   - source_table_path: Path          # 被解析的原生 table 路径（审计用）
#   - columns: tuple[str, ...]         # 固定 schema 列名（含单位后缀）
#   - row_count: int
#   - time_s: tuple[float, ...]        # t 列，s；严格递增
#   - data: dict[str, tuple[float, ...]]  # 列名 -> 数值序列；键与 columns 一致
#   - original_header: tuple[str, ...] # 原生 MuMax3 表头（映射审计用）
#   - warnings: tuple[str, ...]        # 仅非关键诊断（见失败政策）
# 校验（构造时）：columns 与 schema v1 完全一致；各列长度 == row_count；
#                 time_s 严格递增；核心数值有限
# 异常：TableParseError / TrajectoryValidationError


# -----------------------------------------------------------------------------
# 失败政策（硬约束；与旧设计的本质区别）
# -----------------------------------------------------------------------------
# 以下任一情况必须使该 run 失败（抛出异常 -> pipeline 标记 run failed），
# 不得只记 warning、不得静默删行、不得静默插值：
#   1. 缺少核心列（schema v1 任一列无法从原生表映射得到）
#   2. 核心数值出现 NaN/Inf
#   3. 时间非严格递增
#   4. 期望采样点缺失（对照调用方传入的期望采样信息：
#      sample_times_s / expected_sample_count，容差实现时确定并记录）
# warnings 仅用于非关键诊断（如原生表包含 schema 外的额外列、尾部冗余行），
# 且必须进入 RunRecord.warnings 与 manifest。


# -----------------------------------------------------------------------------
# 拟议函数签名
# -----------------------------------------------------------------------------
# def parse_table(table_path: Path,
#                 run_id: str,
#                 expected_sample_times_s: tuple[float, ...] | None = None
#                 ) -> TrajectoryResult
#   输入：table 文件路径、run 标识、期望采样时刻（来自 RunDerivedPhysics；
#         None 表示跳过采样点数校验——仅限测试场景）
#   输出：TrajectoryResult
#   职责：读取 table 文本 -> 识别原生表头 -> 映射到 schema v1 -> 解析数值
#         -> 构造 TrajectoryResult
#   校验：文件存在且非空；表头可识别；核心列齐全；数值可解析且有限；
#         时间严格递增；期望采样点齐全
#   异常：TableParseError（缺失/空/格式不可解析/核心列缺失）
#         TrajectoryValidationError（非有限值/时间非严格递增/采样点缺失）
#
# def validate_trajectory(result: TrajectoryResult,
#                         expected_sample_times_s: tuple[float, ...] | None = None
#                         ) -> list[str]
#   输入：TrajectoryResult、期望采样时刻
#   输出：问题消息列表（空 = 通过）
#   职责：独立复核上述失败政策各项（防御式；parse_table 已内联校验）
#   说明：返回非空列表 = 校验失败（调用方必须使 run 失败），不是 warning
#
# def write_trajectory_csv(result: TrajectoryResult, path: Path) -> Path
#   输入：TrajectoryResult、目标 CSV 路径
#   输出：写入的 CSV 路径
#   职责：UTF-8 写出；表头 = schema v1 列名；数值确定性格式；
#         头部注释区记录 schema_version / parser_version / 原生表头映射
#   校验：写出行数 == row_count；父目录存在
#   异常：OSError 透传
# =============================================================================


# -----------------------------------------------------------------------------
# 注意事项
# -----------------------------------------------------------------------------
# 1. B/H 命名纪律：CSV 中外场列一律 *_T（B_ext）；不出现 H_ext 列。
# 2. 原生 table 与日志始终保留（保留政策见 mumax3_config.OutputConfig 与
#    ARCHITECTURE.md；staging 内固定名 script.out/table.txt 与 run.log）；
#    本模块只读不删。
# 3. 真实 MuMax3 table 格式的固化在宏自旋最小 run 验证后进行（见
#    ARCHITECTURE.md 验证边界）；在此之前不虚构列名。
# 4. 本模块当前不可执行；仓库中不存在任何已实现的训练、推理或模拟流程。
