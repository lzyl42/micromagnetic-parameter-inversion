# =============================================================================
# 模块：mumax3_config.py —— 配置模型、加载与身份（架构审查骨架，纯注释）
# =============================================================================
# 状态声明：
#   - 本文件当前只包含以 # 开头的中文注释与空行；不含 import、常量、类、
#     函数或任何可执行 Python 语句。
#   - 所有对象与函数均为未来实现的设计说明，供架构审查使用。
#
# 未来职责：
#   1. 定义单 case 多脉冲实验配置的数据模型（拟议 frozen dataclass）。
#   2. 从 YAML 加载并解析配置，产出 LoadedConfig（含原文与哈希，供审计）。
#   3. 配置级校验：必填项、正负号、枚举、单位一致性、跨字段一致性。
#   4. 唯一单位换算点：b_ext_amplitude_mT -> b_ext_amplitude_T（MuMax3 B_ext）。
#   5. 稳定身份：parameter_set_id（规范化 alpha+Ku）与 case_id（规范化完整
#      物理 case）；两者绝不包含激励、执行、输出或 run 顺序。
#
# 依赖（未来实现时引入，当前不存在）：
#   - 标准库：pathlib、dataclasses、typing、hashlib、json
#   - 第三方：YAML 解析库（实现时与项目依赖策略对齐，经 uv 管理）
#   - 项目内：micromagnetic_parameter_inversion.paths（data_root 解析）
#   - 注意：本模块不做任何进程执行；MuMax3 调用一律经 external.py。
# =============================================================================


# -----------------------------------------------------------------------------
# 身份规则（硬约束，先于对象定义）
# -----------------------------------------------------------------------------
# - 规范化（拟议）：将参与身份的字段序列化为"键排序、固定分隔符、数值用
#   确定性格式"的规范 JSON，再取 SHA-256。规范化规则实现时固定并写入
#   manifest（schema version），保证跨版本稳定。
# - parameter_set_id = SHA256(规范化{alpha, Ku})
#     * 只含两个反演目标；绝不包含激励、执行、输出、run 顺序。
#     * 数据集层 split 的唯一分组键：同参数组所有激励同 split。
# - case_id = SHA256(规范化{material 全字段, ellipse 几何, mesh, 初态, 平衡策略})
#     * 描述"物理 case 本身"；绝不包含激励、执行、输出、run 顺序。
#     * 同一 case 下换激励，case_id 不变。
# - 激励定义不写入参数组合；campaign/多 alpha-Ku 扫描是后续独立层，
#   以多个单 case 配置为输入。


# -----------------------------------------------------------------------------
# 对象：MaterialConfig（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 字段（拟议，全部 SI 单位）：
#   - Ms: float                      # 饱和磁化强度，A/m，必填，> 0
#   - Aex: float                     # 交换刚度常数，J/m，必填，> 0
#   - alpha: float                   # Gilbert 阻尼系数，无量纲，必填，> 0（反演目标之一）
#   - Ku: float                      # 单轴磁各向异性常数，J/m^3，必填，>= 0（反演目标之一）
#   - anisotropy_axis: tuple[float, float, float]  # 易轴单位向量，无量纲，必填
# 校验：数值有限；Ms/Aex/alpha > 0；Ku >= 0；轴向量非零且模长为 1（容差内）
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：EllipseGeometryConfig（拟议 frozen dataclass；首版仅支持 ellipse）
# -----------------------------------------------------------------------------
# 字段（拟议，长度单位 m）：
#   - shape: str                     # 固定枚举 "ellipse"；其他值校验失败（禁止 box 泛化）
#   - semi_major_m: float            # 长半轴，m，> 0
#   - semi_minor_m: float            # 短半轴，m，> 0 且 <= semi_major_m
#   - thickness_m: float             # 厚度，m，> 0
# 校验：上述数值约束；有限性
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：MeshConfig（拟议 frozen dataclass；与几何分离）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - cells: tuple[int, int, int]    # 网格单元数，正整数三元组
#   - cell_size: tuple[float, float, float]  # 单元尺寸，m，各分量 > 0
# 说明：
#   - mesh extent 不再是独立真值：由 EllipseGeometryConfig + thickness 派生
#     （physics.py 计算），消除 size/grid/cell 三真值矛盾。
#   - 无 pbc 字段：当前孤立椭圆默认无周期性边界。
#   - cells 上限是否受 MuMax3 限制：实现前以官方文档核实。
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：InitialStateConfig（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - direction: tuple[float, float, float]  # 初始磁化方向单位向量，无量纲，模长 1
#   - uniform: bool                  # 首版仅支持均匀初态（true；false 校验失败）
# 校验：方向向量归一化；uniform 为 true
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：EquilibrationConfig（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - strategy: str                  # 枚举（弛豫/能量最小化；拟议集合实现前核定）
#   - duration_s: float | None       # strategy=relax 时必填，s，> 0
#   - convergence: dict | None       # strategy=minimize 时必填；具体求解器与
#                                    # 收敛容差实现前依据官方文档/验证确定，不虚构
# 硬规则（校验强制）：
#   - 平衡阶段允许辅助（较大）阻尼加速收敛，但平衡结束后必须恢复实验真实
#     alpha 再进入脉冲与记录；该恢复是渲染层的固定职责。
#   - 平衡态（及其哈希）作为 provenance 保留（见 pipeline manifest 注释）。
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：RecordingConfig（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - total_duration_s: float        # 关场后记录总时长，s，> 0
#   - sample_interval_s: float       # 标称采样间隔，s，> 0
# 跨字段校验（与激励联动，在 validate_config 执行）：
#   - 每个激励的 pulse_duration_s 必须为 sample_interval_s 的整数倍
#     （脉冲关闭边界精确落在采样网格；不依赖 TableAutoSave 隐式等距）
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：ExcitationConfig（拟议 frozen dataclass；短矩形脉冲，不属于 case 身份）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - pulse_id: str                  # 脉冲唯一标识，实验内不得重复
#   - b_ext_amplitude_mT: float      # 脉冲幅值，人类配置单位 mT，> 0
#   - b_ext_amplitude_T: float       # 派生字段：= mt_to_t(b_ext_amplitude_mT)；
#                                    # 这是 MuMax3 B_ext（T），不是 H_ext（A/m）
#   - field_direction: tuple[float, float, float]  # 脉冲方向单位向量，无量纲
#   - pulse_duration_s: float        # 脉冲时长，s，> 0，且为 sample_interval_s 整数倍
# 语义（首版固定）：
#   - t=0 为记录时间起点；脉冲从 t=0 施加，pulse_duration_s 时刻精确关场，
#     之后自由衰减记录。无 frequency_hz、无泛化 waveform 字段。
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：ExecutionConfig（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - timeout_s: float               # 单 run 进程超时，s，> 0；经 external.py 传入
#   - gpu_index: int | None          # 可选；未来只经 external.py 白名单参数传递
#                                    # （当前 external.py 不改；None = 默认）。
#                                    # 不声称 CPU MuMax3：MuMax3 是 GPU 求解器。
# 说明：
#   - 无 workdir：工作目录由 pipeline 固定为该 run 的 staging 目录。
#   - 无 device=auto/cuda/cpu：那是 PyTorch 语义，不适用于 MuMax3。
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：OutputConfig（拟议 frozen dataclass）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - cleanup_ovf_snapshots: bool    # 唯一清理开关：是否清理大型 OVF 快照
# 政策（硬约束）：
#   - 数据根目录遵从 data/raw 政策：经 paths.data_root() 解析
#     （MICROMAG_DATA_ROOT 或 <project>/data），绝不放 artifacts/。
#   - 最终目录：data/raw/<experiment_name>/<case_id>/；已存在则拒绝隐式覆盖。
#   - manifest/index 一律写相对路径（相对数据根）；pipeline 内部路径对象可
#     绝对解析，但不落盘为绝对路径。
#   - rendered script、原始 table、stdout/stderr 日志、配置快照、manifest、
#     index 始终保留；无笼统 keep_intermediate / copy_rendered_mx3 开关。
# 异常：ConfigValidationError


# -----------------------------------------------------------------------------
# 对象：Mumax3SimulationConfig（拟议 frozen dataclass，聚合根）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - experiment_name: str           # 非空；输出目录一级
#   - description: str | None
#   - seed: int                      # 与 configs/base.yaml 对齐
#   - material: MaterialConfig
#   - geometry: EllipseGeometryConfig
#   - mesh: MeshConfig
#   - initial_state: InitialStateConfig
#   - equilibration: EquilibrationConfig
#   - recording: RecordingConfig
#   - excitations: tuple[ExcitationConfig, ...]   # 长度 >= 1
#   - execution: ExecutionConfig
#   - output: OutputConfig
# 跨字段校验（validate_config）：
#   - excitations 非空且 pulse_id 互不重复
#   - 每个激励 pulse_duration_s 为 recording.sample_interval_s 整数倍
#   - shape == "ellipse"；initial_state.uniform 为 true
# 异常：ConfigValidationError（聚合全部违规项）


# -----------------------------------------------------------------------------
# 对象：LoadedConfig（拟议 frozen dataclass；加载产物，供 pipeline 审计）
# -----------------------------------------------------------------------------
# 字段（拟议）：
#   - source_path: Path              # 原 YAML 路径
#   - source_text: str               # 原 YAML 文本（原样保存为配置快照）
#   - source_sha256: str             # 原 YAML 字节 SHA256（manifest 引用）
#   - config: Mumax3SimulationConfig # 解析并校验后的配置
#   - parameter_set_id: str          # 规范化 alpha+Ku 身份
#   - case_id: str                   # 规范化完整物理 case 身份
# 说明：pipeline 据此保存原始配置与全部哈希，无需二次读文件。
# 异常：ConfigLoadError / ConfigValidationError


# -----------------------------------------------------------------------------
# 拟议函数签名
# -----------------------------------------------------------------------------
# def load_config(path: Path) -> LoadedConfig
#   职责：读字节 -> SHA256 -> YAML 解析 -> 分节解析 -> validate_config
#         -> 计算 parameter_set_id / case_id -> 组装 LoadedConfig
#   异常：ConfigLoadError（缺失/不可读/语法错误）；ConfigValidationError
#
# def _parse_material(raw: dict) -> MaterialConfig
# def _parse_geometry(raw: dict) -> EllipseGeometryConfig
# def _parse_mesh(raw: dict) -> MeshConfig
# def _parse_initial_state(raw: dict) -> InitialStateConfig
# def _parse_equilibration(raw: dict) -> EquilibrationConfig
# def _parse_recording(raw: dict) -> RecordingConfig
# def _parse_excitation(raw: dict) -> ExcitationConfig
# def _parse_execution(raw: dict) -> ExecutionConfig
# def _parse_output(raw: dict) -> OutputConfig
#   职责：字段提取、类型转换、必填检查；缺失必填项立即抛 ConfigValidationError
#
# def validate_config(config: Mumax3SimulationConfig) -> None
#   职责：全部字段级与跨字段校验（含脉冲/采样整数倍关系）；失败即抛出
#   异常：ConfigValidationError
#
# def mt_to_t(value_mt: float) -> float
#   职责：全项目唯一 mT -> T 换算点（= value_mt * 1e-3）；只接受有限且 > 0
#   异常：ConfigValidationError
#
# def normalized_parameter_dict(config) -> dict
# def normalized_case_dict(config) -> dict
#   职责：产出参与身份的规范化字典（键排序、确定性格式；排除激励/执行/输出）
#
# def compute_parameter_set_id(config) -> str
# def compute_case_id(config) -> str
#   职责：规范化字典 -> SHA256；实现时固定规范化规则并记录 schema version
# =============================================================================


# -----------------------------------------------------------------------------
# 异常类型（拟议）
# -----------------------------------------------------------------------------
# - ConfigLoadError：文件不存在、不可读、YAML 语法错误
# - ConfigValidationError：必填缺失、数值越界、枚举非法、跨字段不一致
# - 两者拟议继承同一自定义基类，便于上层统一捕获


# -----------------------------------------------------------------------------
# 注意事项
# -----------------------------------------------------------------------------
# 1. "配置先于实验"：任何研究数值必须先落入 YAML 并通过校验才可进入流程。
# 2. 本注释 YAML 文件本身可被 YAML 解析器解析（结果为 null）——正确表述是
#    "结构校验必然失败、不可用于运行"，而非"不可解析"。
# 3. 本模块当前不可执行；仓库中不存在任何已实现的训练、推理或模拟流程。
