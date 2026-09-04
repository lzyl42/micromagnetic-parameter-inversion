# MuMax3 模拟（vertical slice）

MuMax3 是每台机器自行安装的外部依赖，不入库：加入 `PATH` 或设置
`MUMAX3_BIN`；一律经 `src/micromagnetic_parameter_inversion/external.py`
调用（subprocess 参数列表，禁止 `shell=True`）。原始输出进 `data/raw/`
（不入库），模板（`.mx3.in`）与本文档入库。

## 定位

- 一份 `configs/experiments/*.yaml` = 一个 `(alpha, Ku)` parameter set 的
  多脉冲实验；同一参数组的所有激励必须同 split 组。
- 输出身份：`dataset_name` 是固定材料/几何/模拟协议的命名约定——协议改变
  必须使用新的 dataset_name；`parameter_set_id` 是仅依赖 alpha+Ku 的稳定
  短 ID（不代表完整配置/环境的 hash provenance），是数据集 split 的唯一
  分组键；目录唯一性 = `(dataset_name, parameter_set_id)`，
  `FileExistsError` 仅在复用同一已有目录时拒绝覆盖。首版不自动验证同一
  dataset 下跨不同 parameter_set_id 的协议一致性。
- 输入边界：`alpha >= 0`；`Ku` 只须有限（可为负或零）；脉冲幅值
  `b_ext_amplitude_mT >= 0`。其余约束不变：所有层级严格 schema（拒绝缺失/
  未知字段、null、bool 冒充数值、非有限值），尺寸/Ms/Aex/duration/
  sample interval 须为正数，cells/sample_count 须为正整数，
  `anisotropy_axis`/`initial_m`/`direction` 须为单位向量（范数容差 1e-6），
  `dataset_name`/`pulse_id` 须为安全单路径段。
- 单位边界：mT 只存在于 YAML（`b_ext_amplitude_mT`），load_config 边界内
  立即乘 1e-3 存为 `PulseConfig.b_ext_amplitude_t`，运行时渲染全用 T。
- 几何语义：`size_m=[dx,dy,dz]` 是椭球三轴全直径（= 包围盒尺寸），
  `cells=[nx,ny,nz]` 各分量为任意正整数：`nz=1` 时单层体素离散自然表现
  为恒厚椭圆截面薄片，`nz>1` 时才逐层解析 z 方向椭球表面；正式研究须
  通过网格收敛测试确定 cells。
- 平衡态每个 parameter set 只算一次，全部 pulse 共享同一 `equilibrium.ovf`。
- 采样静态契约（已经短 test-only pilot 基础执行验证；正式数值协议仍待研究
  配置与收敛/QC 验证）：运行脉冲 -> 关场
  -> 立即 TableSave（`sample_index=0`、`t_s=0`）并保存 `m_t0.ovf` -> 再执行
  `sample_count-1` 次 `Run(sample_interval_s)+TableSave` -> 末点保存
  `m_tfinal.ovf`；其后取整数索引，`t_s = sample_index * sample_interval_s`，
  固定 `sample_count` 行。

## 固定物理假设（不是配置项）

0 K 无热噪声；单一均匀材料；扁椭球薄纳米磁体（公共模型段恒渲染完整三轴
`SetGeom(Ellipsoid(dx, dy, dz))`，不对 nz 做条件分支）；能量项只有交换 +
单轴各向异性 + 退磁 + Zeeman（demag 开启）；开放边界、无 PBC；无 DMI；无
STT/电流；无缺陷/晶粒/空间涨落；无静态偏置场；关场后 `B_ext=0`；固定初
态与平衡算法。

## 输出布局（固定，不进 YAML）

MuMax3 默认契约：输出目录 = `<脚本名>.out`，位于进程工作目录内。
equilibrium 工作目录渲染脚本固定名 `equilibrium.mx3`，pulse 工作目录渲染
脚本固定名 `simulation.mx3`；`run.log` 由 pipeline 从捕获的 stdout/stderr
写出。

```text
data/raw/<dataset_name>/<parameter_set_id>/
├── config.yaml                              # 配置快照（按原字节复制）
├── index.csv                                # 最终参数→数据映射（见下）
├── equilibrium/                             # equilibrium 工作目录
│   ├── equilibrium.mx3                      # 渲染脚本（固定名）
│   ├── run.log
│   └── equilibrium.out/
│       └── equilibrium.ovf                  # 共享平衡态（须存在且非空）
└── runs/<pulse_id>/                         # pulse 工作目录
    ├── simulation.mx3                       # 渲染脚本（固定名）
    ├── run.log
    ├── trajectory.csv
    └── simulation.out/
        ├── table.txt                        # MuMax3 原生 table（固定 schema，见下）
        ├── m_t0.ovf                         # 暂时保留的快照（须存在且非空）
        └── m_tfinal.ovf
```

每个 pulse 脚本从固定相对路径
`../../equilibrium/equilibrium.out/equilibrium.ovf` 加载共享平衡态。
MuMax3 可能额外在 `.out/` 目录产生原生 `log.txt`/`references.bib`；它们
不是 pipeline 必需输出，也不作为合法性条件，故不在此列出。

- `trajectory.csv` 列固定：`sample_index,t_s,m_x,m_y,m_z`。
- `table.txt` 契约：表头（去 `#`、按 tab 切分）必须逐列恰为
  `t (s), mx (), my (), mz ()`；首个原生时刻必须匹配 `pulse_duration_s`，
  此后为固定网格 `pulse_duration_s + i*sample_interval_s`；全部数值有限，
  且平均磁化满足 `|分量| <= 1+1e-6`、`mx^2+my^2+mz^2 <= 1+1e-6`；通过后
  重锚为整数 `sample_index` 与 `t_s = i*sample_interval_s`（不透传原生
  时间）。
- `index.csv` 是唯一的参数→数据映射，最小字段：`parameter_set_id,pulse_id,
  alpha,ku_j_per_m3,b_ext_x_T,b_ext_y_T,b_ext_z_T,pulse_duration_s,
  trajectory_path`；全部 pulse 成功后先写 `index.csv.tmp`，再原子替换为
  最终 `index.csv`。最终 `index.csv` 存在即表示该 parameter set 完整完成；
  失败目录保留供人工诊断，但没有最终 index（临时 `index.csv.tmp` 会被
  清除）。
- 任一步失败直接上抛并停止；输出目录已存在抛 `FileExistsError`；失败
  现场保留供人工诊断，不做自动恢复/重试。

## 验证状态与待验证项（不进 YAML）

已验证（2026-09-02，短 test-only 执行链冒烟 pilot：8×4×1 网格、1 pulse、
3 samples，数值不代表研究参数）：两个 `.mx3.in` 模板可解析执行；Relax 产生
`equilibrium.ovf`；simulation 经固定相对路径 `m.LoadFile` 共享 OVF；脉冲/
关场/手工 TableSave 产生原生表头 `# t (s) mx () my () mz ()` 与恰
`sample_count` 行；parser 时间重锚定；`m_t0.ovf`/`m_tfinal.ovf` 存在性
检查、config.yaml 原字节快照与 index.csv 原子写出；`mumax3 -test` 同轮
通过。

正式实验前仍待验证：最终研究参数（当前 YAML 全 null）、最终几何与网格
收敛、solver/Relax 收敛阈值与鲁棒性、EdgeSmooth 选择、OVF 物理 QC、批量
重复性。上述协议一经验证即固定在模板内；YAML 不提供开关。模板虽可执行，
其数值协议在正式验证完成前不得用于产生正式科研数据。

## 验证策略

- 独立 macrospin LLG 参考实现 + 真实 MuMax3 链路双重验证。
- 平均磁化轨迹不足以排除边缘模/多畴/翻转：必须对 `equilibrium.ovf`、
  `m_t0.ovf`、`m_tfinal.ovf` 做 OVF QC；必要时在 pilot 验证后于模板固定
  增加中点快照。
