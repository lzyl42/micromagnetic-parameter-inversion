# MuMax3 模拟（vertical slice）

MuMax3 是每台机器自行安装的外部依赖，不入库：加入 `PATH` 或设置
`MUMAX3_BIN`；一律经 `src/micromagnetic_parameter_inversion/external.py`
调用（subprocess 参数列表，禁止 `shell=True`）。原始输出进 `data/raw/`
（不入库），模板（`.mx3.in`）与本文档入库。

## 定位

- 一份 `configs/experiments/*.yaml` = 一个 `(alpha, Ku)` parameter set 的
  多脉冲实验；`parameter_set_id` 仅由 alpha 与 Ku 生成，是数据集 split 的
  唯一分组键（同一参数组的所有激励必须同组）。
- 平衡态每个 parameter set 只算一次，全部 pulse 共享同一 `equilibrium.ovf`。
- 关场后首个状态定义为训练时间零点：`sample_index=0`、`t_s=0`，其后取
  整数索引，`t_s = sample_index * sample_interval_s`，固定 `sample_count` 行。

## 固定物理假设（不是配置项）

0 K 无热噪声；单一均匀材料；椭圆薄纳米磁体；能量项只有交换 + 单轴各向
异性 + 退磁 + Zeeman（demag 开启）；开放边界、无 PBC；无 DMI；无 STT/
电流；无缺陷/晶粒/空间涨落；无静态偏置场；关场后 `B_ext=0`；固定初态与
平衡算法。

## 输出布局（固定，不进 YAML）

MuMax3 默认契约：输出目录 = `<脚本名>.out`，位于进程工作目录内。
equilibrium 工作目录渲染脚本固定名 `equilibrium.mx3`，pulse 工作目录渲染
脚本固定名 `simulation.mx3`；`run.log` 由 pipeline 从捕获的 stdout/stderr
写出。

```text
data/raw/<dataset_name>/<parameter_set_id>/
├── config.yaml                              # 配置快照
├── index.csv                                # 每个 pulse 一行
├── manifest.json                            # 全部 pulse 成功后最后写一次
├── equilibrium/                             # equilibrium 工作目录
│   ├── equilibrium.mx3                      # 渲染脚本（固定名）
│   ├── run.log
│   └── equilibrium.out/
│       ├── equilibrium.ovf                  # 共享平衡态
│       ├── log.txt
│       └── references.bib
└── runs/<pulse_id>/                         # pulse 工作目录
    ├── simulation.mx3                       # 渲染脚本（固定名）
    ├── run.log
    ├── trajectory.csv
    └── simulation.out/
        ├── table.txt                        # MuMax3 原生 table
        ├── m_t0.ovf / m_tfinal.ovf          # QC 快照
        ├── log.txt
        └── references.bib
```

每个 pulse 脚本从固定相对路径
`../../equilibrium/equilibrium.out/equilibrium.ovf` 加载共享平衡态。

- `trajectory.csv` 列固定：`sample_index,t_s,m_x,m_y,m_z`。
- `index.csv` 最小字段：`parameter_set_id,pulse_id,alpha,ku_j_per_m3,
  b_ext_x_T,b_ext_y_T,b_ext_z_T,pulse_duration_s,trajectory_path`。
- `manifest.json` 仅在全部 pulse 成功后最后写一次，最少记录：
  parameter_set_id、模型版本、Git commit、uv.lock hash、MuMax banner、
  配置 hash、equilibrium hash，以及配置/脚本/table/CSV/log/QC OVF 的
  相对路径与 hash。
- 任一步失败直接上抛并停止；输出目录已存在抛 `FileExistsError`；失败
  现场保留供人工诊断，不做自动恢复/重试。

## 待真实 MuMax3 单 cell 验证后固定（不进 YAML）

solver 选择（Relax vs Minimize）、MaxErr/MaxDt、gammaLL、EdgeSmooth、
TableAutoSave vs 显式 Run+TableSave：先在单 cell 最小用例上对比验证，
之后固定在模板内；YAML 不提供开关。两个 `.mx3.in` 模板当前为结构骨架，
未经真实 MuMax3 验证，不可直接运行。

## 验证策略

- 独立 macrospin LLG 参考实现 + 真实 MuMax3 链路双重验证。
- 平均磁化轨迹不足以排除边缘模/多畴/翻转：必须对 `equilibrium.ovf`、
  `m_t0.ovf`、`m_tfinal.ovf` 做 OVF QC；必要时在 pilot 验证后于模板固定
  增加中点快照。
