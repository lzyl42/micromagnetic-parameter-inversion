# progress.md —— 数值/网格/协议冻结测试进度与剩余阶段执行手册

> 本文档只覆盖**本轮 benchmark 数值协议冻结测试**：目标是在训练开始前，把
> benchmark 定义之外的 numerics（EdgeSmooth/solver/MaxErr/MaxDt/GammaLL/
> RelaxTorqueThreshold）、z/xy 网格、Relax 行为、pulse 设计、recording 逐项
> 冻结。alpha/Ku 端点本身只验证"稳定可辨识"，不做物理标定。
> 目前首版训练与评估工程已实现；本文档仍仅关于 MuMax3 前向数据生成与
> QC，详见 `train.md` 与 `training_implementation_log_2026-09-09.md`。
> 不包含环境安装要求与代码库架构介绍。

---

## A. 范围与原则

### 目标
- 训练前冻结：numerics 全套、z 网格、xy 网格、Relax 行为、pulse 设计、recording。
- benchmark 定义**直接冻结**（不再测试）：
  - 0 K、无热噪声、单一均匀有效介质、synthetic CoFeB-inspired（不声称复现具体实验 stack）；
  - `Ms = 1.25e6 A/m`、`Aex = 15e-12 J/m`；
  - 真三轴椭球全直径 `100 × 50 × 2 nm`（size_m = [100e-9, 50e-9, 2e-9]）；
  - 各向异性易轴 `+x`、initial `+x`（F4 初态稳健性测试除外）；
  - `GammaLL = 1.7595e11 rad/(T·s)`。
- alpha/Ku 主域候选不变：alpha ∈ [0.004, 0.020]（log 空间）、Ku ∈ [2e3, 3e4] J/m³
  （线性空间）；Ku=0 仅物理 control。

### 数据纪律（硬规则）
1. **每套协议独立 `dataset_name`**；不同协议的数据**不得混用**。
2. 改变任一固定协议字段（material/geometry/recording/numerics/pulses）必须更换 dataset_name。
3. `data/raw/<dataset_name>/<parameter_set_id>/` 内**存在 index.csv 才算完成**。
4. **不覆盖已有目录**：pipeline 对已存在输出目录抛 `FileExistsError`。运行前必须确认
   目标目录不存在。
5. 失败/中断目录**保留现场**：换新 dataset_name 重跑，或人工处置；不得删除。

---

## B. 当前代码能力与使用方法

### 组织方式
`scripts/generate_dataset.py` 顶部两个常量：
- `FIXED_CONFIGS: list[dict]` —— 每项一套**完整固定协议**（dataset_name、material
  的 ms/aex/axis——不含 alpha/Ku、geometry、initial_m、recording、numerics、pulses）；
- `PARAMETERS: list[dict]` —— 每项仅 `{"alpha": …, "ku_j_per_m3": …}`。

运行 `uv run python scripts/generate_dataset.py` 会对 **每个 fixed config × 每个
parameter** 组合：构造 YAML → 写入 → 立即顺序运行模拟。任何一组失败即整体停止
（fail-fast），无 resume/并行/清理。

### 手工操作规则（重要，含踩过的坑）
- **新增一套协议候选**：在 FIXED_CONFIGS 里手工复制一个完整 entry 并改字段与
  dataset_name。不要写合并/继承逻辑。
- **向已有 dataset 追加新参数点（如 PL→追加 PH）**：把 `PARAMETERS` **只留新参数**
  后再运行。若 PARAMETERS 里仍含旧参数，generate_dataset 会先跑旧参数 →
  pipeline 对已存在目录抛 FileExistsError → 整体停止（本轮在 es8 上因此多跑过一套
  Ku=0，保留为额外数据）。
- **正式数据生成时**：FIXED_CONFIGS 只保留冻结后的那一套，PARAMETERS 为全量压力点。

### 命令与输出位置
```bash
uv run python scripts/generate_dataset.py
```
- 生成 YAML：`artifacts/generated_configs/<dataset_name>/NN_alpha…_ku….yaml`（git 忽略）
- 模拟结果：`data/raw/<dataset_name>/<parameter_set_id>/`，其中
  - `config.yaml`（原字节快照）、`index.csv`（每 pulse 一行；**存在 = 完成**）
  - `equilibrium/equilibrium.mx3`、`equilibrium/run.log`（含 DIAGNOSTIC 行）
  - `equilibrium/equilibrium.out/equilibrium.ovf`（共享平衡态，契约必需）
  - `equilibrium/equilibrium.out/geom.ovf`（几何体素化诊断，非契约必需）
  - `runs/<pulse_id>/simulation.mx3`、`run.log`、`trajectory.csv`
  - `runs/<pulse_id>/simulation.out/{table.txt, m_t0.ovf, m_tfinal.ovf}`
- **DIAGNOSTIC 行**（stdout 被 pipeline 原文写入各 run.log）：
  - equilibrium：`//DIAGNOSTIC relax_converged= … MaxTorque_T= … E_total_J= … step= … NEval= …`
    （Relax 后；MaxTorque 单位 T，E_total 单位 J）；
  - simulation：`//DIAGNOSTIC PeakErr= … LastErr= … dt_s= … step= … NEval= …`
    （采样完成后；PeakErr 为该进程历史峰值）。

### 长任务执行与中断策略
- **务必完全脱离终端运行**：`setsid nohup uv run python scripts/generate_dataset.py > /tmp/<log> 2>&1 < /dev/null &`
  （前台跑会被工具/SSH 超时连坐杀死，本轮发生过一次，留下不完整目录）。
- 每 20–30 分钟检查：进程存活、已完成 index.csv 数量、当前 `run.log`/`table.txt`
  字节增长。**无 stdout ≠ 卡死**（mumax 的 log.txt 在退出前保持 0 字节，属正常缓冲）。
- 判停条件（任一）：单个 equilibrium > 10 min；30 min 无任何产物字节增长；
  DIAGNOSTIC `relax_converged=false`；MaxTorque 超门槛。
- 中断后：保留目录 → 换 dataset_name（v2/v3…）重跑失败组；同一 dataset 中
  尚未执行的新参数可用 `run_mumax3_simulation.py --config <已生成的yaml>` 单独补跑。

### 数据回传
- 必须回传：本次全部有效 `data/raw/cofeb_*` 的**完整目录树**（含所有 .ovf、run.log、
  config.yaml、index.csv、trajectory.csv、table.txt）。
- 可选：`artifacts/generated_configs/`（复现用）。
- **两处不完整目录必须标记 excluded**（见 D 节）：`cofeb_qc01_rt_default_es0_xy40_z4_v1`、
  `cofeb_qc01_rt1em4_es0_xy40_z4_v1`。
- 回传前 Linux 检查命令示例（不涉及安装）：
```bash
# 每组 index 存在性与行数
for d in data/raw/cofeb_qc04*; do echo "$d"; wc -l "$d"/*/index.csv; done
# 轨迹点数（应 = sample_count + 1 表头）
awk 'END{print NR}' data/raw/<ds>/<set>/runs/pulse_A2/trajectory.csv
# DIAGNOSTIC 存在性
grep -l DIAGNOSTIC data/raw/<ds>/*/equilibrium/run.log data/raw/<ds>/*/runs/*/run.log
# OVF 非空
find data/raw/<ds> -name "*.ovf" -size -1k   # 应无输出
# 打包
tar czf cofeb_qc04_es12_xy40_z16_a2_v1.tar.gz data/raw/cofeb_qc04_es12_xy40_z16_a2_v1
```

---

## C. 压力点精确表（alpha, ku_j_per_m3）

| 代号 | alpha | Ku (J/m³) |
| --- | --- | --- |
| P0 | 0.004 | 0.0 |
| PL | 0.004 | 2000.0 |
| PH | 0.004 | 30000.0 |
| PD | 0.020 | 2000.0 |
| PDH | 0.020 | 30000.0 |
| PM | 0.00894427190999916 | 16000.0 |
| PA− | 0.006839903786706788 | 16000.0 |
| PA+ | 0.011696070952851464 | 16000.0 |
| PK− | 0.00894427190999916 | 11333.333333333334 |
| PK+ | 0.00894427190999916 | 20666.666666666668 |

（PM = sqrt(0.004×0.020)；PA±/PK± 为主域内 ±1/3 半径的 log/线性偏移点。）

---

## D. 已完成数据集 inventory 与有效性

> parameter_set_id 仅由 (alpha, Ku) 决定：PL = `24737dde54fa8f18`，
> PH = `e309537fa3529566`，P0 = `6609ee7e41347231`。

| dataset | 协议概要 | 状态 |
| --- | --- | --- |
| `cofeb_synthetic_sentinel_v1` | ES0、z4、R10(10ps×1001)、A(y 10mT)+B(z 10mT)；**旧模板（无数值协议控制，等效 RT=-1 默认）** | ✅ 7 点 × 2 pulses 完整有效（P0/PL/PH/PD/PDH/PM/P_D0(.020,0)） |
| `cofeb_qc01_rt_default_es0_xy40_z4_v1` | S1 default（Y30 初态、ZA10、RT=-1） | ❌ **incomplete, excluded**（基础设施中断：equilibrium 完成后 pulse_zero 半途，无 index） |
| `cofeb_qc01_rt1em4_es0_xy40_z4_v1` | S1 RT=1e-4 | ❌ **incomplete, excluded**（equilibrium 59 min 未终止，人工停止） |
| `cofeb_qc01_rt_default_es0_xy40_z4_v2` | 同 S1 default，v2 完整重跑 | ✅ complete（P0）；RT=-1 门通过（见 E） |
| `cofeb_qc02_es0_xy40_z4_v1` | ES0 对照 | ✅ complete（PL） |
| `cofeb_qc02_es4_xy40_z4_v1` | ES4 | ✅ complete（PL） |
| `cofeb_qc02_es8_xy40_z4_v1` | ES8 | ✅ complete（PL + PH）；另含一套**误跑但完整的 P0**（`6609ee7e…`，操作失误产物，保留、不入正式矩阵） |
| `cofeb_qc02b_es8_a2_a10_xy40_z4_v1` | ES8、A2+A10、R4、initial +x | ✅ complete（PL；PH 为 D3 产物） |
| `cofeb_qc02b_es12_a2_a10_xy40_z4_v1` | ES12 | ✅ complete（PL；PH 为步骤 2 后补） |
| `cofeb_qc02b_es16_a2_a10_xy40_z4_v1` | ES16 | ✅ complete（PL；PH 为 D3 产物） |
| `cofeb_qc03_coupled_es12_xy40_z8_a2_v1` | ES12、**z8**、A2-only | ✅ complete（PL + PH）⚠️ MaxTorque 1.48–1.82e-5 T 略超 1e-5 保护线（见 E 注） |
| `cofeb_qc03_coupled_es16_xy40_z8_a2_v1` | ES16、**z8**、A2-only | ✅ complete（PL + PH）⚠️ MaxTorque 1.59–2.00e-5 T 同上 |

sentinel（ES0/z4/R10）实测概要（my 基频，10 ns 窗；τ 为 Hilbert 包络拟合）：
- f：Ku=0/2k/16k/30k ≈ **6.79 / 7.09 / 8.69 / 10.19 GHz**，跨 alpha 不变；
- τ：alpha = .004 / .008944 / .020 ≈ **1.82–1.95 / 0.84–0.85 / 0.36–0.40 ns**；
- 全部轨迹 finite，mean|m| ≈ 0.99996–0.99998。

**本机累计核心实验 ≈ 150 min**（仅模拟墙钟，不含分析/等待；z8 单组约 7.5–8 min）。
该数字**不等于**跨机预计耗时（GPU/驱动/缓存不同，需以首组实测外推）。

---

## E. 当前结论表

| 项 | 结论 | 依据 |
| --- | --- | --- |
| RelaxTorqueThreshold | **RT=-1 选定** | RT=1e-4 在本网格 59 min 不收敛（病态）；RT=-1 约 2 min 收敛。每组必须记录 converged/MaxTorque：**≤1e-5 绿；1e-5~1e-4 黄（必须另过 zero drift<1e-4）；>1e-4 失败** |
| EdgeSmooth | **ES0/ES4 拒绝；ES8 拒绝冻结；ES12 领先（暂定）；ES16 参考** | 精确拟合：PL 上 ES12/16 δf=0.00–0.18%（过），ES8 0.55–0.74%（超）；PH 上 ES12 0.255%（过）、ES8 1.05%（超）；ES0 与 sentinel 频率一致但与 ES8/12/16 整体不一致 |
| z 网格 | **z4 拒绝；z8 暂定；z16 待测** | z4/z8 频差 PL 7.8%、PH 6.6–6.8%（≫0.5%）；m_t0 厚度平均空间 corr≈0.987（数值上满足 ≥0.98，但**不能抵消**频率/范数两项失败）；范数差 ≈7%（>5%） |
| QC pulse | **A2 = 2 mT y 50 ps 冻结为数值 QC pulse** | 2 mT 线性区（A2/A10 主频差 0.000%）；A10 存在 4.738 GHz 二模态（A2/A10 精确频差 1.2–1.9%）→ **A10 作为正式 pulse 待定** |
| B=z 激励 | 幅值/可分性弱 5–6 倍，不严格冗余但价值未定 | PL: my 响应 0.066 vs A 的 0.386 |
| recording | **10 ps × 1001（10 ns）仍为候选**，未冻结 | 高分辨率对照未做（F6） |
| solver/MaxErr/MaxDt | solver5 / 1e-5 / 10 ps 仅为当前基线，**未冻结** | F3 待做 |
| xy 网格 | **未测** | F2 待做 |
| 32 点 | **未跑** | F8，且必须最后 |
| ⚠️ z8 Relax 深度 | z8 下 RT=-1 停止于 MaxTorque 1.48–2.00e-5 T（4/4 略超 1e-5 线） | 组间比较两侧同条件，比值指标预计有效；绝对深度待 Oracle 裁决（收紧 RT 或设 RelaxWallClockTime 均属协议改动） |

---

## F. 剩余阶段（逐阶段执行手册）

> 通用规则（适用以下所有阶段）：每套独立 dataset_name；执行前确认目标 set 目录
> 不存在；每组记录 equilibrium DIAGNOSTIC（converged/MaxTorque）并应用 E 表
> RT 色带；单 equilibrium > 10 min 或 30 min 无产物增长 → 停；追加参数时把
> PARAMETERS 只留新参数；指标定义见 G 节。

### F1 z16（已预设，可直接运行）
- 固定：ES12、xy40、**z16**（cells [40,20,16]）、A2-only、R4、RT=-1、solver5/1e-5/10 ps。
- dataset：`cofeb_qc04_es12_xy40_z16_a2_v1`（脚本已预设）。
- PARAMETERS：先 PL，通过后 PH（顺序执行；若 PL 失败即停）。
- 比较对象（已有）：ES12/z8 = `cofeb_qc03_coupled_es12_xy40_z8_a2_v1`（PL `24737dde…`、
  PH `e309537…`）。
- 门（z16 vs z8，PL 与 PH 各自）：tracked f δf ≤ 0.5%；Gtheta 与前三周期 G_RMS 差 ≤ 3%；
  D_short(3 周期) ≤ 0.08；谱 overlap ≥ 0.98；无峰交换；m_t0 厚度平均空间 corr ≥ 0.98、
  norm 差 ≤ 5%；RE 差 < 10%（诊断性，>10% 阻塞）。
- 判定：z8/z16 通过 → **冻结 z8**；不通过 → **z16 暂定**，停止后续 xy（z8/z16 之外的
  z 需另行讨论）。
- 预计：PL ~10–15 min、PH ~10–15 min（z16 步数成本更高，以首组实测外推）。

### F2 xy（仅 F1 通过并冻结 z8 后）
- 固定：ES12、z8、A2-only、R4。若 F1 判定 z8/z16 未收敛并暂定 z16，必须先停止并重新设计
  z 方向参考，不能直接进入 xy。
- 顺序：先 **xy80**（参考），通过再 **xy64**；PL 先行、通过后 PH。
- dataset：`cofeb_qc05_es12_xy80_z{Z}_a2_v1`、`cofeb_qc05_es12_xy64_z{Z}_a2_v1`
  （当前通过条件下 `{Z}=8`）。
- 已有 xy40 作最粗候选。门与 F1 相同（ tracked f/G/D_short/overlap/峰/空间）。
- 选择：**40 → 64 → 80 取最粗通过者**（xy40 已有数据直接参与比较）。

### F3 solver（仅最终几何确定后）
- 固定：最终几何、A2-only、**R10**（10 ps × 1001）；先跑 PH。
- 三个独立 dataset：
  1. `cofeb_qc06_s5_e1em5_dt10ps_v1`（候选基线：solver5/1e-5/10 ps）
  2. `cofeb_qc06_s5_e1em7_dt2ps_v1`（strict 参考一：solver5/1e-7/2 ps）
  3. `cofeb_qc06_s6_e1em7_dt2ps_v1`（strict 参考二：solver6/1e-7/2 ps）
- 顺序：2 → 3 互比（strict refs：df < 0.1%、稳健衰减比/能量比一致、D_full < 0.002）；
  通过后 1 对 strict 参考：df < 0.2%、D_full < 0.005。
- 通过后：候选 + solver5-strict **追加 PD**（PARAMETERS 只留 PD）。仍不一致才考虑
  MaxDt=1 ps，**不默认收紧**。

### F4 Relax/初态闭合
- 固定：最终几何+最终 numerics、**R10**、pulses = `pulse_zero`（0 mT y 50 ps）+
  `pulse_A2`（2 mT y 50 ps）。
- 初态定义：X = (1,0,0)；Y30 = (0.8660254037844386, 0.5, 0)；Z30 = (0.8660254037844386, 0, 0.5)。
- 5 个独立 dataset（示例命名，可按此模式扩展）：
  1. `cofeb_qc07_initx_p0_final_v1`（X, P0）
  2. `cofeb_qc07_inity30_p0_final_v1`（Y30, P0）
  3. `cofeb_qc07_inity30_ph_final_v1`（Y30, PH）
  4. `cofeb_qc07_initz30_p0_final_v1`（Z30, P0）
  5. `cofeb_qc07_initz30_ph_final_v1`（Z30, PH）
- 判据：converged=true；MaxTorque 色带；pulse_zero 全程漂移 < 1e-4；不同初态的
  E_total（DIAGNOSTIC）相对差 < 1e-6 视为同一能量盆地。**不得声称全局基态唯一**。

### F5 pulse 设计
- 幅值组：同一 config 内三脉冲 `pulse_A2/A5/A10`（2/5/10 mT，y，50 ps），PM 点，
  **R6 = 5 ps × 1201（0..6 ns）**。dataset 示例
  `cofeb_qc09_amp_pm_r6_v1`。
- 时长组：A25/A50（25 ps/50 ps）× PL+PH（两个 dataset，各含 2 pulse），
  示例 `cofeb_qc09_dur_pl_v1`、`cofeb_qc09_dur_ph_v1`。
- A+B 信息点：PM/PA−/PA+/PK−/PK+；A 使用幅值/时长测试选出的 y 向脉冲，
  B 使用同幅值、同持续时间的 z 向脉冲。
- 判据：线性度（幅值/场强比）、频率/增益稳定性、**Fisher/Jacobian 20% 规则**
  （新增 pulse 或信息点对参数 Fisher 信息行列式/最小奇异值的提升 < 20% 则不采纳）；
  第三激励继续暂缓。

### F6 recording
- 两组：`cofeb_qc10_rechires_ph_v1`（**2.5 ps × 4001**，PH，10 ns）；
  `cofeb_qc10_reclong_pl_v1`（**5 ps × 3001**，PL，15 ns）。与既有 10 ps 基线对比。
- 门：df ≤ 0.1%；衰减（τ）0.5–1%；幅值 1%；D ≤ 0.003；末端包络/功率稳定性。
- 注意：**输出采样 dt（TableSave 间隔）与 solver `MaxDt` 是两个不同参数**，互不替换。

### F7 endpoint closure（端点闭合）
- 最终 dataset：`cofeb_prefreeze_a_v1`（仅 F5 选出的 y 向 A 脉冲）或
  `cofeb_prefreeze_ab_v1`（所选 A + z 向 B；仅当 F5 证明 B 有足够信息增益）。
- 7 点全量：四角 PL/PH/PD/PDH + PM + controls P0(.004,0) 与 P_D0(.020,0)。
- **全部新协议重跑**；旧 ES0 sentinel 数据不得混入比较。
- 检查：Ku→f 单调、alpha→衰减单调、无翻转/峰交换；物理分离量至少为数值误差的 10 倍。

### F8 32 点（仅 F1–F7 全部通过后）
- 4 log-alpha `[0.004, 0.006839903786706788, 0.011696070952851464, 0.020]` ×
  8 Ku `[0, 2000, 6666.666666666667, 11333.333333333334, 16000, 20666.666666666668,
  25333.333333333332, 30000]` = 32 组（28 主域 + 4 个 Ku=0 control）。
- **若计划把数据带回本机后再分析，则 F8 必须等 F1–F7 的分析确认之后再运行，
  不能盲跑**（避免 32 组数据因协议问题报废）。
- 正式训练集的 Sobol 采样在 F8 之后另行设计。

---

## G. 分析定义（公式与硬门）

所有频率类比较使用**同一窗口、同一方法**成对计算；禁止用单一 FFT bin 或
不稳定 τ 作为几何/协议硬门（τ 需多窗+多方法一致，否则仅报告不判门）。

- **tracked f**（三法交叉）：
  1. 联合阻尼正弦：my、mz 共享 (f, τ)，各曲线独立 offset/cos/sin 系数，
     `m(t) = c0 + a·e^(−t/τ)cos(2πft) + b·e^(−t/τ)sin(2πft)`（FFT 仅作初值，
     多起点非线性最小二乘）；窗 0–2 / 0–3 / 0–4 / 0.25–4 ns。
  2. zero-crossing 回归：去均值后同向过零时刻 t_k 对序号 k 线性回归，斜率 = f。
  3. Hilbert 相位加权回归：解析相位 unwrap 后以包络²加权线性回归 → f；
     包络对数加权回归 → τ（仅参考）。
  稳定性 = 方法×窗口的极差/均值；比较门槛 = max(0.5%, 3σ_f)。
- **平衡方向**：ê = normalize(mean_{t∈[80%,100%]} m(t))（3D，含 mx）。
- **Gtheta** = acos(ê·m(t0)) / B（rad/T；小角等价 |u(t0)|/B）；u(t) = m − (m·ê)ê。
- **G_RMS** = sqrt(mean_{t∈[0,3T_ref]} |u(t)|²) / B；与参考差 ≤ 3%。
- **D_short** = sqrt( Σ_{t≤3T_ref} ‖mc−mr‖² / Σ_{t≤3T_ref} ‖mr−m_eq,r‖² ) ≤ 0.08
  （mr 为参考轨迹，m_eq,r 其平衡方向；3 周期按参考频率定义）。
- **谱 overlap** = Σ√(Sc·Sr) / sqrt(ΣSc·ΣSr)（my 去均值 + Hann 幅值谱，0–4 ns）
  ≥ 0.98；主峰位置一致（无交换）。
- **RE** = ∫_{5T}^{7T}|u|²dt / ∫_{0}^{3T}|u|²dt（诊断；与参考差 > 10% 阻塞）。
- **几何体积** V = Σg · Vcell（g = geom.ovf 填充率）；Qg = Σg²/Σg（记录用，不作硬门；
  实测 ΣN×Qg ≈ 0.754 ≠ 1，mumax 核归一化机制未闭合，已记录）。
- **Relax gates**：`relax_converged=true` 必须；MaxTorque ≤ 1e-5 绿 /
  1e-5~1e-4 黄（须另过 zero drift < 1e-4）/ > 1e-4 失败。

---

## H. 回传验收清单

对每个回传 dataset：
1. dataset 名在批准清单内；目录含 config.yaml 与 index.csv；
2. index.csv 行数 = pulse 数（表头除外）；
3. 每条 trajectory.csv 行数 = sample_count + 1（表头）；
4. `equilibrium/run.log` 与每个 `runs/*/run.log` 含 `//DIAGNOSTIC` 行；
5. `equilibrium.ovf`、`geom.ovf`、`m_t0.ovf`、`m_tfinal.ovf` 存在且非空；
6. 失败/中断记录（若有）附 run.log 与目录清单，标注 excluded；
7. GPU 墙钟（可选）：每次运行起止时间。
**不得伪造或补写 index.csv**——index 只能由 pipeline 在全部 pulse 成功后原子写出。

---

*最后更新：本文件生成于 z16 预设时点；F1–F8 均未执行或仅部分执行（F1 待运行）。
训练/推理未实现。*
