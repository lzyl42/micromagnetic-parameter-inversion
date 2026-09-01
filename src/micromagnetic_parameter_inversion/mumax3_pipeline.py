"""MuMax3 模拟流水线（vertical slice）。

最小调用流（run_parameter_set）：
1. load_config(config_path)          # 拒绝仍为 null 的必填研究值
2. parameter_set_id(alpha, ku)       # 参数组身份
3. 固定输出目录 data/raw/<dataset_name>/<parameter_set_id>/
   （已存在 -> FileExistsError，不隐式覆盖）
4. 渲染/执行 equilibrium 模板一次，保存共享 equilibrium.ovf
5. 对每个 pulse：从同一平衡态渲染/执行 simulation 模板
   -> parse_table -> write_trajectory_csv
6. 汇总写 index.csv（每个 pulse 一行）
7. 全部 pulse 成功后，最后写单个 manifest.json

任一步失败直接上抛并停止；失败留下的文件供人工诊断，不做自动恢复。
MuMax3 一律经现有 external.run_mumax3 调用（subprocess 参数列表，
禁止 shell=True）。
"""

from __future__ import annotations

from pathlib import Path


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256（index/manifest 记录哈希用）。"""
    # TODO: 分块读取并以 hexdigest 返回十六进制字符串。
    raise NotImplementedError


def run_parameter_set(config_path: Path) -> Path:
    """执行一个 parameter set 的全部模拟，返回其输出目录。"""
    # TODO: 按模块 docstring 的 1-7 步实现：
    #  - 固定布局（MuMax3 默认契约：输出目录 = <脚本名>.out，位于进程工作
    #    目录内；run.log 由 pipeline 从捕获的 stdout/stderr 写出）：
    #    <dataset>/<parameter_set_id>/{config.yaml, index.csv, manifest.json,
    #      equilibrium/（工作目录：equilibrium.mx3、run.log、
    #        equilibrium.out/：equilibrium.ovf、log.txt、references.bib）,
    #      runs/<pulse_id>/（工作目录：simulation.mx3、run.log、
    #        trajectory.csv、
    #        simulation.out/：table.txt、m_t0.ovf、m_tfinal.ovf、log.txt、
    #        references.bib）}
    #  - equilibrium 渲染/执行仅一次，多 pulse 共用同一
    #    equilibrium/equilibrium.out/equilibrium.ovf；
    #  - 每个 pulse 从同一平衡态渲染/执行；解析 table、写 trajectory.csv；
    #  - index.csv 最小字段：parameter_set_id,pulse_id,alpha,ku_j_per_m3,
    #    b_ext_x_T,b_ext_y_T,b_ext_z_T,pulse_duration_s,trajectory_path；
    #  - manifest.json 仅在全部 pulse 成功后最后写一次：parameter_set_id、
    #    模型版本、Git commit、uv.lock hash、MuMax banner、配置 hash、
    #    equilibrium hash，以及配置/脚本/table/CSV/log/QC OVF 的相对路径
    #    与 hash（哈希经 sha256_file）；
    #  - 执行一律 external.run_mumax3（参数列表，禁止 shell=True）；
    #    输出目录已存在 -> FileExistsError；任一步失败直接上抛并停止。
    raise NotImplementedError
