"""CNN1D 模型职责清单（**本文件当前不包含任何测试函数**）。

本文件只记录 ``CNN1DRegressor``（``src/micromagnetic_parameter_inversion/
models/cnn1d.py``）未来实现后要满足的模型职责与用例方向；**它不是可执行测试**：
没有 ``test_*`` 函数、没有断言、没有 ``pytest.skip``、也没有 ``pass`` 占位，
因此 **pytest 当前不会从本文件收集到任何用例（收集数为 0）**。配置、checkpoint、
入口与评估的大纲分别放在 ``tests/test_training_config.py``、
``tests/test_training.py``、``tests/test_evaluation.py`` 的注释中，不在此重复。

测试纪律：仅用 ``tmp_path`` / 合成数据，device 恒为 ``cpu``，不触 GPU、MuMax3
或真实数据；不写空 ``pass`` 测试、不写 ``assert True``、不用 ``skip``。

模型职责与用例方向（仅注释）：
"""

# - 接口：``CNN1DRegressor(input_shape=(P, T, 3), *, channels, kernel_sizes,
#   pool_bins, head_hidden_dims)``；``forward``: ``[N, P, T, 3] -> [N, 2]``。
# - 输入重排：``permute(0, 1, 3, 2)`` -> ``contiguous()`` -> ``reshape(N, 3P, T)``；
#   通道顺序固定 pulse-major、component-minor（索引 = p*3 + c）；不同 pulse 的
#   时间维不得首尾相接，``T`` 在卷积阶段原样保留。
# - 卷积：``stride=1`` + 奇数 kernel + ``padding=(k-1)//2`` ⇒ 长度保持 ``T``；
#   ``AdaptiveAvgPool1d(pool_bins)`` 聚合时间；``Flatten`` 后接 ``Linear + ReLU``
#   回归头（``head_hidden_dims`` 允许为空 = 直接线性输出），末层输出 2。
# - 形状：[N, P, T, 3] → [N, 2]，覆盖多 N、多 P 与 N=1。
# - 错误输入拒绝：ndim 错误；``T`` 与 ``input_shape`` 不符；``pool_bins > T``
#   （``<= T`` 由模型层校验，配置层不校验 ``> T``）；非连续 tensor 必须被接受
#   （内部显式 ``contiguous()``）。
# - 通道编码真实检查：用可区分的 pulse/分量/时间构造输入，核对首层真实收到
#   p/c/t 编码（pulse-major、component-minor），而不是只测输出形状。
# - 参数校验（``ValueError``）：``channels`` 非空、各维 > 0、非 bool；
#   ``kernel_sizes`` 非空、各维正奇数、非 bool、与 ``channels`` 等长；
#   ``head_hidden_dims`` 各维 > 0、允许为空、非 bool。
# - 反向梯度：标量 ``loss.backward()`` 后所有需梯度参数 grad 有限（非 NaN/Inf）。
# - ``state_dict`` 往返：save→load 后同输入输出逐位一致；同 seed 两次构造结构一致。
# - 池化：``pool_bins > 1`` 保留多段局部时间信息；``pool_bins == 1`` 仅作对照变体。
# - 模块边界：不读取数据、不做任何标准化/归一化。
# - 实验时记录：感受野 / 池化配置 / 参数量（不设固定物理下界）。
