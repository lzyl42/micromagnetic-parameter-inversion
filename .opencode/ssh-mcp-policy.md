# SSH MCP 双通道使用策略（项目级指令，所有 agent 均须遵守）

本项目通过两个本地 MCP 服务器访问远程主机：`ssh-mcp-read` 与 `ssh-mcp-ask`。
只读命令策略 [ssh-mcp-command-policy.json](ssh-mcp-command-policy.json) 仅驱动
`ssh-mcp-read` 的白名单；`ssh-mcp-ask` 不注入任何命令白/黑名单，仅由 OpenCode
权限 `ssh-mcp-ask_*: ask` 保护（所有工具调用前需用户批准）。

## 工具描述不暴露策略

MCP 工具描述只给出通用的 `execute-command` 参数 schema，**不会**透露 read 白名单内容。
判断某条命令能否走只读通道时，不得依赖工具描述，必须以策略文件为准。

## 通道选择规则

- 不确定某条命令是否被允许时，先阅读 [.opencode/ssh-mcp-command-policy.json](ssh-mcp-command-policy.json)。
- **只读操作必须走 `ssh-mcp-read`**：仅当**完整命令**与策略中某条正则整体匹配
  （不允许 shell 组合/重定向/变量/换行/引号）时，才使用 `ssh-mcp-read_execute-command`。
  read 支持**有界安全流水线**（最多 4 段，且每一段都命中策略），如
  `cat /etc/os-release | head -n 2`。
- **所有非只读操作与文件传输一律走 `ssh-mcp-ask_*`**，需要用户批准；ask 通道
  不限制任何命令（写入、变更、任意命令均可达，批准后才执行）。
- **不要**用 ask 通道执行只读操作：ask 不再黑名单拦截只读命令，但会触发
  **一次不必要的批准弹窗**——能走 read 的一律走 read。
- 路由失败（read 拒绝）时：如实报告失败原因，选择正确通道重试，**不得**绕过策略
  （如拼接、转义、改在本地 shell 执行等变通手段）。
- `ssh-mcp-read_list-servers` 自动放行；read 通道的 `upload`/`download` 一律拒绝
  （请走 ask 通道，需用户批准）。

## 只读通道边界

read 通道仅允许只读巡检命令（身份/系统信息/进程/网络/存储/日志/包列表等），
命令必须整体匹配策略正则（含 `[ \t]+\|[ \t]+` 分隔的安全流水线）；
禁止重定向、`;`/`&&`/`||` 链、命令替换、反引号、`tee`/`xargs`、解释器
（python/bash -c 等）与 mutating 选项（如 `touch`/`rm`/`tail -f`/`find -exec`
等）——含 shell 元字符或无法整体匹配的输入一律拒绝。
