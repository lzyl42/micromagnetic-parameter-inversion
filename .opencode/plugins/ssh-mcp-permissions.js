// 项目级 SSH MCP 权限覆盖插件（自动发现于 .opencode/plugins/）。
// 在 OMOS 之后运行：对每个已解析的 config.agent 条目（外加缺失的
// build/plan/general/explore）保留既有权限，并按固定顺序删除/重放以下键，
// 使它们胜出：
//   ssh-mcp-read_*                 -> deny   （read 通道 upload/download 及未列出工具被拒）
//   ssh-mcp-read_execute-command   -> allow  （仅当完整命令匹配策略正则时放行）
//   ssh-mcp-read_list-servers      -> allow
//   ssh-mcp-ask_*                  -> ask    （ask 通道所有工具需用户批准）
// 同时清除遗留的旧 ssh-mcp-server* 权限键。
export default async function sshMcpPermissionsPlugin() {
  return {
    async config(config) {
      const agentConfig = config.agent ?? (config.agent = {});
      const names = new Set([...Object.keys(agentConfig), "build", "plan", "general", "explore"]);
      for (const name of names) {
        const agent = agentConfig[name] ?? (agentConfig[name] = {});
        let permission = agent.permission;
        if (typeof permission === "string") {
          // 字符串简写（如 "ask"）等价于 "*" 通配，保留其语义再加细粒度规则。
          permission = { "*": permission };
        } else if (!permission || typeof permission !== "object" || Array.isArray(permission)) {
          permission = {};
        }
        // 清除旧 ssh-mcp-server 通道的遗留权限键。
        for (const key of Object.keys(permission)) {
          if (key === "ssh-mcp-server" || key.startsWith("ssh-mcp-server")) {
            delete permission[key];
          }
        }
        // 先删除再按固定顺序重放；observer 仅负责读取视觉附件，
        // 不授予任何 SSH MCP 通道。
        for (const [key] of PERMISSION_KEYS) delete permission[key];
        const permissionKeys = name === "observer" ? OBSERVER_PERMISSION_KEYS : PERMISSION_KEYS;
        for (const [key, value] of permissionKeys) permission[key] = value;
        agent.permission = permission;
      }
    },
  };
}

const PERMISSION_KEYS = [
  ["ssh-mcp-read_*", "deny"],
  ["ssh-mcp-read_execute-command", "allow"],
  ["ssh-mcp-read_list-servers", "allow"],
  ["ssh-mcp-ask_*", "ask"],
];

const OBSERVER_PERMISSION_KEYS = [
  ["ssh-mcp-read_*", "deny"],
  ["ssh-mcp-ask_*", "deny"],
];
