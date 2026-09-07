#!/usr/bin/env node
/**
 * ssh-mcp-server 双通道包装器（项目本地，不落盘临时配置/密钥）。
 *
 * 参数：
 *   --mode read|ask          通道模式
 *   --source-config <path>   外部凭据源 JSON（仅内存读取，绝不打印）
 *   --policy <path>          只读命令策略 JSON（.opencode/ssh-mcp-command-policy.json）
 *
 * 行为：
 *   - read：对每个连接注入 commandWhitelist = 展开后的只读策略 patterns，
 *     并设置不可用的 SFTP 本地/远端路径白名单作为纵深防御。
 *   - ask ：不注入任何 commandWhitelist/commandBlacklist 或 SFTP 路径限制；
 *     仅由 OpenCode 权限 `ssh-mcp-ask_*: ask` 保护，所有命令在用户批准后才执行。
 *   - 通过 CommandLineParser.normalizeConfig 归一化，monkeypatch
 *     CommandLineParser.parseArgs 返回构造好的 configs，再直接运行 SshMcpServer。
 *   - 错误只输出到 stderr，且不含配置内容。
 */
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

// 共享只读命令策略模块（单一来源：.opencode/ssh-mcp-command-policy.json）。
import { loadPolicy as loadPolicySource, expandCommandPatterns } from "./ssh-mcp-policy.mjs";

// 通过绝对文件 URL 引入已安装包内部模块（不复制依赖进仓库）。
const PKG_BUILD = "/home/lzyl/.config/opencode/node_modules/@fangjunjie/ssh-mcp-server/build";
const { CommandLineParser } = await import(`${PKG_BUILD}/cli/command-line-parser.js`);
const { SshMcpServer } = await import(`${PKG_BUILD}/core/mcp-server.js`);

/** read 模式下 SFTP 本地/远端路径白名单的“不可用”哨兵路径。 */
export const SFTP_DISABLED_SENTINEL = "/nonexistent/ssh-mcp-read-sftp-disabled";

/** 解析包装器自身 argv（支持 --key value 与 --key=value）。 */
export function parseArgv(argv) {
  const out = { mode: undefined, sourceConfig: undefined, policy: undefined };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    let key = arg;
    let value;
    const eq = arg.indexOf("=");
    if (arg.startsWith("--") && eq > 0) {
      key = arg.slice(0, eq);
      value = arg.slice(eq + 1);
    }
    if (value === undefined && i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      value = argv[++i];
    }
    switch (key) {
      case "--mode":
        out.mode = value;
        break;
      case "--source-config":
        out.sourceConfig = value;
        break;
      case "--policy":
        out.policy = value;
        break;
      default:
        throw new Error(`Unknown argument: ${key}`);
    }
  }
  return out;
}

/**
 * 读取并展开共享策略为「注入用」完整命令正则数组（兼容既有 API 形状：字符串数组）。
 * 实际读取/校验/展开委托 .opencode/ssh-mcp-policy.mjs；仅供 read 白名单注入
 * （ask 通道不注入命令策略）。
 */
export function loadPolicy(policyPath) {
  return expandCommandPatterns(loadPolicySource(policyPath));
}

/** 读取凭据源 JSON：仅内存使用，绝不打印内容。 */
export function loadSourceConfig(sourceConfigPath) {
  let raw;
  try {
    raw = JSON.parse(readFileSync(sourceConfigPath, "utf8"));
  } catch (err) {
    throw new Error(`Cannot read source config: ${err.message}`);
  }
  if (!Array.isArray(raw) && (typeof raw !== "object" || raw === null)) {
    throw new Error("Source config must be an array or object of SSH connection configs");
  }
  return raw;
}

/**
 * 构造连接 configs 映射（key = 连接名）。
 * read 模式注入 commandWhitelist + SFTP 哨兵路径；ask 模式不注入任何命令策略
 * 或路径限制（仅靠 OpenCode 权限 `ssh-mcp-ask_*: ask` 保护）。
 * 两种模式都先清除来源配置中的 commandWhitelist/commandBlacklist/
 * allowedLocalPaths/allowedRemotePaths，避免陈旧来源限制残留；
 * ask 模式再经 finalizeConfig 删除 normalizeConfig 重建的 undefined 键，
 * 使最终 ask 配置字面不含这四键（read 输出保持原样）。
 */
export function buildConfigs(sourceConfigPath, policyPath, mode) {
  if (mode !== "read" && mode !== "ask") {
    throw new Error(`--mode must be 'read' or 'ask', got: ${String(mode)}`);
  }
  const patterns = loadPolicy(policyPath);
  const raw = loadSourceConfig(sourceConfigPath);
  const configs = {};

  const inject = (entry) => {
    const clone = { ...entry };
    delete clone.commandWhitelist;
    delete clone.commandBlacklist;
    delete clone.allowedLocalPaths;
    delete clone.allowedRemotePaths;
    if (mode === "read") {
      clone.commandWhitelist = patterns.slice();
      clone.allowedLocalPaths = [SFTP_DISABLED_SENTINEL];
      clone.allowedRemotePaths = [SFTP_DISABLED_SENTINEL];
    }
    return clone;
  };

  if (Array.isArray(raw)) {
    for (const entry of raw) {
      if (!entry || typeof entry !== "object" || !entry.name || !entry.host || !entry.port || !entry.username) {
        throw new Error("Each array config entry must include name, host, port, username");
      }
      const normalized = CommandLineParser.normalizeConfig(inject(entry));
      configs[normalized.name] = finalizeConfig(normalized, mode);
    }
  } else {
    for (const [name, entry] of Object.entries(raw)) {
      if (!entry || typeof entry !== "object") {
        throw new Error("Source config entries must be objects");
      }
      const normalized = CommandLineParser.normalizeConfig(inject(entry));
      normalized.name = name;
      configs[name] = finalizeConfig(normalized, mode);
    }
  }
  return configs;
}

/**
 * 归一化后收尾：ask 模式删除 normalizeConfig 重新引入的四键（值为 undefined，
 * 非来源值——来源值已在 inject 中被删除），使最终配置字面不含这些键；
 * read 模式不做任何改动，保持输出与以往完全一致。
 */
function finalizeConfig(config, mode) {
  if (mode !== "ask") return config;
  delete config.commandWhitelist;
  delete config.commandBlacklist;
  delete config.allowedLocalPaths;
  delete config.allowedRemotePaths;
  return config;
}

/** 包装器主入口：monkeypatch parseArgs 后直接运行 SshMcpServer。 */
export async function main(argv) {
  const args = parseArgv(argv);
  if (!args.mode) throw new Error("--mode <read|ask> is required");
  if (!args.sourceConfig) throw new Error("--source-config <path> is required");
  if (!args.policy) throw new Error("--policy <path> is required");
  const configs = buildConfigs(args.sourceConfig, args.policy, args.mode);
  CommandLineParser.parseArgs = () => ({ configs, preConnect: false });
  const server = new SshMcpServer();
  await server.run();
}

// 仅直接执行时启动服务；被测试 import 时不触发。
if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(`[ssh-mcp-wrapper] ${err && err.message ? err.message : String(err)}`);
    process.exit(1);
  });
}
