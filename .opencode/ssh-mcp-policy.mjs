/**
 * 共享只读命令策略模块（项目本地，零依赖）。
 *
 * 单一策略源：.opencode/ssh-mcp-command-policy.json（declarative JSON），供
 * SSH read 通道 commandWhitelist 使用（expandCommandPatterns 的完整正则数组）。
 * ask 通道不再注入任何白/黑名单，仅由 OpenCode 权限 `ssh-mcp-ask_*: ask` 保护。
 *
 * 设计约束：
 *   - segments 为「命令段」正则片段；expandCommandPatterns 将其包装为全文锚定
 *     ^<body>(?![\\s\\S]) 的单命令正则，并生成仅由已批准段组成的流水线正则
 *     ^(?:B1|B2|...)(?:[ \\t]+\\|[ \\t]+(?:B1|B2|...)){1,max-1}(?![\\s\\S])。
 *   - 段间管道要求两侧空白（[ \\t]+\\|[ \\t]+），禁止 ; && || < > 重定向、命令替换、
 *     反引号、后台 &、换行等——这些字符不在参数字符集内，整体无法匹配即拒绝（fail-closed）。
 *   - 负向断言（如拒绝 ss -K / date -s / tail -f）只会导致更多拒绝，不会放行。
 */
import { readFileSync } from "node:fs";

const END_ANCHOR = "(?![\\s\\S])";
const DEFAULT_MAX_SEGMENTS = 4;

/** 读取并校验策略文件；返回原始 policy 对象（segments + pipelines 配置）。 */
export function loadPolicy(policyPath) {
  let raw;
  try {
    raw = JSON.parse(readFileSync(policyPath, "utf8"));
  } catch (err) {
    throw new Error(`Cannot read policy file: ${err.message}`);
  }
  validatePolicy(raw);
  return raw;
}

/** 校验 policy 结构：非空 segments 对象、每段为非空可编译正则片段、pipelines.maxSegments 合法。 */
export function validatePolicy(policy) {
  if (!policy || typeof policy !== "object" || Array.isArray(policy)) {
    throw new Error("Policy must be an object");
  }
  const segments = policy.segments;
  if (!segments || typeof segments !== "object" || Array.isArray(segments) || Object.keys(segments).length === 0) {
    throw new Error("Policy must contain a non-empty 'segments' object");
  }
  for (const [name, body] of Object.entries(segments)) {
    if (typeof body !== "string" || body.trim().length === 0) {
      throw new Error(`Segment '${name}' must be a non-empty string`);
    }
    try {
      // 以全文锚定形式编译，确保片段本身可作正则使用
      // eslint-disable-next-line no-new
      new RegExp(`^(?:${body})${END_ANCHOR}`);
    } catch (err) {
      throw new Error(`Invalid regex for segment '${name}': ${err.message}`);
    }
  }
  const maxSegments = policy.pipelines?.maxSegments;
  if (maxSegments !== undefined && (!Number.isInteger(maxSegments) || maxSegments < 2 || maxSegments > 10)) {
    throw new Error("pipelines.maxSegments must be an integer between 2 and 10");
  }
  return policy;
}

/**
 * 展开策略为「注入用」完整命令正则数组（SSH read 通道 commandWhitelist）：
 *   - 每个 segments 片段 → ^<body>(?![\\s\\S])
 *   - 当 pipelines.maxSegments >= 2 时，追加一条流水线正则（2..maxSegments 段，段间 [ \\t]+\\|[ \\t]+）。
 * 返回数组与 JSON 字段顺序无关；仅 read 白名单使用（ask 通道不注入命令策略）。
 */
export function expandCommandPatterns(policy) {
  validatePolicy(policy);
  const bodies = Object.values(policy.segments);
  const singles = bodies.map((body) => `^${body}${END_ANCHOR}`);
  const maxSegments = policy.pipelines?.maxSegments ?? DEFAULT_MAX_SEGMENTS;
  if (maxSegments < 2) {
    return singles;
  }
  const alt = `(?:${bodies.join("|")})`;
  const pipeline = `^${alt}(?:[ \\t]+\\|[ \\t]+${alt}){1,${maxSegments - 1}}${END_ANCHOR}`;
  return [...singles, pipeline];
}
