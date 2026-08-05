# OpenCode runtime

DataFlow-Evolver 的 RecipeAgent 与 PipelineAgent 通过同一个 `AgentRuntime` 协议运行。
设置 `AGENT_RUNTIME=opencode` 后只替换 agent harness；RecipeValidator、搜索树、prompt、
recipe/pipeline 产物契约和后续 Review/Train/Eval 均不变，因此可直接用于 Claude Agent SDK
与 OpenCode 的消融实验。

## 安装

OpenCode CLI 由 Node 环境安装；Python 侧继续使用现有 `df-agent` 环境中的 MCP SDK：

```bash
npm install -g opencode-ai
opencode --version
```

如果全局 npm bin 不在 `PATH`，把绝对路径放进 `OPENCODE_BIN`。框架不依赖常驻
`opencode serve`：每次 stage 使用官方的 `opencode run --format json`，由 CLI 管理本地
server 与 session，并在瞬时失败时用 session id 续跑。

## 配置

最小配置沿用现有 Anthropic key/base URL：

```bash
export AGENT_RUNTIME=opencode
export OPENCODE_BIN=/vepfs-mlp2/c20250602/500050/lh/xqf/.npm-global/bin/opencode
export OPENCODE_MODEL=anthropic/claude-opus-4-8
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=http://123.129.219.111:3000/
```

可选项：

- `OPENCODE_PROVIDER`：`OPENCODE_MODEL` 不含 `/` 时补的 provider，默认 `anthropic`。
- `OPENCODE_API_KEY_ENV`：key 所在环境变量名，默认 `ANTHROPIC_API_KEY`。配置只保存变量名，
  不会把 key 写进命令行、日志或 `opencode.json`。
- `OPENCODE_BASE_URL`：provider base URL；留空沿用 `ANTHROPIC_BASE_URL`。
- `OPENCODE_VARIANT`：OpenCode provider-specific reasoning variant；留空沿用 `agent_sdk.effort`。
- `OPENCODE_MAX_STEPS`：单 stage 最大 agent steps，默认 50。
- `OPENCODE_MCP_TIMEOUT_MS`：local MCP 启动超时，默认 30000ms；算子 registry 首次加载较重，
  不使用 OpenCode 的 5000ms 默认值。

也可以直接修改 `configs/default.yaml` 的 `agent_sdk.opencode` 段。运行时使用
`OPENCODE_CONFIG_CONTENT` 注入每个 stage 的最高优先级临时配置，不在 agent workspace
落一份包含实验参数的 `opencode.json`。

## 两个 stage 的工具边界

OpenCode local MCP 是 stdio 子进程，但复用与 Claude runtime 相同的 Python tool handler：

| stage | MCP | 内置能力 |
|---|---|---|
| RecipeAgent | `dataset-search`、可选 `data-discovery` | 读 recipe/context、只写 recipe；禁用 bash/web/task/skill |
| PipelineAgent | `dataflow-operators` | 读写节点代码、bash compile-test、`dataflow-dev` skill；禁用外部数据 MCP/web/task |

每次 run 只把该 stage 的 MCP 写进 inline config，因而不是“全部 server 都挂载后靠 prompt
自觉不用”。外部数据仍由 RecipeAgent 下载到统一的 `data_links/`，PipelineAgent 只消费
validated recipe 中声明的本地来源。

## skills 路径

无需复制或改名现有 skill。OpenCode 原生扫描当前工作目录向上的以下位置：

- `.opencode/skills/<name>/SKILL.md`
- `.claude/skills/<name>/SKILL.md`
- `.agents/skills/<name>/SKILL.md`

项目现有的 `_agent_root/.claude/skills/dataflow-dev/SKILL.md` 已有合法的 `name` 和
`description` frontmatter；CLI 的 `--dir` 固定为 `_agent_root`，因此 PipelineAgent 可以直接
发现它。RecipeAgent 的 skill 权限被关闭，避免把实现期上下文注入配方规划阶段。

## 快速检查

先做不调用模型的配置/CLI检查：

```bash
opencode --version
AGENT_RUNTIME=opencode \
python -c 'from dataflow_evolver.utils.config import load_config; from dataflow_evolver.llm.agent_runtime import build_agent_runtime; c=load_config("configs/default.yaml"); print(type(build_agent_runtime(c.agent_sdk)).__name__)'
```

真正端到端测试继续使用原来的 `python -m dataflow_evolver.main ...` 命令，只额外 export
上面的 OpenCode 变量。建议先把 `PTB_TARGET_SAMPLES`、`PTB_MAX_ITERATIONS` 调小，分别跑一次
forced `hybrid` 和 `synthetic`，再进行完整消融。
