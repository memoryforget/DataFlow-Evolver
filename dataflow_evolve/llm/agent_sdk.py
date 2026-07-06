"""Claude Agent SDK 封装：把异步 query() 包成供同步 MCTS 循环调用的 run_pipeline_agent()。

PipelineAgent 通过它启动一次无头 Agent 运行，让 SDK 内的 Claude 借助本地技能与文件
工具把完整 DataFlow 流水线写到节点 workspace。详见 TODO.md §5.9。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from dataflow_evolve.utils.config import Config
from dataflow_evolve.utils.logging import get_logger

logger = get_logger("llm.agent_sdk")

_DEFAULT_SYSTEM_PROMPT_APPEND = (
    "You write DataFlow pipelines with CUSTOM operators for Math SFT.\n\n"

    "## CRITICAL: Read DataFlow Development Guide FIRST\n"
    "Before writing ANY code, you MUST read the DataFlow development guide:\n"
    "1. Read `.claude/skills/dataflow-dev/diagnostics/known_issues.md` for common API errors\n"
    "2. Key facts from the guide:\n"
    "   - FileStorage.write() takes ONLY ONE argument: `storage.write(df)`\n"
    "   - LLM serving uses: `self.llm_serving.generate_from_input(user_inputs=list[str], system_prompt=str)`\n"
    "   - storage.step() usage: `op.run(storage=self.storage.step(), ...)`\n"
    "   - Import from parent module: `from dataflow.operators.reasoning import OpName`\n\n"

    "## Step 1: Design Custom Operators\n"
    "For each processing step, write a dedicated operator class:\n"
    "- `from dataflow.core import OperatorABC`\n"
    "- Inherit OperatorABC, call super().__init__()\n"
    "- run(self, storage, input_*='col', output_*='col') -> list[str]\n"
    "- Inside run: df = storage.read('dataframe') → process → storage.write(df)\n"
    "- LLM operators: accept llm_serving in __init__, store as self.llm_serving\n"
    "- Add get_desc(lang='zh') staticmethod (zh/en)\n"
    "- Do NOT use @OPERATOR_REGISTRY.register() — custom ops skip registry\n"
    "- Each operator in its own file under operators/ in the node workspace\n\n"

    "## Critical API Examples (Copy Exactly):\n"
    "```python\n"
    "# FileStorage API - ONLY ONE ARGUMENT\n"
    "def run(self, storage, input_key='text', output_key='result'):\n"
    "    df = storage.read('dataframe')\n"
    "    # ... process ...\n"
    "    storage.write(df)  # ✓ Correct - ONE argument only\n"
    "    # storage.write(df, 'dataframe')  # ✗ WRONG - will crash!\n"
    "    return [output_key]\n\n"

    "# LLM Serving API - use generate_from_input()\n"
    "def __init__(self, llm_serving, ...):\n"
    "    super().__init__()\n"
    "    self.llm_serving = llm_serving  # Must be named llm_serving\n\n"

    "def run(self, storage, input_key='text', output_key='result'):\n"
    "    df = storage.read('dataframe')\n"
    "    user_inputs = df[input_key].tolist()  # list[str]\n"
    "    \n"
    "    # ✓ Correct API call\n"
    "    responses = self.llm_serving.generate_from_input(\n"
    "        user_inputs=user_inputs,\n"
    "        system_prompt='You are a helpful assistant'\n"
    "    )\n"
    "    \n"
    "    # Handle None responses (API failures)\n"
    "    df[output_key] = [r if r is not None else '' for r in responses]\n"
    "    storage.write(df)\n"
    "    return [output_key]\n"
    "```\n\n"

    "## Step 2: Write Pipeline\n"
    "- Import custom operators from local operators/ directory\n"
    "- `from dataflow.pipeline import PipelineABC`\n"
    "- Operator instances as member vars (self.op_stepN) for compile() detection\n"
    "- forward(): self.op_stepN.run(storage=self.storage.step(), ...)\n"
    "- Write pipeline.py + decision.json to the node directory\n"
    "- Self-contained with `if __name__ == '__main__':` entry\n\n"

    "## Step 3: Compile-Test (MANDATORY)\n"
    "After writing all files, verify via compile():\n"
    "    pipeline = MyPipeline(...)\n"
    "    pipeline.compile()\n"
    "    if os.environ.get('DF_COMPILE_ONLY'): sys.exit(0)\n"
    "    pipeline.forward()\n"
    "Run: DF_COMPILE_ONLY=1 python <node_dir>/pipeline.py\n"
    "Fix any KeyError/ImportError/TypeError and re-run until compile passes.\n\n"

    "## MCP Reference (Use for Implementation Patterns)\n"
    "The dataflow-operators MCP exposes existing reasoning/math operators with FULL SOURCE CODE.\n"
    "Use get_operator_detail to study:\n"
    "- Prompt construction patterns\n"
    "- LLM call conventions (see 'source_code' field)\n"
    "- Response parsing techniques\n"
    "Then write your own operators following those patterns.\n\n"

    "## Error Fixing Strategy\n"
    "If compile/execution fails:\n"
    "1. Read `.claude/skills/dataflow-dev/diagnostics/known_issues.md`\n"
    "2. Match error message to Issue #001-#011\n"
    "3. Apply the exact fix from the Issue\n"
    "4. Common errors:\n"
    "   - Issue #011: storage.write(df, 'dataframe') → storage.write(df)\n"
    "   - Issue #010: self.llm_serving.request() → self.llm_serving.generate_from_input()\n\n"

    "## Constraints\n"
    "- Headless: no AskUserQuestion, no git/gh subflows.\n"
    "- Correct imports: `from dataflow.core import OperatorABC`,\n"
    "  `from dataflow.pipeline import PipelineABC`,\n"
    "  `from dataflow.utils.storage import FileStorage`,\n"
    "  `from dataflow.serving import APILLMServing_request`\n"
    "- Handle LLM parse failures gracefully (skip bad rows, don't crash)\n"
    "- Strip <think> tags for non-CoT target fields\n"
    "- If compile/import fails, Read source to debug — fix in place.\n"
)



@dataclass
class AgentRunResult:
    """一次 Agent SDK 运行的结果。"""

    success: bool
    result_text: str = ""                  # ResultMessage.result 的最终文本
    assistant_texts: list[str] = field(default_factory=list)  # 过程中的助手文本块
    error: str | None = None
    error_type: str | None = None          # 异常类名，便于上层分类处理
    exit_code: int | None = None           # ProcessError 时填充


def _build_options(agent_cfg: Config, cwd: str) -> Any:
    """据 agent_sdk 配置段构造 ClaudeAgentOptions。

    - tools：用 claude_code 预设放开**全部内置工具**（不再用 allowed_tools 白名单限制）。
    - system_prompt：保留 claude_code 预设，仅 **append** 我们的 DataFlow 指令（不覆盖系统提示）。
    - mcp_servers：挂载 dataflow-operators MCP server 供 agent 动态发现 operator。
    - max_turns：单次 SDK 运行的最大轮数（默认 50）。
    SDK 鉴权（ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL）经环境变量传入，子进程默认继承 os.environ。
    """
    from claude_agent_sdk import ClaudeAgentOptions

    from dataflow_evolve.llm.mcp_operators import operator_mcp_server

    append_text = agent_cfg.get("system_prompt_append", _DEFAULT_SYSTEM_PROMPT_APPEND)
    return ClaudeAgentOptions(
        cwd=cwd,
        model=agent_cfg.get("model", None),
        tools={"type": "preset", "preset": "claude_code"},
        system_prompt={"type": "preset", "preset": "claude_code", "append": append_text},
        permission_mode=agent_cfg.get("permission_mode", "bypassPermissions"),
        setting_sources=list(agent_cfg.get("setting_sources", ["project"])),
        max_turns=agent_cfg.get("max_turns", 50),
        mcp_servers={"dataflow-operators": operator_mcp_server},
    )


async def _run(prompt: str, options: Any) -> AgentRunResult:
    from claude_agent_sdk import (
        AssistantMessage,
        CLIConnectionError,
        CLIJSONDecodeError,
        CLINotFoundError,
        ProcessError,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
    )

    assistant_texts: list[str] = []
    result_text = ""
    turn = 0
    try:
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                turn += 1
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        assistant_texts.append(block.text)
                        preview = block.text[:120].replace("\n", " ")
                        logger.debug("Agent SDK turn %d: %s", turn, preview)
                    elif isinstance(block, ToolUseBlock):
                        logger.debug("Agent SDK turn %d: tool_use %s", turn, block.name)
            elif isinstance(msg, ResultMessage):
                result_text = msg.result or ""
    except CLINotFoundError as exc:
        return AgentRunResult(False, error=str(exc), error_type="CLINotFoundError")
    except ProcessError as exc:
        return AgentRunResult(
            False, error=str(exc), error_type="ProcessError",
            exit_code=getattr(exc, "exit_code", None),
        )
    except CLIJSONDecodeError as exc:
        return AgentRunResult(False, error=str(exc), error_type="CLIJSONDecodeError")
    except CLIConnectionError as exc:
        return AgentRunResult(False, error=str(exc), error_type="CLIConnectionError")
    except Exception as exc:
        return AgentRunResult(False, error=str(exc), error_type=type(exc).__name__)

    return AgentRunResult(
        success=True,
        result_text=result_text,
        assistant_texts=assistant_texts,
    )


# query 在模块级惰性绑定，方便测试时打桩
def query(*args: Any, **kwargs: Any) -> Any:  # noqa: D401
    from claude_agent_sdk import query as _query

    return _query(*args, **kwargs)


def run_pipeline_agent(prompt: str, cwd: str, agent_cfg: Config) -> AgentRunResult:
    """供同步 MCTS 循环调用：在 `cwd` 下跑一次 Agent SDK。

    捕获所有 SDK 级异常并降级为 AgentRunResult(success=False)，绝不抛到主循环。
    """
    options = _build_options(agent_cfg, cwd)
    logger.info("启动 Agent SDK 运行：cwd=%s", cwd)
    result = asyncio.run(_run(prompt, options))
    if not result.success:
        logger.error("Agent SDK 运行失败 [%s]：%s", result.error_type, result.error)
    return result
