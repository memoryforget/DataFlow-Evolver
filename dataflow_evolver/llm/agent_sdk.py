"""Claude Agent SDK 封装：把异步 query() 包成供同步 MCTS 循环调用的 run_pipeline_agent()。

PipelineAgent 通过它启动一次无头 Agent 运行，让 SDK 内的 Claude 借助本地技能与文件
工具把完整 DataFlow 流水线写到节点 workspace。详见 TODO.md §5.9。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from dataflow_evolver.utils.config import Config
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("llm.agent_sdk")


class AgentInfraError(RuntimeError):
    """Agent SDK 基础设施级失败（429/5xx/连接/超时/进程崩溃等），区别于「跑完但产出不合格」。

    `retryable`：True=瞬时错误（重试可能恢复，如 429/超时/连接）；False=永久错误
    （如 CLI 缺失、鉴权失败，重试无益）。主循环据此决定「退避重试本轮」还是「终止运行」。
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


# 判定为「瞬时、可重试」的错误文本标记（对错误信息 + 最近若干条助手文本做小写子串匹配）。
_RETRYABLE_ERROR_TYPES = {"Timeout", "CLIConnectionError"}
_TRANSIENT_MARKERS = (
    "429", "request rejected", "temporary issues", "temporarily unavailable",
    "temporarily", "overloaded", "rate limit", "rate_limit", "too many requests",
    "502", "503", "504", "internal server error", "bad gateway",
    "service unavailable", "gateway timeout", "timed out", "timeout",
    "connection reset", "connection error", "connection refused", "econnreset",
)

_DEFAULT_SYSTEM_PROMPT_APPEND = (
    "You write DataFlow pipelines with CUSTOM operators for Math SFT.\n\n"

    "## CRITICAL: Read API Reference FIRST\n"
    "Before writing ANY code, you MUST read:\n"
    "1. `.claude/skills/dataflow-dev/templates/operator_api_reference.md` - Complete operator example with correct APIs\n"
    "2. `.claude/skills/dataflow-dev/diagnostics/known_issues.md` - Common API errors and fixes\n\n"

    "Key facts you'll learn:\n"
    "- FileStorage.write() takes ONLY ONE argument: `storage.write(df)`\n"
    "- LLM serving uses: `self.llm_serving.generate_from_input(user_inputs=list[str], system_prompt=str)`\n"
    "- Operator run() must return list of output keys: `return ['output_key']`\n"
    "- Handle None responses: `[r if r is not None else '' for r in responses]`\n\n"

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

    "## Critical API Examples (Copy from operator_api_reference.md):\n"
    "```python\n"
    "# FileStorage API - ONLY ONE ARGUMENT\n"
    "def run(self, storage, input_key='text', output_key='result'):\n"
    "    df = storage.read('dataframe')\n"
    "    # ... process ...\n"
    "    storage.write(df)  # ✓ Correct - ONE argument only\n"
    "    return [output_key]  # ✓ Must return output keys\n\n"

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
    "Use get_operator_detail(name='ReasoningAnswerGenerator') to study:\n"
    "- Prompt construction patterns (see source_code field)\n"
    "- LLM call conventions\n"
    "- Response parsing techniques\n"
    "Then write your own operators following those patterns.\n\n"

    "## Error Fixing Strategy\n"
    "If compile/execution fails:\n"
    "1. Read `.claude/skills/dataflow-dev/diagnostics/known_issues.md`\n"
    "2. Match error message to Issue #001-#011\n"
    "3. Apply the exact fix from the Issue\n"
    "4. If not in known_issues, check DataFlow source code:\n"
    "   - `dataflow/serving/api_llm_serving_request.py` for LLM API\n"
    "   - `dataflow/utils/storage.py` for FileStorage API\n\n"

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
    retryable: bool = False                # 失败是否为瞬时可重试错误（供上层区分基础设施失败类型）
    session_id: str | None = None          # 本次运行的会话 id（用于失败后 resume 续跑，见 run_pipeline_agent）


def _build_options(agent_cfg: Config, cwd: str, resume: str | None = None) -> Any:
    """据 agent_sdk 配置段构造 ClaudeAgentOptions。

    - tools：用 claude_code 预设放开**全部内置工具**（不再用 allowed_tools 白名单限制）。
    - system_prompt：保留 claude_code 预设，仅 **append** 我们的 DataFlow 指令（不覆盖系统提示）。
    - mcp_servers：挂载 dataflow-operators MCP server 供 agent 动态发现 operator。
    - max_turns：单次 SDK 运行的最大轮数（默认 50）。
    - resume：非空则续跑指定会话（载入其历史对话），用于瞬时失败后接着上次进度跑，
      不 fork（继续同一会话 id，把新 turn 追加进去）。
    - effort：Claude 努力/思考档位（low|medium|high|xhigh|max）。配置留空则用模型默认（high）。
    - thinking：显式传 adaptive 思考配置。opus-4-8 等新模型**只支持 adaptive thinking + effort**，
      不接受老式 thinking.enabled（budget）。这里显式指定 {"type":"adaptive"}，从协议层复刻实测
      可用的请求格式，从而不受环境里 CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1（会退化成 thinking.enabled
      → Bedrock 400）影响；即项目自带正确配置，不依赖全局/外部 env。配置 thinking: none 可关闭该显式指定。
    SDK 鉴权（ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL）经环境变量传入，子进程默认继承 os.environ。
    """
    from claude_agent_sdk import ClaudeAgentOptions

    from dataflow_evolver.llm.mcp_operators import operator_mcp_server

    append_text = agent_cfg.get("system_prompt_append", _DEFAULT_SYSTEM_PROMPT_APPEND)
    effort = agent_cfg.get("effort", None) or None  # 空串/None → 不传，走模型默认
    thinking_mode = (agent_cfg.get("thinking", "adaptive") or "adaptive").lower()
    thinking = {"type": "adaptive"} if thinking_mode == "adaptive" else None
    return ClaudeAgentOptions(
        cwd=cwd,
        model=agent_cfg.get("model", None),
        tools={"type": "preset", "preset": "claude_code"},
        system_prompt={"type": "preset", "preset": "claude_code", "append": append_text},
        permission_mode=agent_cfg.get("permission_mode", "bypassPermissions"),
        setting_sources=list(agent_cfg.get("setting_sources", ["project"])),
        max_turns=agent_cfg.get("max_turns", 50),
        mcp_servers={"dataflow-operators": operator_mcp_server},
        resume=resume,
        effort=effort,
        thinking=thinking,
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
    # 会话 id：init/assistant/result 等消息都可能携带；捕获用于失败后 resume 续跑。
    # 用 nonlocal-ish 闭包变量以便异常路径也能带出已捕获的 session_id。
    session_id: str | None = None
    try:
        async for msg in query(prompt=prompt, options=options):
            sid = getattr(msg, "session_id", None)
            if sid:
                session_id = sid
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
    except (
        CLINotFoundError, ProcessError, CLIJSONDecodeError, CLIConnectionError, Exception,
    ) as exc:
        # 关键：CLI 在 turn 中途 API 报错时，真实错误（如 "API Error: 400 ... invalid beta flag"）
        # 是作为 assistant 文本返回的，而 SDK 抛出的 ProcessError 常被替换成
        # "Claude Code returned an error result: <subtype>"（subtype 可能是误导性的 "success"）。
        # 故这里把已收到的末条助手文本并入错误、并把 assistant_texts 带出，供 _is_retryable 判定，
        # 也让日志能显示真实原因，而非无信息的 "success"。
        err = str(exc)
        tail = assistant_texts[-1].strip() if assistant_texts else ""
        if tail and tail not in err:
            err = f"{err}｜CLI 末条输出: {tail[:800]}"
        etype = "ProcessError" if isinstance(exc, ProcessError) else type(exc).__name__
        return AgentRunResult(
            False, error=err, error_type=etype,
            exit_code=getattr(exc, "exit_code", None),
            assistant_texts=assistant_texts, session_id=session_id,
        )

    return AgentRunResult(
        success=True,
        result_text=result_text,
        assistant_texts=assistant_texts,
        session_id=session_id,
    )


# query 在模块级惰性绑定，方便测试时打桩
def query(*args: Any, **kwargs: Any) -> Any:  # noqa: D401
    from claude_agent_sdk import query as _query

    return _query(*args, **kwargs)


def _is_retryable(result: AgentRunResult) -> bool:
    """据错误类型与错误/助手文本判断该失败是否为瞬时、可重试错误。"""
    if result.error_type in _RETRYABLE_ERROR_TYPES:
        return True
    haystack = " ".join([result.error or ""] + result.assistant_texts[-3:]).lower()
    return any(marker in haystack for marker in _TRANSIENT_MARKERS)


async def _run_with_timeout(prompt: str, options: Any, timeout: float) -> AgentRunResult:
    """给单次 SDK 运行套超时；<=0 表示不限时。超时会取消协程并向上抛 TimeoutError。"""
    if timeout and timeout > 0:
        return await asyncio.wait_for(_run(prompt, options), timeout)
    return await _run(prompt, options)


# 续跑时发给 agent 的简短提示（原始任务 prompt 已在会话历史里，无需重发整段）。
_RESUME_NUDGE = (
    "The previous run was interrupted by a transient API error. "
    "Continue from where you left off and finish the task: write a working "
    "pipeline.py and decision.json into the target node directory, then self-test "
    "with DF_COMPILE_ONLY=1 until it compiles."
)


def run_pipeline_agent(prompt: str, cwd: str, agent_cfg: Config) -> AgentRunResult:
    """供同步 MCTS 循环调用：在 `cwd` 下跑一次 Agent SDK（带瞬时错误退避重试 + 运行超时 + 续跑）。

    - 瞬时错误（429/5xx/连接/超时）→ 指数退避重试 `max_retries` 次；
    - 续跑（resume_on_retry，默认开）：若上次运行已建立会话（拿到 session_id），重试时
      `resume` 该会话、只发一句简短续跑提示，接着上次进度跑，避免重跑已完成的 turn 白烧 token；
      若上次连会话都没建起（如 turn 1 就 429，无 session_id）则退回全新重跑；
    - 续跑遇到非瞬时错误（如会话丢失/resume 本身出错）→ 丢弃会话、按可重试处理，下次全新重跑，
      绝不因 resume 自身问题把结果判成不可重试（避免上层误熔断）；
    - 永久错误（CLI 缺失/鉴权等，且发生在全新运行）或重试用尽 → 返回 success=False（不抛异常）；
    - 返回结果的 `retryable` 字段标注最终失败是否为瞬时错误，供上层区分处理。
    """
    max_retries = max(1, int(agent_cfg.get("max_retries", 4)))
    base_delay = float(agent_cfg.get("retry_base_delay", 10))
    timeout = float(agent_cfg.get("run_timeout_sec", 1800))
    resume_enabled = bool(agent_cfg.get("resume_on_retry", True))

    resume_id: str | None = None
    result = AgentRunResult(False, error="未发起任何 SDK 运行", error_type="NoAttempt")
    for attempt in range(1, max_retries + 1):
        used_resume = resume_enabled and resume_id is not None
        options = _build_options(agent_cfg, cwd, resume=resume_id if used_resume else None)
        cur_prompt = _RESUME_NUDGE if used_resume else prompt
        logger.info(
            "启动 Agent SDK 运行（尝试 %d/%d，%s）：cwd=%s",
            attempt, max_retries,
            f"resume={resume_id}" if used_resume else "全新会话", cwd,
        )
        try:
            result = asyncio.run(_run_with_timeout(cur_prompt, options, timeout))
        except (asyncio.TimeoutError, TimeoutError):
            result = AgentRunResult(
                False,
                error=f"Agent SDK 运行超过 {timeout:.0f}s 超时",
                error_type="Timeout",
            )

        # 记录会话 id（成功/失败都可能带出），供下次重试续跑
        if result.session_id:
            resume_id = result.session_id

        if result.success:
            return result

        result.retryable = _is_retryable(result)

        # 续跑遇到非瞬时错误：多半是会话丢失 / resume 本身出错。丢弃会话改全新重跑，
        # 并按可重试处理，避免把 resume 的问题误判成不可重试而让上层熔断。
        if used_resume and not result.retryable:
            logger.warning(
                "resume 续跑遇到非瞬时错误（%s），丢弃会话改为全新重跑：%s",
                result.error_type, result.error,
            )
            resume_id = None
            result.retryable = True

        if attempt < max_retries and result.retryable:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Agent SDK 第 %d/%d 次失败（%s，可重试%s），%.0fs 后重试：%s",
                attempt, max_retries, result.error_type,
                "，将续跑" if (resume_enabled and resume_id is not None) else "",
                delay, result.error,
            )
            time.sleep(delay)
            continue

        logger.error(
            "Agent SDK 运行失败 [%s]（retryable=%s，已尝试 %d 次）：%s",
            result.error_type, result.retryable, attempt, result.error,
        )
        return result
    return result
