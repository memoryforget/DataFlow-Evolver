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
    "You write DataFlow pipelines. Use the generating-dataflow-pipeline skill to author "
    "pipelines; consult dataflow-dev (knowledge_base / known_issues) for operators beyond "
    "the core set and to fix execution errors. Do NOT run interactive (AskUserQuestion) or "
    "git/gh repo-sync subflows — this is a headless automated run."
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
    - max_turns：单次 SDK 运行的最大轮数（默认 10）；不设预算上限。
    SDK 鉴权（ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL）经环境变量传入，子进程默认继承 os.environ。
    """
    from claude_agent_sdk import ClaudeAgentOptions

    append_text = agent_cfg.get("system_prompt_append", _DEFAULT_SYSTEM_PROMPT_APPEND)
    return ClaudeAgentOptions(
        cwd=cwd,
        model=agent_cfg.get("model", None),
        tools={"type": "preset", "preset": "claude_code"},
        system_prompt={"type": "preset", "preset": "claude_code", "append": append_text},
        permission_mode=agent_cfg.get("permission_mode", "bypassPermissions"),
        setting_sources=list(agent_cfg.get("setting_sources", ["project"])),
        max_turns=agent_cfg.get("max_turns", 10),
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
    )

    assistant_texts: list[str] = []
    result_text = ""
    try:
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        assistant_texts.append(block.text)
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
