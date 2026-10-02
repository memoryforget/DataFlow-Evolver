"""Runtime boundary shared by PipelineAgent backends."""
from __future__ import annotations

from typing import Protocol

from dataflow_evolver.llm.agent_sdk import (
    AgentRunResult,
    default_diagnostic_system_prompt_append,
    default_system_prompt_append,
    run_pipeline_agent,
)
from dataflow_evolver.utils.config import Config


class AgentRuntime(Protocol):
    """Execute one autonomous agent session."""

    def run(
        self,
        prompt: str,
        *,
        cwd: str,
        agent_cfg: Config,
        tool_log_dir: str | None,
        phase: str,
    ) -> AgentRunResult: ...


class ClaudeAgentRuntime:
    """Adapter around the existing Claude Agent SDK implementation."""

    def __init__(self, system_prompt_append: str) -> None:
        self.system_prompt_append = system_prompt_append

    def run(
        self,
        prompt: str,
        *,
        cwd: str,
        agent_cfg: Config,
        tool_log_dir: str | None,
        phase: str,
    ) -> AgentRunResult:
        return run_pipeline_agent(
            prompt,
            cwd=cwd,
            agent_cfg=agent_cfg,
            tool_log_dir=tool_log_dir,
            phase=phase,
            system_prompt_append=self.system_prompt_append,
        )


def _build_agent_runtime(
    agent_cfg: Config,
    system_prompt_append: str,
) -> AgentRuntime:
    backend = str(agent_cfg.get("backend", "claude") or "claude").strip().lower()
    if backend == "claude":
        return ClaudeAgentRuntime(system_prompt_append)
    if backend == "opencode":
        from dataflow_evolver.llm.opencode_runtime import OpenCodeAgentRuntime

        return OpenCodeAgentRuntime(system_prompt_append)
    if backend == "codex":
        from dataflow_evolver.llm.codex_runtime import CodexAgentRuntime

        return CodexAgentRuntime(system_prompt_append)
    raise ValueError(
        f"unknown agent.backend: {backend!r}; expected claude, opencode, or codex"
    )


def build_agent_runtime(agent_cfg: Config) -> AgentRuntime:
    """Build PipelineAgent with its fixed authoring system prompt."""
    backend = str(agent_cfg.get("backend", "claude") or "claude").strip().lower()
    return _build_agent_runtime(agent_cfg, default_system_prompt_append(backend))


def build_pipeline_diagnostic_runtime(agent_cfg: Config) -> AgentRuntime:
    """Build Call B with its fixed read-only diagnostic system prompt."""
    backend = str(agent_cfg.get("backend", "claude") or "claude").strip().lower()
    return _build_agent_runtime(
        agent_cfg, default_diagnostic_system_prompt_append(backend)
    )
