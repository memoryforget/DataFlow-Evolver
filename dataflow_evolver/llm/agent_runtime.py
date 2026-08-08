"""Runtime boundary shared by PipelineAgent backends."""
from __future__ import annotations

from typing import Protocol

from dataflow_evolver.llm.agent_sdk import AgentRunResult, run_pipeline_agent
from dataflow_evolver.utils.config import Config


class AgentRuntime(Protocol):
    """Execute one autonomous pipeline-authoring session."""

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
        )


def build_agent_runtime(agent_cfg: Config) -> AgentRuntime:
    """Build the selected backend without leaking it into PipelineAgent."""
    backend = str(agent_cfg.get("backend", "claude") or "claude").strip().lower()
    if backend == "claude":
        return ClaudeAgentRuntime()
    if backend == "opencode":
        from dataflow_evolver.llm.opencode_runtime import OpenCodeAgentRuntime

        return OpenCodeAgentRuntime()
    if backend == "codex":
        from dataflow_evolver.llm.codex_runtime import CodexAgentRuntime

        return CodexAgentRuntime()
    raise ValueError(
        f"unknown agent.backend: {backend!r}; expected claude, opencode, or codex"
    )
