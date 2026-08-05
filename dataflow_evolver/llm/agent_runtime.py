"""Stage-agent runtime boundary.

Recipe and pipeline agents depend on this small protocol instead of a concrete
agent SDK.  Claude and OpenCode implement the same one-shot ``run`` contract,
without leaking CLI/session details into the search or prompt layers.
"""
from __future__ import annotations

from typing import Protocol

from dataflow_evolver.llm.agent_sdk import AgentRunResult, run_pipeline_agent
from dataflow_evolver.utils.config import Config


class AgentRuntime(Protocol):
    def run(
        self,
        prompt: str,
        *,
        cwd: str,
        agent_cfg: Config,
        tool_log_dir: str | None,
        phase: str,
        system_prompt_append: str | None = None,
        tool_profile: str,
    ) -> AgentRunResult: ...


class ClaudeAgentRuntime:
    """Claude Agent SDK adapter for the stage-neutral runtime contract."""

    def run(
        self,
        prompt: str,
        *,
        cwd: str,
        agent_cfg: Config,
        tool_log_dir: str | None,
        phase: str,
        system_prompt_append: str | None = None,
        tool_profile: str,
    ) -> AgentRunResult:
        return run_pipeline_agent(
            prompt,
            cwd=cwd,
            agent_cfg=agent_cfg,
            tool_log_dir=tool_log_dir,
            phase=phase,
            system_prompt_append=system_prompt_append,
            tool_profile=tool_profile,
        )


def build_agent_runtime(agent_cfg: Config) -> AgentRuntime:
    """Build the configured runtime without exposing SDK details to agents."""
    runtime = str(agent_cfg.get("runtime", "claude") or "claude").strip().lower()
    if runtime == "claude":
        return ClaudeAgentRuntime()
    if runtime == "opencode":
        from dataflow_evolver.llm.opencode_runtime import OpenCodeAgentRuntime

        return OpenCodeAgentRuntime()
    raise ValueError(f"未知 agent runtime: {runtime!r}")
