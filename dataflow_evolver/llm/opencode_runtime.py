"""OpenCode headless runtime for RecipeAgent and PipelineAgent.

The business agents use the same one-shot ``AgentRuntime`` contract for Claude
and OpenCode.  This adapter drives OpenCode's supported headless CLI and parses
its NDJSON event stream; OpenCode itself owns the agent loop, native tools,
skills, sessions and MCP clients.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dataflow_evolver.core.provenance import record_tool_uses, truncate_arg
from dataflow_evolver.llm.agent_sdk import (
    AgentRunResult,
    _RECIPE_RESUME_NUDGE,
    _RESUME_NUDGE,
    _agent_env,
    _is_retryable,
)
from dataflow_evolver.utils.config import Config
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("llm.opencode_runtime")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "https://opencode.ai/config.json"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Config):
        return value.to_dict()
    return dict(value or {})


def _opencode_cfg(agent_cfg: Config) -> dict[str, Any]:
    return _mapping(agent_cfg.get("opencode", {}))


def _model(agent_cfg: Config) -> str:
    cfg = _opencode_cfg(agent_cfg)
    raw = str(cfg.get("model", "") or agent_cfg.get("model", "") or "").strip()
    if not raw:
        raise ValueError("agent_sdk.opencode.model（或 agent_sdk.model）不能为空")
    provider = str(cfg.get("provider", "anthropic") or "anthropic").strip()
    return raw if "/" in raw else f"{provider}/{raw}"


def _mcp_command(profile: str, cwd: str, timeout_ms: int) -> dict[str, Any]:
    return {
        "type": "local",
        "command": [
            sys.executable,
            "-m",
            "dataflow_evolver.llm.mcp_stdio",
            profile,
        ],
        "enabled": True,
        "timeout": timeout_ms,
        "environment": {"DF_AGENT_ROOT": cwd},
    }


def _permissions(tool_profile: str, external_data: bool, discovery: bool) -> dict[str, Any]:
    """Return a deny-by-default, phase-specific OpenCode permission set."""
    common: dict[str, Any] = {
        "*": "deny",
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "question": "deny",
        "task": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "external_directory": "deny",
    }
    if tool_profile == "recipe":
        common.update({"edit": "allow", "bash": "deny", "skill": "deny"})
        if external_data:
            common["dataset-search_*"] = "allow"
            if discovery:
                common["data-discovery_*"] = "allow"
        return common
    if tool_profile == "pipeline":
        common.update(
            {
                "edit": "allow",
                "bash": "allow",
                "lsp": "allow",
                "skill": {"*": "deny", "dataflow-dev": "allow"},
                "dataflow-operators_*": "allow",
            }
        )
        return common
    raise ValueError(f"unknown agent tool profile: {tool_profile!r}")


def build_opencode_config(
    agent_cfg: Config,
    *,
    cwd: str,
    tool_profile: str,
    system_prompt: str,
) -> tuple[dict[str, Any], str, str]:
    """Build the highest-precedence inline config for one phase.

    Keeping this config per invocation prevents RecipeAgent and PipelineAgent
    from sharing MCP visibility.  Credentials are referenced through OpenCode's
    ``{env:NAME}`` syntax and are never serialized into config or command args.
    """
    oc = _opencode_cfg(agent_cfg)
    model = _model(agent_cfg)
    provider = model.split("/", 1)[0]
    external_data = bool(agent_cfg.get("external_data", False))
    discovery = bool(agent_cfg.get("discovery", True))
    agent_name = "df-recipe" if tool_profile == "recipe" else "df-pipeline"
    mcp_timeout_ms = int(oc.get("mcp_timeout_ms", 30000) or 30000)

    mcp: dict[str, Any] = {}
    if tool_profile == "pipeline":
        mcp["dataflow-operators"] = _mcp_command("operators", cwd, mcp_timeout_ms)
    elif tool_profile == "recipe" and external_data:
        mcp["dataset-search"] = _mcp_command("datasets", cwd, mcp_timeout_ms)
        if discovery:
            mcp["data-discovery"] = _mcp_command("discovery", cwd, mcp_timeout_ms)

    provider_cfg: dict[str, Any] = {}
    key_env = str(oc.get("api_key_env", "ANTHROPIC_API_KEY") or "").strip()
    base_url = str(oc.get("base_url", "") or os.environ.get("ANTHROPIC_BASE_URL", "")).strip()
    options: dict[str, Any] = {}
    if key_env:
        options["apiKey"] = f"{{env:{key_env}}}"
    if base_url:
        options["baseURL"] = base_url
    if options:
        provider_cfg[provider] = {"options": options}

    config: dict[str, Any] = {
        "$schema": _SCHEMA,
        "agent": {
            agent_name: {
                "description": f"DataFlow-Evolver {tool_profile} stage",
                "mode": "primary",
                "model": model,
                "prompt": system_prompt,
                "steps": int(oc.get("max_steps", 0) or agent_cfg.get("max_turns", 50)),
                "permission": _permissions(tool_profile, external_data, discovery),
            }
        },
    }
    if provider_cfg:
        config["provider"] = provider_cfg
    if mcp:
        config["mcp"] = mcp
    return config, agent_name, model


def parse_opencode_events(stdout: str) -> AgentRunResult:
    """Parse ``opencode run --format json`` NDJSON into AgentRunResult."""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    texts: list[str] = []
    tools: list[dict[str, Any]] = []
    errors: list[str] = []
    session_id: str | None = None
    turn = 0
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            # Some CLI/bootstrap failures are plain text.  The caller also has
            # stderr and return code, so do not misclassify them as assistant text.
            continue
        session_id = str(event.get("sessionID") or session_id or "") or None
        kind = event.get("type")
        part = event.get("part") or {}
        if kind == "step_start":
            turn += 1
        elif kind == "text":
            text = str(part.get("text") or "")
            if text:
                texts.append(text)
        elif kind == "tool_use":
            state = part.get("state") or {}
            tools.append(
                {
                    "turn": max(turn, 1),
                    "name": str(part.get("tool") or "unknown"),
                    "input": truncate_arg(state.get("input")),
                }
            )
            # A tool may fail and the agent may recover in a later step.  Claude's
            # adapter likewise records that call without turning a completed agent
            # run into infrastructure failure; only a top-level OpenCode error does.
        elif kind == "error":
            payload = event.get("error")
            if isinstance(payload, dict):
                data = payload.get("data")
                message = data.get("message") if isinstance(data, dict) else None
                errors.append(str(message or payload.get("message") or payload.get("name") or payload))
            elif payload:
                errors.append(str(payload))
    return AgentRunResult(
        success=not errors,
        result_text="\n".join(texts),
        assistant_texts=texts,
        error="\n".join(errors) or None,
        error_type="OpenCodeError" if errors else None,
        session_id=session_id,
        tool_uses=tools,
    )


def _run_once(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    tool_profile: str,
    system_prompt: str,
    resume: str | None,
    timeout: float,
) -> AgentRunResult:
    oc = _opencode_cfg(agent_cfg)
    config, agent_name, model = build_opencode_config(
        agent_cfg, cwd=cwd, tool_profile=tool_profile, system_prompt=system_prompt
    )
    binary = str(oc.get("binary", "opencode") or "opencode")
    command = [binary, "run", "--format", "json", "--agent", agent_name, "--model", model, "--dir", cwd]
    effort = str(oc.get("variant", "") or agent_cfg.get("effort", "") or "").strip()
    if effort:
        command.extend(["--variant", effort])
    if resume:
        command.extend(["--session", resume])

    env = os.environ.copy()
    env.update(_agent_env(agent_cfg))
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_PROJECT_ROOT) + (os.pathsep + pythonpath if pythonpath else "")
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, ensure_ascii=False)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout if timeout > 0 else None,
            check=False,
        )
    except FileNotFoundError as exc:
        return AgentRunResult(False, error=str(exc), error_type="CLINotFoundError")
    except subprocess.TimeoutExpired as exc:
        parsed = parse_opencode_events(exc.stdout or "")
        parsed.success = False
        parsed.error = f"OpenCode 运行超过 {timeout:.0f}s 超时"
        parsed.error_type = "Timeout"
        return parsed
    except OSError as exc:
        return AgentRunResult(False, error=str(exc), error_type=type(exc).__name__)

    parsed = parse_opencode_events(completed.stdout)
    parsed.exit_code = completed.returncode
    if completed.returncode != 0:
        parsed.success = False
        detail = parsed.error or completed.stderr.strip() or completed.stdout.strip()
        parsed.error = detail or f"OpenCode exited with code {completed.returncode}"
        parsed.error_type = parsed.error_type or "OpenCodeProcessError"
    return parsed


def run_opencode_agent(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    tool_log_dir: str | None = None,
    phase: str = "",
    system_prompt_append: str | None = None,
    tool_profile: str = "pipeline",
) -> AgentRunResult:
    """Run OpenCode with the same retry/resume semantics as the Claude adapter."""
    if tool_profile not in {"recipe", "pipeline"}:
        raise ValueError(f"unknown agent tool profile: {tool_profile!r}")
    system_prompt = str(system_prompt_append or "")
    max_retries = max(1, int(agent_cfg.get("max_retries", 4)))
    base_delay = float(agent_cfg.get("retry_base_delay", 10))
    timeout = float(agent_cfg.get("run_timeout_sec", 1800))
    resume_enabled = bool(agent_cfg.get("resume_on_retry", True))
    resume_id: str | None = None
    result = AgentRunResult(False, error="未发起任何 OpenCode 运行", error_type="NoAttempt")

    for attempt in range(1, max_retries + 1):
        used_resume = bool(resume_enabled and resume_id)
        current_prompt = (
            (_RECIPE_RESUME_NUDGE if phase.startswith(("plan_recipe", "repair_recipe")) else _RESUME_NUDGE)
            if used_resume
            else prompt
        )
        logger.info(
            "启动 OpenCode 运行（尝试 %d/%d，%s）：cwd=%s",
            attempt,
            max_retries,
            f"resume={resume_id}" if used_resume else "全新会话",
            cwd,
        )
        result = _run_once(
            current_prompt,
            cwd=cwd,
            agent_cfg=agent_cfg,
            tool_profile=tool_profile,
            system_prompt=system_prompt,
            resume=resume_id if used_resume else None,
            timeout=timeout,
        )
        if tool_log_dir:
            label = phase if attempt == 1 else f"{phase}#retry{attempt - 1}"
            record_tool_uses(tool_log_dir, result.tool_uses, phase=label)
        if result.session_id:
            resume_id = result.session_id
        if result.success:
            return result

        result.retryable = _is_retryable(result)
        if used_resume and not result.retryable:
            resume_id = None
            result.retryable = True
        if attempt < max_retries and result.retryable:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("OpenCode 第 %d/%d 次失败，%.0fs 后重试：%s", attempt, max_retries, delay, result.error)
            time.sleep(delay)
            continue
        return result
    return result


class OpenCodeAgentRuntime:
    """AgentRuntime implementation backed by ``opencode run``."""

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
        return run_opencode_agent(
            prompt,
            cwd=cwd,
            agent_cfg=agent_cfg,
            tool_log_dir=tool_log_dir,
            phase=phase,
            system_prompt_append=system_prompt_append,
            tool_profile=tool_profile,
        )
