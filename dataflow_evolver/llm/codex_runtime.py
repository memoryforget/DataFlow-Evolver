"""OpenAI Codex Python SDK runtime for PipelineAgent."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dataflow_evolver.llm.agent_credentials import (
    backend_config,
    backend_value,
    resolve_agent_endpoint,
)
from dataflow_evolver.llm.agent_sdk import (
    AgentRunResult,
    _DEFAULT_SYSTEM_PROMPT_APPEND,
    _RESUME_NUDGE,
    _agent_env,
    _is_retryable,
    _record_tool_uses,
    _truncate_arg,
)
from dataflow_evolver.llm.session_state import (
    finalize_agent_session,
    prepare_agent_session_store,
    record_codex_turn_usage,
)
from dataflow_evolver.utils.config import Config
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("llm.codex_runtime")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPERATOR_TOOLS = (
    "list_operator_categories",
    "recommend_operator_categories",
    "list_operators",
    "get_operator_detail",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True, mode="json")
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_codex_usage(value: Any) -> dict[str, int] | None:
    """Normalize the SDK's per-turn ``usage.last`` shape without double counting."""
    raw = _jsonable(value)
    if not isinstance(raw, dict):
        return None
    selected = _pick(raw, "last", "last_usage", "lastUsage")
    if not isinstance(selected, dict):
        selected = _pick(raw, "total", "total_usage", "totalUsage")
    if not isinstance(selected, dict):
        selected = raw
    values = {
        "input_tokens": int(_pick(selected, "input_tokens", "inputTokens") or 0),
        "cached_input_tokens": int(
            _pick(selected, "cached_input_tokens", "cachedInputTokens") or 0
        ),
        "output_tokens": int(_pick(selected, "output_tokens", "outputTokens") or 0),
        "reasoning_tokens": int(
            _pick(
                selected,
                "reasoning_tokens",
                "reasoningTokens",
                "reasoning_output_tokens",
                "reasoningOutputTokens",
            )
            or 0
        ),
        "total_tokens": int(_pick(selected, "total_tokens", "totalTokens") or 0),
    }
    if not values["total_tokens"]:
        values["total_tokens"] = (
            values["input_tokens"] + values["output_tokens"]
        )
    return values if any(values.values()) else None


def _item_payload(item: Any) -> dict[str, Any]:
    payload = _jsonable(item)
    return payload if isinstance(payload, dict) else {"value": payload}


def _tool_trace(items: list[Any]) -> list[dict[str, Any]]:
    uses: list[dict[str, Any]] = []
    for item in items:
        payload = _item_payload(item)
        kind = str(payload.get("type") or payload.get("kind") or "").lower()
        name = ""
        if "mcp" in kind:
            server = str(_pick(payload, "server", "server_name", "serverName") or "mcp")
            tool = str(_pick(payload, "tool", "tool_name", "toolName") or "tool")
            name = f"{server}_{tool}"
        elif "command" in kind:
            name = "shell_command"
        elif "filechange" in kind or "file_change" in kind:
            name = "apply_patch"
        elif "tool" in kind:
            name = str(_pick(payload, "name", "tool", "tool_name", "toolName") or kind)
        if name:
            uses.append({"turn": 1, "name": name, "input": _truncate_arg(payload)})
    return uses


def _assistant_texts(items: list[Any], final_response: str) -> list[str]:
    texts: list[str] = []
    for item in items:
        payload = _item_payload(item)
        kind = str(payload.get("type") or payload.get("kind") or "").lower()
        if "agentmessage" not in kind and "agent_message" not in kind:
            continue
        text = str(_pick(payload, "text", "content", "message") or "").strip()
        if text:
            texts.append(text)
    if final_response and final_response not in texts:
        texts.append(final_response)
    return texts


def result_from_codex_turn(turn: Any, *, session_id: str) -> AgentRunResult:
    items = list(getattr(turn, "items", None) or [])
    final_response = str(getattr(turn, "final_response", None) or "")
    status_value = getattr(getattr(turn, "status", None), "value", None)
    status = str(status_value or getattr(turn, "status", "")).lower()
    error_obj = getattr(turn, "error", None)
    error_payload = _jsonable(error_obj)
    if isinstance(error_payload, dict):
        error = str(_pick(error_payload, "message", "details") or error_payload)
    else:
        error = str(error_payload or "")
    success = status in {"completed", "success", "succeeded"} and not error
    return AgentRunResult(
        success=success,
        result_text=final_response,
        assistant_texts=_assistant_texts(items, final_response),
        error=error or (None if success else f"Codex turn ended with status {status or 'unknown'}"),
        error_type=None if success else "CodexTurnError",
        session_id=session_id,
        tool_uses=_tool_trace(items),
        usage=normalize_codex_usage(getattr(turn, "usage", None)),
    )


def build_codex_thread_config(cwd: str, timeout_ms: int) -> dict[str, Any]:
    pythonpath = str(_PROJECT_ROOT)
    inherited = os.environ.get("PYTHONPATH", "")
    if inherited:
        pythonpath += os.pathsep + inherited
    return {
        "sandbox_workspace_write": {"network_access": False},
        "web_search": "disabled",
        "mcp_servers": {
            "dataflow-operators": {
                "command": sys.executable,
                "args": ["-m", "dataflow_evolver.llm.mcp_stdio"],
                "cwd": str(_PROJECT_ROOT),
                "env": {"DF_WORKSPACE": cwd, "PYTHONPATH": pythonpath},
                "enabled": True,
                "required": True,
                "startup_timeout_ms": timeout_ms,
                "tool_timeout_sec": max(timeout_ms / 1000.0, 1.0),
                "default_tools_approval_mode": "approve",
                "enabled_tools": list(_OPERATOR_TOOLS),
            }
        },
    }


def _sdk_components() -> dict[str, Any]:
    from openai_codex import (  # type: ignore[import-not-found]
        ApprovalMode,
        AsyncCodex,
        CodexConfig,
        Sandbox,
        SkillInput,
        TextInput,
        is_retryable_error,
    )

    return {
        "ApprovalMode": ApprovalMode,
        "AsyncCodex": AsyncCodex,
        "CodexConfig": CodexConfig,
        "Sandbox": Sandbox,
        "SkillInput": SkillInput,
        "TextInput": TextInput,
        "is_retryable_error": is_retryable_error,
    }


async def _run_once_async(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    resume: str | None,
    timeout: float,
    session_env: dict[str, str] | None,
) -> AgentRunResult:
    session_id = resume
    try:
        sdk = _sdk_components()
    except (ImportError, ModuleNotFoundError) as exc:
        return AgentRunResult(False, error=str(exc), error_type="SDKNotInstalledError")

    cfg = backend_config(agent_cfg, "codex")
    endpoint = resolve_agent_endpoint(agent_cfg, "codex", provider="openai")
    env = os.environ.copy()
    env.update(_agent_env(agent_cfg))
    if session_env:
        env.update(session_env)
    if endpoint.api_key:
        env["CODEX_API_KEY"] = endpoint.api_key

    provider = str(cfg.get("model_provider", "") or "").strip() or None
    overrides: list[str] = []
    if provider and endpoint.base_url:
        provider_key = f"model_providers.{provider}"
        overrides.extend(
            [
                f"{provider_key}.name={json.dumps(provider)}",
                f"{provider_key}.base_url={json.dumps(endpoint.base_url)}",
                f'{provider_key}.wire_api="responses"',
                f'{provider_key}.env_key="CODEX_API_KEY"',
            ]
        )
    elif endpoint.base_url:
        overrides.append(f"openai_base_url={json.dumps(endpoint.base_url)}")
    binary = str(cfg.get("binary", "") or "").strip() or None
    client_config = sdk["CodexConfig"](
        codex_bin=binary,
        config_overrides=tuple(overrides),
        cwd=cwd,
        env=env,
        client_name="dataflow_evolver",
        client_title="DataFlow Evolver PipelineAgent",
    )

    model = str(backend_value(agent_cfg, "codex", "model", default="gpt-5.4"))
    effort = str(backend_value(agent_cfg, "codex", "effort", default="high") or "high")
    timeout_ms = int(cfg.get("mcp_timeout_ms", 30000) or 30000)
    thread_config = build_codex_thread_config(cwd, timeout_ms)
    developer_instructions = str(
        agent_cfg.get("system_prompt_append", _DEFAULT_SYSTEM_PROMPT_APPEND)
    )

    try:
        async with sdk["AsyncCodex"](client_config) as codex:
            if endpoint.api_key:
                await codex.login_api_key(endpoint.api_key)
            thread_kwargs = {
                "approval_mode": sdk["ApprovalMode"].deny_all,
                "config": thread_config,
                "cwd": cwd,
                "developer_instructions": developer_instructions,
                "model": model,
                "model_provider": provider,
                "sandbox": sdk["Sandbox"].workspace_write,
            }
            if resume:
                thread = await codex.thread_resume(resume, **thread_kwargs)
            else:
                thread = await codex.thread_start(ephemeral=False, **thread_kwargs)
            session_id = str(thread.id)
            skill_path = Path(cwd) / ".agents/skills/dataflow-dev/SKILL.md"
            turn_input: Any = prompt
            if skill_path.is_file():
                turn_input = [
                    sdk["SkillInput"](name="dataflow-dev", path=str(skill_path.resolve())),
                    sdk["TextInput"](text=prompt),
                ]
            turn = await asyncio.wait_for(
                thread.run(
                    turn_input,
                    approval_mode=sdk["ApprovalMode"].deny_all,
                    cwd=cwd,
                    effort=effort,
                    model=model,
                    sandbox=sdk["Sandbox"].workspace_write,
                ),
                timeout=timeout if timeout > 0 else None,
            )
            return result_from_codex_turn(turn, session_id=session_id)
    except (asyncio.TimeoutError, TimeoutError):
        return AgentRunResult(
            False,
            error=f"Codex SDK 运行超过 {timeout:.0f}s 超时",
            error_type="Timeout",
            session_id=session_id,
        )
    except Exception as exc:  # SDK exception hierarchy remains provider/runtime-specific.
        retryable = False
        try:
            retryable = bool(sdk["is_retryable_error"](exc))
        except Exception:
            pass
        return AgentRunResult(
            False,
            error=str(exc),
            error_type=type(exc).__name__,
            retryable=retryable,
            session_id=session_id,
        )


def _run_once(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    resume: str | None,
    timeout: float,
    session_env: dict[str, str] | None = None,
) -> AgentRunResult:
    return asyncio.run(
        _run_once_async(
            prompt,
            cwd=cwd,
            agent_cfg=agent_cfg,
            resume=resume,
            timeout=timeout,
            session_env=session_env,
        )
    )


def run_codex_agent(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    tool_log_dir: str | None = None,
    phase: str = "",
) -> AgentRunResult:
    """Run Codex with the same retry/resume contract as the other backends."""
    max_retries = max(1, int(agent_cfg.get("max_retries", 4)))
    base_delay = float(agent_cfg.get("retry_base_delay", 10))
    timeout = float(agent_cfg.get("run_timeout_sec", 1800))
    resume_enabled = bool(agent_cfg.get("resume_on_retry", True))
    session_store = prepare_agent_session_store(
        agent_cfg,
        backend="codex",
        cwd=cwd,
        tool_log_dir=tool_log_dir,
    )
    session_env = session_store.environment() if session_store is not None else None
    resume_id: str | None = None
    result = AgentRunResult(False, error="未发起任何 Codex 运行", error_type="NoAttempt")

    for attempt in range(1, max_retries + 1):
        used_resume = bool(resume_enabled and resume_id)
        current_prompt = _RESUME_NUDGE if used_resume else prompt
        logger.info(
            "启动 Codex SDK（尝试 %d/%d，%s）：cwd=%s",
            attempt,
            max_retries,
            f"resume={resume_id}" if used_resume else "全新会话",
            cwd,
        )
        result = _run_once(
            current_prompt,
            cwd=cwd,
            agent_cfg=agent_cfg,
            resume=resume_id if used_resume else None,
            timeout=timeout,
            session_env=session_env,
        )
        label = phase if attempt == 1 else f"{phase}#retry{attempt - 1}"
        if tool_log_dir:
            _record_tool_uses(tool_log_dir, result.tool_uses, phase=label)
        if session_store is not None and result.usage:
            record_codex_turn_usage(
                session_store.root,
                phase=label,
                session_id=result.session_id,
                usage=result.usage,
            )
        if result.session_id:
            resume_id = result.session_id
        if result.success:
            return finalize_agent_session(session_store, phase=phase, result=result)

        result.retryable = result.retryable or _is_retryable(result)
        if used_resume and not result.retryable:
            logger.warning("Codex resume 失败，下一次改用新会话：%s", result.error)
            resume_id = None
            result.retryable = True
        if attempt < max_retries and result.retryable:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Codex 第 %d/%d 次失败，%.0fs 后重试：%s",
                attempt,
                max_retries,
                delay,
                result.error,
            )
            time.sleep(delay)
            continue
        logger.error(
            "Codex 运行失败 [%s]（retryable=%s）：%s",
            result.error_type,
            result.retryable,
            result.error,
        )
        return finalize_agent_session(session_store, phase=phase, result=result)
    return finalize_agent_session(session_store, phase=phase, result=result)


class CodexAgentRuntime:
    """AgentRuntime implementation backed by the official Python Codex SDK."""

    def run(
        self,
        prompt: str,
        *,
        cwd: str,
        agent_cfg: Config,
        tool_log_dir: str | None,
        phase: str,
    ) -> AgentRunResult:
        return run_codex_agent(
            prompt,
            cwd=cwd,
            agent_cfg=agent_cfg,
            tool_log_dir=tool_log_dir,
            phase=phase,
        )
