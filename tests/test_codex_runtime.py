from __future__ import annotations

import asyncio
from types import SimpleNamespace

import dataflow_evolver.llm.codex_runtime as runtime
from dataflow_evolver.llm.agent_runtime import build_agent_runtime
from dataflow_evolver.llm.agent_sdk import AgentRunResult
from dataflow_evolver.llm.codex_runtime import (
    CodexAgentRuntime,
    build_codex_thread_config,
    normalize_codex_usage,
    result_from_codex_turn,
    run_codex_agent,
)
from dataflow_evolver.utils.config import Config


def _cfg(**extra):
    raw = {
        "backend": "codex",
        "max_retries": 1,
        "retry_base_delay": 0,
        "resume_on_retry": True,
        "codex": {"model": "gpt-5.4", "effort": "high"},
    }
    raw.update(extra)
    return Config(raw)


def test_runtime_factory_builds_codex_backend():
    assert isinstance(build_agent_runtime(_cfg()), CodexAgentRuntime)


def test_codex_thread_config_is_local_and_operator_only(tmp_path):
    config = build_codex_thread_config(str(tmp_path), 25000)
    assert config["sandbox_workspace_write"]["network_access"] is False
    assert config["web_search"] == "disabled"
    assert set(config["mcp_servers"]) == {"dataflow-operators"}
    server = config["mcp_servers"]["dataflow-operators"]
    assert server["required"] is True
    assert server["default_tools_approval_mode"] == "approve"
    assert set(server["enabled_tools"]) == {
        "list_operator_categories",
        "recommend_operator_categories",
        "list_operators",
        "get_operator_detail",
    }


def test_usage_prefers_last_turn_over_cumulative_total():
    usage = normalize_codex_usage(
        {
            "total": {"inputTokens": 1000, "outputTokens": 200},
            "last": {
                "inputTokens": 100,
                "cachedInputTokens": 20,
                "outputTokens": 30,
                "reasoningTokens": 10,
                "totalTokens": 130,
            },
        }
    )
    assert usage == {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "output_tokens": 30,
        "reasoning_tokens": 10,
        "total_tokens": 130,
    }


def test_completed_turn_converts_text_tools_and_usage():
    turn = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        error=None,
        final_response="done",
        items=[
            {"type": "agent_message", "text": "working"},
            {
                "type": "mcp_tool_call",
                "server": "dataflow-operators",
                "tool": "list_operators",
            },
        ],
        usage={"last": {"input_tokens": 12, "output_tokens": 3}},
    )
    result = result_from_codex_turn(turn, session_id="thread-1")
    assert result.success
    assert result.session_id == "thread-1"
    assert result.assistant_texts == ["working", "done"]
    assert result.tool_uses[0]["name"] == "dataflow-operators_list_operators"
    assert result.usage["total_tokens"] == 15


def test_retry_resumes_codex_thread(monkeypatch, tmp_path):
    calls = []
    results = [
        AgentRunResult(
            False,
            error="429 rate limit",
            error_type="CodexTurnError",
            session_id="thread-retry",
        ),
        AgentRunResult(True, session_id="thread-retry"),
    ]

    def fake_once(prompt, **kwargs):
        calls.append((prompt, kwargs["resume"]))
        return results.pop(0)

    monkeypatch.setattr(runtime, "_run_once", fake_once)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    result = run_codex_agent(
        "original",
        cwd=str(tmp_path),
        agent_cfg=_cfg(max_retries=2),
    )
    assert result.success
    assert calls[0] == ("original", None)
    assert calls[1][1] == "thread-retry"
    assert calls[1][0] != "original"


def test_sdk_boundary_maps_generic_endpoint_and_explicit_skill(monkeypatch, tmp_path):
    captured = {}
    skill_path = tmp_path / ".agents/skills/dataflow-dev/SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\nname: dataflow-dev\n---\n", encoding="utf-8")

    class FakeThread:
        id = "thread-sdk"

        async def run(self, turn_input, **kwargs):
            captured["turn_input"] = turn_input
            captured["turn_kwargs"] = kwargs
            return SimpleNamespace(
                status=SimpleNamespace(value="completed"),
                error=None,
                final_response="done",
                items=[],
                usage={"last": {"input_tokens": 10, "output_tokens": 2}},
            )

    class FakeCodex:
        def __init__(self, config):
            captured["client_config"] = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def login_api_key(self, key):
            captured["login_key"] = key

        async def thread_start(self, **kwargs):
            captured["thread_kwargs"] = kwargs
            return FakeThread()

    class FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setenv("DF_AGENT_API_KEY", "generic-key")
    monkeypatch.setattr(
        runtime,
        "_sdk_components",
        lambda: {
            "ApprovalMode": SimpleNamespace(deny_all="deny-all"),
            "AsyncCodex": FakeCodex,
            "CodexConfig": FakeConfig,
            "Sandbox": SimpleNamespace(workspace_write="workspace-write"),
            "SkillInput": lambda **kwargs: ("skill", kwargs),
            "TextInput": lambda **kwargs: ("text", kwargs),
            "is_retryable_error": lambda _: False,
        },
    )
    cfg = Config(
        {
            "api_key_env": "DF_AGENT_API_KEY",
            "base_url": "https://codex-gateway.example/v1",
            "codex": {
                "model": "gpt-5.4",
                "effort": "high",
                "model_provider": "openai-custom",
            },
        }
    )

    result = asyncio.run(
        runtime._run_once_async(
            "write pipeline",
            cwd=str(tmp_path),
            agent_cfg=cfg,
            resume=None,
            timeout=30,
            session_env={"CODEX_HOME": str(tmp_path / "sessions/codex")},
        )
    )

    assert result.success
    assert captured["login_key"] == "generic-key"
    client = captured["client_config"]
    assert client.env["CODEX_API_KEY"] == "generic-key"
    assert client.env["CODEX_HOME"].endswith("sessions/codex")
    assert client.config_overrides == (
        'model_providers.openai-custom.name="openai-custom"',
        'model_providers.openai-custom.base_url="https://codex-gateway.example/v1"',
        'model_providers.openai-custom.wire_api="responses"',
        'model_providers.openai-custom.env_key="CODEX_API_KEY"',
    )
    thread = captured["thread_kwargs"]
    assert thread["approval_mode"] == "deny-all"
    assert thread["sandbox"] == "workspace-write"
    assert thread["model_provider"] == "openai-custom"
    assert set(thread["config"]["mcp_servers"]) == {"dataflow-operators"}
    assert captured["turn_input"][0][0] == "skill"
    assert captured["turn_input"][0][1]["path"] == str(skill_path.resolve())
