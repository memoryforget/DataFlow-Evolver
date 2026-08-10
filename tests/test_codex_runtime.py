from __future__ import annotations

import dataflow_evolver.llm.codex_runtime as runtime
from dataflow_evolver.llm.agent_runtime import build_agent_runtime
from dataflow_evolver.llm.agent_sdk import AgentRunResult, _is_retryable
from dataflow_evolver.llm.codex_runtime import (
    CodexAgentRuntime,
    normalize_codex_usage,
    result_from_codex_cli,
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


def test_cli_jsonl_converts_text_tools_session_and_usage():
    stdout = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-cli"}',
            '{"type":"turn.started"}',
            (
                '{"type":"item.completed","item":{"id":"item-1",'
                '"type":"mcp_tool_call","server":"dataflow-operators",'
                '"tool":"list_operators"}}'
            ),
            (
                '{"type":"item.completed","item":{"id":"item-2",'
                '"type":"agent_message","text":"done"}}'
            ),
            (
                '{"type":"turn.completed","usage":{"input_tokens":12,'
                '"cached_input_tokens":2,"output_tokens":3}}'
            ),
        ]
    )

    result = result_from_codex_cli(stdout, "", exit_code=0)

    assert result.success
    assert result.session_id == "thread-cli"
    assert result.result_text == "done"
    assert result.tool_uses[0]["name"] == "dataflow-operators_list_operators"
    assert result.usage == {
        "input_tokens": 12,
        "cached_input_tokens": 2,
        "output_tokens": 3,
        "reasoning_tokens": 0,
        "total_tokens": 15,
    }


def test_cli_error_event_is_preserved_when_turn_also_completes():
    stdout = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-cli"}',
            (
                '{"type":"error","error":{"message":'
                '"bwrap: No permissions to create a new namespace"}}'
            ),
            (
                '{"type":"turn.completed","usage":'
                '{"input_tokens":12,"output_tokens":3}}'
            ),
        ]
    )

    result = result_from_codex_cli(stdout, "", exit_code=0)

    assert not result.success
    assert result.error_type == "CodexCLIError"
    assert "bwrap" in result.error


def test_cli_runtime_uses_configured_binary_provider_and_session_home(
    monkeypatch, tmp_path
):
    captured = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            return (
                "\n".join(
                    [
                        '{"type":"thread.started","thread_id":"thread-cli"}',
                        (
                            '{"type":"item.completed","item":'
                            '{"type":"agent_message","text":"done"}}'
                        ),
                        (
                            '{"type":"turn.completed","usage":'
                            '{"input_tokens":5,"output_tokens":1}}'
                        ),
                    ]
                ),
                "",
            )

    monkeypatch.setattr(runtime.subprocess, "Popen", FakeProcess)
    monkeypatch.setenv("DF_AGENT_API_KEY", "generic-key")
    cfg = Config(
        {
            "api_key_env": "DF_AGENT_API_KEY",
            "base_url": "https://codex-gateway.example/v1",
            "system_prompt_append": "developer guard",
            "codex": {
                "binary": "/opt/codex-0.118/codex",
                "model": "gpt-5.4",
                "effort": "xhigh",
                "model_provider": "openai-custom",
            },
        }
    )

    result = runtime._run_once(
        "write pipeline",
        cwd=str(tmp_path),
        agent_cfg=cfg,
        resume=None,
        timeout=30,
        session_env={"CODEX_HOME": str(tmp_path / "sessions/codex")},
    )

    assert result.success
    args = captured["args"]
    assert args[0] == "/opt/codex-0.118/codex"
    model_index = args.index("--model")
    assert ["--model", "gpt-5.4"] == args[model_index : model_index + 2]
    assert any(
        item == 'model_providers.openai-custom.base_url="https://codex-gateway.example/v1"'
        for item in args
    )
    assert "model_providers.openai-custom.supports_websockets=false" in args
    assert any(item.startswith("mcp_servers.dataflow-operators.command=") for item in args)
    assert args[-1].startswith("developer guard\n\nwrite pipeline")
    assert captured["kwargs"]["env"]["CODEX_API_KEY"] == "generic-key"
    assert captured["kwargs"]["env"]["OPENAI_API_KEY"] == "generic-key"
    assert captured["kwargs"]["env"]["CODEX_HOME"].endswith("sessions/codex")
    assert captured["kwargs"]["env"]["PATH"].split(runtime.os.pathsep)[0] == (
        "/opt/codex-0.118"
    )


def test_custom_base_url_uses_http_only_provider_when_name_is_omitted(monkeypatch):
    monkeypatch.setenv("DF_AGENT_API_KEY", "generic-key")
    cfg = Config(
        {
            "api_key_env": "DF_AGENT_API_KEY",
            "base_url": "https://gateway.example/v1",
            "codex": {},
        }
    )

    overrides, env, provider = runtime._codex_provider_overrides(cfg)

    assert provider == "dataflow-openai-compatible"
    assert env["CODEX_API_KEY"] == "generic-key"
    assert 'model_provider="dataflow-openai-compatible"' in overrides
    assert (
        "model_providers.dataflow-openai-compatible.supports_websockets=false"
        in overrides
    )


def test_cli_runtime_bypass_sandbox_requires_explicit_opt_in(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, args, **kwargs):
            captured["args"] = args

        def communicate(self, timeout=None):
            return (
                "\n".join(
                    [
                        '{"type":"thread.started","thread_id":"thread-cli"}',
                        '{"type":"turn.completed","usage":{"input_tokens":1}}',
                    ]
                ),
                "",
            )

    monkeypatch.setattr(runtime.subprocess, "Popen", FakeProcess)
    cfg = Config(
        {
            "codex": {
                "model": "gpt-5.4",
                "effort": "xhigh",
                "bypass_sandbox": True,
            }
        }
    )

    result = runtime._run_once(
        "write pipeline",
        cwd=str(tmp_path),
        agent_cfg=cfg,
        resume=None,
        timeout=30,
    )

    assert result.success
    args = captured["args"]
    assert "--dangerously-bypass-approvals-and-sandbox" in args
    assert "--full-auto" not in args
    assert 'sandbox_mode="workspace-write"' not in args
    assert 'sandbox_workspace_write.network_access=false' not in args
    assert ".agents/skills/dataflow-dev/templates/operator_api_reference.md" in args[-1]
    assert ".claude/skills/dataflow-dev" not in args[-1]


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


def test_upstream_stream_disconnect_is_retryable():
    result = AgentRunResult(
        False,
        error=(
            "Reconnecting... 1/5 (stream disconnected before completion: "
            "Upstream request failed)"
        ),
        error_type="CodexCLIError",
    )

    assert _is_retryable(result)
