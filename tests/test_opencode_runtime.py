from __future__ import annotations

import json
import subprocess


import dataflow_evolver.llm.opencode_runtime as runtime
from dataflow_evolver.llm.agent_runtime import build_agent_runtime
from dataflow_evolver.llm.agent_sdk import AgentRunResult
from dataflow_evolver.llm.opencode_runtime import (
    OpenCodeAgentRuntime,
    build_opencode_config,
    parse_opencode_events,
    run_opencode_agent,
)
from dataflow_evolver.utils.config import Config


def _cfg(**extra):
    raw = {
        "backend": "opencode",
        "model": "claude-opus-4-8",
        "max_turns": 17,
        "max_retries": 1,
        "retry_base_delay": 0,
        "resume_on_retry": True,
        "opencode": {
            "binary": "opencode",
            "provider": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": "https://gateway.example/",
            "variant": "",
        },
    }
    raw.update(extra)
    return Config(raw)


def test_runtime_factory_builds_both_backends():
    assert isinstance(build_agent_runtime(_cfg()), OpenCodeAgentRuntime)
    assert type(build_agent_runtime(Config({"backend": "claude"}))).__name__ == "ClaudeAgentRuntime"


def test_unknown_runtime_is_rejected():
    try:
        build_agent_runtime(Config({"backend": "other"}))
    except ValueError as exc:
        assert "claude, opencode, or codex" in str(exc)
    else:
        raise AssertionError("unknown backend should fail")


def test_inline_config_uses_skill_without_dataflow_mcp(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-must-not-be-serialized")
    config, agent_name, model = build_opencode_config(_cfg(), cwd=str(tmp_path))

    assert agent_name == "dataflow-pipeline"
    assert model == "anthropic/claude-opus-4-8"
    assert "mcp" not in config
    assert config["agent"][agent_name]["permission"] == "allow"
    prompt = config["agent"][agent_name]["prompt"]
    assert "dataflow-evolver-pipeline" in prompt
    assert "current iteration artifacts" in prompt
    serialized = json.dumps(config)
    assert "secret-must-not-be-serialized" not in serialized
    assert "{env:ANTHROPIC_API_KEY}" in serialized


def test_invalid_permission_mode_is_rejected(tmp_path):
    cfg = _cfg(
        opencode={
            "provider": "anthropic",
            "model": "claude-opus-4-8",
            "permission": "bypassPermissions",
        }
    )
    try:
        build_opencode_config(cfg, cwd=str(tmp_path))
    except ValueError as exc:
        assert "allow, ask, or deny" in str(exc)
    else:
        raise AssertionError("invalid OpenCode permission must be rejected")


def test_provider_prefix_and_non_anthropic_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic-only.example/")
    cfg = _cfg(
        model="unused",
        opencode={
            "provider": "openai",
            "model": "gpt-5",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "",
        },
    )
    config, _, model = build_opencode_config(cfg, cwd=str(tmp_path))
    assert model == "openai/gpt-5"
    assert config["provider"]["openai"]["options"] == {
        "apiKey": "{env:OPENAI_API_KEY}"
    }


def test_parse_opencode_ndjson_records_text_session_and_tools():
    stdout = "\n".join(
        [
            json.dumps({"type": "step_start", "sessionID": "ses_1", "part": {}}),
            json.dumps(
                {
                    "type": "tool_use",
                    "sessionID": "ses_1",
                    "part": {
                        "tool": "example-server_example-tool",
                        "state": {"input": {"category": "reasoning"}},
                    },
                }
            ),
            json.dumps(
                {"type": "text", "sessionID": "ses_1", "part": {"text": "done"}}
            ),
        ]
    )
    result = parse_opencode_events(stdout)
    assert result.success
    assert result.session_id == "ses_1"
    assert result.result_text == "done"
    assert result.tool_uses == [
        {
            "turn": 1,
            "name": "example-server_example-tool",
            "input": {"category": "reasoning"},
        }
    ]


def test_error_event_is_a_failed_result():
    event = {
        "type": "error",
        "sessionID": "ses_bad",
        "error": {"name": "APIError", "data": {"message": "401 invalid key"}},
    }
    result = parse_opencode_events(json.dumps(event))
    assert not result.success
    assert result.error == "401 invalid key"
    assert result.error_type == "OpenCodeError"


def test_headless_run_uses_stdin_inline_config_and_cli_flags(monkeypatch, tmp_path):
    captured = {}
    iteration_dir = tmp_path / "workspace/runs/run/iteration_001"

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        stdout = json.dumps(
            {"type": "text", "sessionID": "ses_ok", "part": {"text": "done"}}
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_opencode_agent(
        "write the pipeline",
        cwd=str(tmp_path),
        agent_cfg=_cfg(),
        tool_log_dir=str(iteration_dir),
        phase="generate",
    )
    assert result.success
    assert captured["input"] == "write the pipeline"
    assert captured["command"][:3] == ["opencode", "run", "--format"]
    inline = json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])
    assert inline["agent"]["dataflow-pipeline"]["model"] == "anthropic/claude-opus-4-8"
    assert "mcp" not in inline
    session_dir = iteration_dir / "sessions/opencode"
    assert captured["env"]["HOME"] == str((session_dir / "home").resolve())
    assert captured["env"]["XDG_DATA_HOME"] == str(
        (session_dir / "data").resolve()
    )
    assert (iteration_dir / "agent_sessions.jsonl").is_file()
    assert (iteration_dir / "agent_token_usage.json").is_file()


def test_retry_resumes_open_session(monkeypatch, tmp_path):
    calls = []
    results = [
        AgentRunResult(
            False,
            error="429 rate limit",
            error_type="OpenCodeError",
            session_id="ses_retry",
        ),
        AgentRunResult(True, session_id="ses_retry"),
    ]

    def fake_once(prompt, **kwargs):
        calls.append((prompt, kwargs["resume"]))
        return results.pop(0)

    monkeypatch.setattr(runtime, "_run_once", fake_once)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    cfg = _cfg(max_retries=2)
    result = run_opencode_agent("original", cwd=str(tmp_path), agent_cfg=cfg)
    assert result.success
    assert calls[0] == ("original", None)
    assert calls[1][1] == "ses_retry"
    assert calls[1][0] != "original"
