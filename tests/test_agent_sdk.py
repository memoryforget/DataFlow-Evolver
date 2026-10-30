import json

import pytest

import dataflow_evolver.llm.agent_sdk as sdk
from dataflow_evolver.llm.agent_sdk import AgentRunResult
from dataflow_evolver.utils.config import Config


def test_tool_arguments_are_truncated():
    value = sdk._truncate_arg({"content": "x" * 1000})
    assert len(value["content"]) < 700


def test_claude_permission_and_root_acknowledgement_are_backend_scoped(tmp_path):
    pytest.importorskip("claude_agent_sdk")
    options = sdk._build_options(
        Config(
            {
                "claude": {
                    "permission_mode": "bypassPermissions",
                    "env": {"IS_SANDBOX": "1"},
                }
            }
        ),
        str(tmp_path),
    )

    assert options.permission_mode == "bypassPermissions"
    assert options.env["IS_SANDBOX"] == "1"


def test_high_demand_is_retryable():
    result = AgentRunResult(
        False,
        error="We're currently experiencing high demand, which may cause temporary errors.",
        error_type="RuntimeError",
    )
    assert sdk._is_retryable(result)


def test_tool_trace_appends_by_phase(tmp_path):
    sdk._record_tool_uses(tmp_path, [{"name": "Read", "input": {"file": "a"}}], "generate")
    sdk._record_tool_uses(tmp_path, [{"name": "Write", "input": {"file": "b"}}], "retry")
    rows = [json.loads(line) for line in (tmp_path / "agent_tools.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["generate", "retry"]


def test_failed_attempt_tool_trace_is_not_lost(tmp_path, monkeypatch):
    captured_session_envs = []
    results = [
        AgentRunResult(False, error="429", error_type="ProcessError", tool_uses=[{"name": "Read"}]),
        AgentRunResult(True, tool_uses=[{"name": "Write"}]),
    ]

    def fake_build_options(*args, **kwargs):
        captured_session_envs.append(kwargs.get("session_env"))
        return object()

    monkeypatch.setattr(sdk, "_build_options", fake_build_options)

    async def fake_run(prompt, options, timeout):
        return results.pop(0)

    monkeypatch.setattr(sdk, "_run_with_timeout", fake_run)
    monkeypatch.setattr(sdk.time, "sleep", lambda seconds: None)
    result = sdk.run_pipeline_agent(
        "prompt",
        str(tmp_path),
        Config({"max_retries": 2, "retry_base_delay": 0, "resume_on_retry": False}),
        tool_log_dir=str(tmp_path),
        phase="generate",
    )
    assert result.success
    rows = [json.loads(line) for line in (tmp_path / "agent_tools.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["generate", "generate#retry1"]
    expected_root = (tmp_path / "sessions/claude").resolve()
    assert captured_session_envs[0]["HOME"] == str(expected_root / "home")
    assert captured_session_envs[0]["CLAUDE_CONFIG_DIR"] == str(expected_root)
    assert (tmp_path / "agent_sessions.jsonl").is_file()
    assert (tmp_path / "agent_token_usage.json").is_file()
