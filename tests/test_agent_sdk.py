import json

import dataflow_evolver.llm.agent_sdk as sdk
from dataflow_evolver.llm.agent_sdk import AgentRunResult
from dataflow_evolver.utils.config import Config


def test_tool_arguments_are_truncated():
    value = sdk._truncate_arg({"content": "x" * 1000})
    assert len(value["content"]) < 700


def test_tool_trace_appends_by_phase(tmp_path):
    sdk._record_tool_uses(tmp_path, [{"name": "Read", "input": {"file": "a"}}], "generate")
    sdk._record_tool_uses(tmp_path, [{"name": "Write", "input": {"file": "b"}}], "retry")
    rows = [json.loads(line) for line in (tmp_path / "agent_tools.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["generate", "retry"]


def test_failed_attempt_tool_trace_is_not_lost(tmp_path, monkeypatch):
    results = [
        AgentRunResult(False, error="429", error_type="ProcessError", tool_uses=[{"name": "Read"}]),
        AgentRunResult(True, tool_uses=[{"name": "Write"}]),
    ]

    monkeypatch.setattr(sdk, "_build_options", lambda *args, **kwargs: object())

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
