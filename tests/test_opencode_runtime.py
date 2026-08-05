from __future__ import annotations

import json
import os
import subprocess
import sys

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from dataflow_evolver.llm.agent_runtime import build_agent_runtime
from dataflow_evolver.llm.opencode_runtime import (
    OpenCodeAgentRuntime,
    build_opencode_config,
    parse_opencode_events,
    run_opencode_agent,
)
from dataflow_evolver.utils.config import Config


def _cfg(**extra):
    raw = {
        "runtime": "opencode",
        "model": "claude-opus-4-8",
        "effort": "max",
        "external_data": True,
        "discovery": True,
        "max_turns": 17,
        "max_retries": 1,
        "opencode": {
            "binary": "opencode",
            "provider": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": "https://gateway.example/",
        },
    }
    raw.update(extra)
    return Config(raw)


def test_runtime_factory_builds_opencode_adapter():
    assert isinstance(build_agent_runtime(_cfg()), OpenCodeAgentRuntime)


def test_inline_configs_isolate_recipe_and_pipeline_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-must-not-be-serialized")
    recipe, recipe_name, model = build_opencode_config(
        _cfg(), cwd=str(tmp_path), tool_profile="recipe", system_prompt="recipe rules"
    )
    pipeline, pipeline_name, _ = build_opencode_config(
        _cfg(), cwd=str(tmp_path), tool_profile="pipeline", system_prompt="pipeline rules"
    )

    assert recipe_name == "df-recipe"
    assert pipeline_name == "df-pipeline"
    assert model == "anthropic/claude-opus-4-8"
    assert set(recipe["mcp"]) == {"dataset-search", "data-discovery"}
    assert set(pipeline["mcp"]) == {"dataflow-operators"}
    assert pipeline["mcp"]["dataflow-operators"]["timeout"] == 30000
    assert recipe["agent"][recipe_name]["permission"]["skill"] == "deny"
    assert pipeline["agent"][pipeline_name]["permission"]["skill"]["dataflow-dev"] == "allow"
    serialized = json.dumps(recipe)
    assert "secret-must-not-be-serialized" not in serialized
    assert "{env:ANTHROPIC_API_KEY}" in serialized


def test_parse_opencode_ndjson_records_text_session_and_tools():
    stdout = "\n".join(
        [
            json.dumps({"type": "step_start", "sessionID": "ses_1", "part": {"type": "step-start"}}),
            json.dumps(
                {
                    "type": "tool_use",
                    "sessionID": "ses_1",
                    "part": {
                        "tool": "dataset-search_search_datasets",
                        "state": {"status": "completed", "input": {"query": "math"}},
                    },
                }
            ),
            json.dumps(
                {"type": "text", "sessionID": "ses_1", "part": {"text": "recipe written"}}
            ),
        ]
    )
    result = parse_opencode_events(stdout)
    assert result.success
    assert result.session_id == "ses_1"
    assert result.result_text == "recipe written"
    assert result.tool_uses == [
        {"turn": 1, "name": "dataset-search_search_datasets", "input": {"query": "math"}}
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


def test_tool_error_can_be_recovered_by_agent():
    events = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_use",
                    "sessionID": "ses_recover",
                    "part": {
                        "tool": "bash",
                        "state": {"status": "error", "input": {"command": "bad"}, "error": "failed"},
                    },
                }
            ),
            json.dumps(
                {"type": "text", "sessionID": "ses_recover", "part": {"text": "fixed it"}}
            ),
        ]
    )
    result = parse_opencode_events(events)
    assert result.success
    assert result.result_text == "fixed it"
    assert result.tool_uses[0]["name"] == "bash"


def test_headless_run_uses_stdin_and_inline_config(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        stdout = json.dumps(
            {"type": "text", "sessionID": "ses_ok", "part": {"text": "done"}}
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_opencode_agent(
        "write the recipe",
        cwd=str(tmp_path),
        agent_cfg=_cfg(),
        phase="plan_recipe",
        tool_profile="recipe",
        system_prompt_append="recipe system",
    )
    assert result.success
    assert result.result_text == "done"
    assert captured["input"] == "write the recipe"
    assert captured["command"][:3] == ["opencode", "run", "--format"]
    inline = json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])
    assert inline["agent"]["df-recipe"]["prompt"] == "recipe system"
    assert set(inline["mcp"]) == {"dataset-search", "data-discovery"}


def test_stdio_bridge_exposes_existing_discovery_tools():
    async def list_names():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "dataflow_evolver.llm.mcp_stdio", "discovery"],
            env=os.environ.copy(),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return {tool.name for tool in result.tools}

    assert anyio.run(list_names) == {"search_web", "search_github", "search_papers"}
