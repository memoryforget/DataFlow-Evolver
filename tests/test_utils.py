import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.utils.config import load_config
from dataflow_evolver.utils.parsing import extract_json, extract_code
from dataflow_evolver.main import _normalize_pipeline_api_url


def test_config_loads_and_expands():
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    )
    assert int(cfg.loop.max_iterations) == 4
    assert cfg.agent.backend == "claude"
    assert cfg.agent.setting_sources == ["project"]
    assert cfg.agent.claude.permission_mode == "bypassPermissions"
    assert cfg.agent.claude.env.IS_SANDBOX == "1"
    assert cfg.agent.opencode.permission == "allow"
    assert cfg.agent.codex.bypass_sandbox is True
    # env default expansion
    assert cfg.llm.api.model_name == "gpt-5.5"
    assert int(cfg.llm.api.max_workers) == 10
    assert cfg.loop.get("missing", 42) == 42


def test_pipeline_api_url_accepts_base_or_full_chat_endpoint():
    full = "https://api.example/v1/chat/completions"
    assert _normalize_pipeline_api_url("https://api.example/v1") == full
    assert _normalize_pipeline_api_url(full + "/") == full
    assert _normalize_pipeline_api_url("https://api.example/custom") == "https://api.example/custom"


def test_extract_json_fenced():
    text = "blah ```json\n{\"a\": 1, \"b\": [2, 3]}\n``` end"
    assert extract_json(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_fenced_nested_object():
    """围栏内嵌套对象：旧的非贪婪 \\{.*?\\} 会截断到第一个 }，新实现须正确解析（review M3）。"""
    text = "前言 ```json\n{\"target_schema\": {\"instruction\": \"str\", \"answer\": \"str\"}}\n``` 后记"
    assert extract_json(text) == {"target_schema": {"instruction": "str", "answer": "str"}}


def test_extract_json_balanced_no_fence():
    text = "noise before {\"x\": {\"y\": 1}} noise after"
    assert extract_json(text) == {"x": {"y": 1}}


def test_extract_json_handles_braces_in_strings():
    text = '{"k": "a } b { c"}'
    assert extract_json(text) == {"k": "a } b { c"}


def test_extract_code_fenced():
    text = "text ```python\nx = 1\nprint(x)\n``` more"
    assert extract_code(text) == "x = 1\nprint(x)"


def test_extract_json_none_on_garbage():
    assert extract_json("no structure here") is None
