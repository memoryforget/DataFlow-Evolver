import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.utils.config import load_config
from dataflow_evolver.utils.parsing import extract_json, extract_code


def test_config_loads_and_expands():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg.search.exploration_constant == 1.414
    assert cfg.agent_sdk.skills == ["dataflow-dev"]
    # env default expansion
    assert cfg.llm.api.model_name  # ${DF_MODEL:gpt-4o} -> gpt-4o by default
    assert cfg.search.get("missing", 42) == 42


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
