from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from dataflow_evolver.main import _pipeline_runtime_contract, _resolve_pipeline_llm
from dataflow_evolver.utils.config import Config


ROOT = Path(__file__).resolve().parents[1]
SERVING_PATH = ROOT / "workspace/dataflow/serving/api_llm_serving_request.py"


def _serving_class():
    spec = importlib.util.spec_from_file_location(
        "dataflow.serving.api_llm_serving_request", SERVING_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.APILLMServing_request


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("PIPELINE_KEY", "EMPTY")


def test_reasoning_field_is_preserved_for_cot_distillation(api_key):
    serving = _serving_class()(
        api_url="http://teacher/v1/chat/completions",
        key_name_of_api_key="PIPELINE_KEY",
        model_name="teacher",
        reasoning_field="reasoning",
        reasoning_output_format="think",
        require_finish_reason_stop=True,
        require_reasoning_and_content=True,
    )

    result = serving.format_response(
        {
            "choices": [
                {
                    "message": {
                        "reasoning": "derive recurrence",
                        "content": "Final Answer: 123",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert result == "<think>\nderive recurrence\n</think>\n\nFinal Answer: 123"


@pytest.mark.parametrize(
    "finish_reason,reasoning,content",
    [
        ("length", "unfinished reasoning", None),
        ("stop", "", "Final Answer: 123"),
        ("stop", "reasoning", None),
    ],
)
def test_incomplete_reasoning_responses_are_retryable_failures(
    api_key, finish_reason, reasoning, content
):
    serving = _serving_class()(
        api_url="http://teacher/v1/chat/completions",
        key_name_of_api_key="PIPELINE_KEY",
        model_name="teacher",
        require_finish_reason_stop=True,
        require_reasoning_and_content=True,
    )

    assert (
        serving.format_response(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning": reasoning,
                            "content": content,
                        },
                        "finish_reason": finish_reason,
                    }
                ]
            }
        )
        is None
    )


def test_pipeline_runtime_contract_includes_serving_kwargs():
    cfg = Config(
        {
            "api": {
                "api_url": "https://review.example/v1/chat/completions",
                "model_name": "review",
                "api_key": "review-key",
                "max_workers": 10,
            },
            "pipeline_llm": {
                "reasoning": {
                    "api_url": "http://teacher:8000/v1",
                    "model_name": "teacher",
                    "api_key": "EMPTY",
                    "max_workers": 100,
                    "serving_kwargs": {
                        "max_tokens": 8192,
                        "chat_template_kwargs": {"enable_thinking": True},
                        "require_finish_reason_stop": True,
                    },
                }
            },
        }
    )

    resolved = _resolve_pipeline_llm(cfg)
    prompt, env = _pipeline_runtime_contract(resolved)

    assert resolved["reasoning"]["api_url"].endswith("/v1/chat/completions")
    assert resolved["reasoning"]["max_workers"] == 100
    assert resolved["reasoning"]["serving_kwargs"]["max_tokens"] == 8192
    assert '"enable_thinking": true' in prompt
    assert "逐项作为构造函数关键字参数传入" in prompt
    assert env["DF_PIPELINE_API_KEY_REASONING"] == "EMPTY"
