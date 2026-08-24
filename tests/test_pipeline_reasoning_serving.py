from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests
from dataflow.core import LLMServingABC
from dataflow.serving.api_llm_serving_request import APILLMServing_request

from dataflow_evolver.compat.openai_serving import (
    install_api_llm_reasoning_compat,
    normalize_streaming_chat_response,
)
from dataflow_evolver.main import _pipeline_runtime_contract


def _native_serving(enable_thinking=False):
    serving = APILLMServing_request.__new__(APILLMServing_request)
    serving.configs = {"enable_thinking": enable_thinking}
    return serving


def _request(payload):
    return SimpleNamespace(
        body=json.dumps(payload).encode("utf-8"),
        url="http://gateway/v1/chat/completions",
    )


def _sse_response(chunks):
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/event-stream"
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    response._content = ("\n\n".join(lines) + "\n\n").encode("utf-8")
    return response


def test_generated_pipelines_use_native_dataflow_serving():
    assert issubclass(APILLMServing_request, LLMServingABC)


def test_reasoning_alias_uses_native_dataflow_format_without_mutating_payload():
    install_api_llm_reasoning_compat()
    install_api_llm_reasoning_compat()
    payload = {
        "choices": [
            {"message": {"reasoning": "derive", "content": "answer"}}
        ]
    }

    result = _native_serving(enable_thinking=True).format_response(payload)

    assert result == "<think>derive</think>\n<answer>answer</answer>"
    assert "reasoning_content" not in payload["choices"][0]["message"]


def test_enable_thinking_false_returns_content_only():
    install_api_llm_reasoning_compat()
    payload = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "hidden derivation",
                    "content": "0.85",
                }
            }
        ]
    }

    assert _native_serving(enable_thinking=False).format_response(payload) == "0.85"


def test_enable_thinking_defaults_to_content_only():
    install_api_llm_reasoning_compat()
    serving = APILLMServing_request.__new__(APILLMServing_request)
    serving.configs = {}
    payload = {
        "choices": [
            {"message": {"reasoning_content": "hidden", "content": "pass"}}
        ]
    }

    assert serving.format_response(payload) == "pass"


def test_enable_thinking_true_has_stable_wrappers_without_reasoning():
    install_api_llm_reasoning_compat()
    payload = {"choices": [{"message": {"content": "answer"}}]}

    assert _native_serving(enable_thinking=True).format_response(payload) == (
        "<think></think>\n<answer>answer</answer>"
    )


def test_enable_thinking_must_be_a_bool():
    install_api_llm_reasoning_compat()
    payload = {"choices": [{"message": {"content": "answer"}}]}

    with pytest.raises(TypeError, match="enable_thinking must be a bool"):
        _native_serving(enable_thinking="false").format_response(payload)


def test_streaming_sse_is_normalized_for_native_dataflow_and_usage_tracking():
    chunks = [
        {
            "id": "chat-1",
            "model": "qwen3.7-plus",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "reasoning_content": "step "}}
            ],
        },
        {
            "id": "chat-1",
            "model": "qwen3.7-plus",
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "two"}}
            ],
        },
        {
            "id": "chat-1",
            "model": "qwen3.7-plus",
            "choices": [{"index": 0, "delta": {"content": "answer"}}],
        },
        {
            "id": "chat-1",
            "model": "qwen3.7-plus",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19,
            },
        },
    ]
    response = _sse_response(chunks)

    normalized = normalize_streaming_chat_response(
        _request(
            {
                "model": "qwen3.7-plus",
                "messages": [{"role": "user", "content": "private"}],
                "stream": True,
            }
        ),
        response,
    )
    payload = normalized.json()

    assert normalized.headers["Content-Type"].startswith("application/json")
    assert payload["choices"][0] == {
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "reasoning_content": "step two",
            "content": "answer",
        },
    }
    assert payload["usage"]["total_tokens"] == 19
    assert _native_serving(enable_thinking=True).format_response(payload) == (
        "<think>step two</think>\n<answer>answer</answer>"
    )


def test_non_streaming_json_response_is_untouched():
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "application/json"
    response._content = b'{"choices": []}'

    normalized = normalize_streaming_chat_response(
        _request({"messages": [], "stream": False}), response
    )

    assert normalized.content == b'{"choices": []}'


def test_runtime_contract_exposes_complete_native_serving_profile():
    text, env = _pipeline_runtime_contract(
        {
            "reasoning": {
                "model_name": "model-x",
                "api_url": "http://gateway/v1/chat/completions",
                "api_key": "secret",
                "max_workers": 4,
                "serving_kwargs": {
                    "enable_thinking": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            }
        }
    )
    assert "[reasoning] model=model-x" in text
    assert "key_name_of_api_key=DF_PIPELINE_API_KEY_REASONING" in text
    assert '"enable_thinking": true' in text
    assert '"stream": true' in text
    assert '"include_usage": true' in text
    assert "禁止把密钥写入 pipeline.py" in text
    assert env == {"DF_PIPELINE_API_KEY_REASONING": "secret"}
