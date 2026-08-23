from __future__ import annotations

from dataflow.core import LLMServingABC

from dataflow_evolver.compat import EvolverAPILLMServing
from dataflow_evolver.main import _pipeline_runtime_contract


def make_serving(monkeypatch, **kwargs):
    monkeypatch.setenv("TEST_PIPELINE_API_KEY", "secret")
    return EvolverAPILLMServing(
        api_url="http://gateway/v1/chat/completions",
        key_name_of_api_key="TEST_PIPELINE_API_KEY",
        model_name="model-x",
        **kwargs,
    )


def test_serving_uses_lightweight_core_contract():
    assert issubclass(EvolverAPILLMServing, LLMServingABC)


def test_cot_response_is_composed_by_config(monkeypatch):
    serving = make_serving(
        monkeypatch,
        response_mode="reasoning_and_content",
        reasoning_fields=["reasoning"],
        reasoning_template="<think>{reasoning}</think>\n{content}",
        allowed_finish_reasons=["stop"],
        require_reasoning=True,
    )
    payload = {"choices": [{"finish_reason": "stop", "message": {
        "reasoning": "derive", "content": "answer"
    }}]}
    assert serving.format_response(payload) == "<think>derive</think>\nanswer"
    serving.cleanup()


def test_non_cot_mode_returns_content_only(monkeypatch):
    serving = make_serving(monkeypatch, response_mode="content")
    payload = {"choices": [{"finish_reason": "length", "message": {
        "reasoning": "hidden", "content": "answer"
    }}]}
    assert serving.format_response(payload) == "answer"
    serving.cleanup()


def test_runtime_contract_names_compatibility_layer():
    text, env = _pipeline_runtime_contract({
        "reasoning": {
            "model_name": "model-x",
            "api_url": "http://gateway/v1/chat/completions",
            "api_key": "secret",
            "max_workers": 4,
            "serving_kwargs": {"response_mode": "reasoning_and_content"},
        }
    })
    assert "[reasoning] model=model-x" in text
    assert "key_name_of_api_key=DF_PIPELINE_API_KEY_REASONING" in text
    assert '"response_mode": "reasoning_and_content"' in text
    assert "禁止把密钥写入 pipeline.py" in text
    assert env == {"DF_PIPELINE_API_KEY_REASONING": "secret"}
