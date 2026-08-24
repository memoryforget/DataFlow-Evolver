from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import threading
from pathlib import Path
from types import MethodType
from typing import Any

from dataflow.serving import APILLMServing_request


THINK_ANSWER_PATTERN = re.compile(
    r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("stats", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = args.pipeline.expanduser().resolve()
    stats_path = args.stats.expanduser().resolve()
    if not pipeline.is_file():
        raise FileNotFoundError(pipeline)

    lock = threading.Lock()
    stats: dict[str, Any] = {
        "schema_version": 1,
        "answer_format_response_calls": 0,
        "reasoning_field_promoted": 0,
        "reasoning_content_field_used": 0,
        "already_wrapped_content": 0,
        "answer_responses_without_reasoning": 0,
        "answer_responses_with_empty_content": 0,
        "serving_retry_exhausted_fallback_count": 0,
    }

    original_retry = APILLMServing_request._api_chat_id_retry

    def retry_then_empty_fallback(self, *retry_args, **retry_kwargs):
        request_id, response = original_retry(self, *retry_args, **retry_kwargs)
        if response is None:
            with lock:
                stats["serving_retry_exhausted_fallback_count"] += 1
            response = ""
        return request_id, response

    APILLMServing_request._api_chat_id_retry = retry_then_empty_fallback

    module = runpy.run_path(str(pipeline), run_name="dataflow_math_reasoning_module")
    pipeline_class = module.get("ReasoningMath_APIPipeline")
    if pipeline_class is None:
        raise RuntimeError("official pipeline does not define ReasoningMath_APIPipeline")
    model = pipeline_class()

    answer_serving = APILLMServing_request(
        api_url=os.environ["DF_PIPELINE_API_URL"],
        model_name=os.environ["DF_PIPELINE_MODEL"],
        max_workers=int(os.environ.get("DF_PIPELINE_MAX_WORKERS", "100")),
        max_retries=int(os.environ.get("DF_PIPELINE_MAX_RETRIES", "5")),
        connect_timeout=10,
        read_timeout=1800,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        chat_template_kwargs={"enable_thinking": True},
    )
    original_formatter = answer_serving.format_response

    def format_answer_response(self, response: dict, is_embedding: bool = False):
        if is_embedding:
            return original_formatter(response, is_embedding=True)
        message = response.get("choices", [{}])[0].get("message", {})
        content = str(message.get("content") or "")
        reasoning_content = str(message.get("reasoning_content") or "")
        reasoning = str(message.get("reasoning") or "")
        with lock:
            stats["answer_format_response_calls"] += 1
        if THINK_ANSWER_PATTERN.fullmatch(content):
            with lock:
                stats["already_wrapped_content"] += 1
            return content
        if not content.strip():
            with lock:
                stats["answer_responses_with_empty_content"] += 1
        if reasoning_content.strip():
            with lock:
                stats["reasoning_content_field_used"] += 1
            selected_reasoning = reasoning_content
        elif reasoning.strip():
            with lock:
                stats["reasoning_field_promoted"] += 1
            selected_reasoning = reasoning
        else:
            with lock:
                stats["answer_responses_without_reasoning"] += 1
            return content
        return f"<think>{selected_reasoning}</think>\n<answer>{content}</answer>"

    answer_serving.format_response = MethodType(format_answer_response, answer_serving)
    model.answer_generator_step7.llm_serving = answer_serving
    try:
        model.forward()
    finally:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
