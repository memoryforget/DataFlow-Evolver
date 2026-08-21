"""Strict response adapter over open-dataflow's API LLM transport."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from dataflow.serving.api_llm_serving_request import APILLMServing_request


ResponseMode = Literal["content", "reasoning_and_content", "auto"]


class EvolverAPILLMServing(APILLMServing_request):
    """Reuse DataFlow transport while enforcing an explicit response contract.

    The formal math experiment uses response_mode="content". Optional reasoning
    composition remains available for other tasks and legacy comparison
    pipelines, but is never inferred from a model name.
    """

    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        key_name_of_api_key: str = "DF_API_KEY",
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_workers: int = 10,
        max_retries: int = 5,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        response_mode: ResponseMode = "content",
        reasoning_fields: Sequence[str] | str | None = None,
        reasoning_template: str = "<think>\n{reasoning}\n</think>\n\n{content}",
        allowed_finish_reasons: Sequence[str] | str | None = None,
        require_reasoning: bool = False,
        require_content: bool = True,
        accept_preformatted: bool = True,
        reasoning_field: str | None = None,
        include_reasoning: bool | None = None,
        reasoning_output_format: str | None = None,
        require_finish_reason_stop: bool = False,
        require_reasoning_and_content: bool = False,
        **configs: Any,
    ) -> None:
        if include_reasoning is not None:
            response_mode = "reasoning_and_content" if include_reasoning else "content"
        if reasoning_field and reasoning_fields is None:
            reasoning_fields = (reasoning_field,)
        if reasoning_output_format == "think_answer":
            reasoning_template = "<think>\n{reasoning}\n</think>\n<answer>{content}</answer>"
        elif reasoning_output_format == "think":
            reasoning_template = "<think>\n{reasoning}\n</think>\n\n{content}"
        if require_finish_reason_stop and allowed_finish_reasons is None:
            allowed_finish_reasons = ("stop",)
        if require_reasoning_and_content:
            require_reasoning = True
            require_content = True
        if response_mode not in {"content", "reasoning_and_content", "auto"}:
            raise ValueError(f"Unsupported response_mode: {response_mode}")

        self.response_mode = response_mode
        self.reasoning_fields = _as_tuple(
            reasoning_fields or ("reasoning", "reasoning_content")
        )
        self.reasoning_template = reasoning_template
        self.allowed_finish_reasons = _as_tuple(allowed_finish_reasons)
        self.require_reasoning = bool(require_reasoning)
        self.require_content = bool(require_content)
        self.accept_preformatted = bool(accept_preformatted)

        super().__init__(
            api_url=api_url,
            key_name_of_api_key=key_name_of_api_key,
            model_name=model_name,
            temperature=temperature,
            max_workers=max_workers,
            max_retries=max_retries,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            **configs,
        )

    def format_response(self, payload: dict[str, Any], is_embedding: bool = False) -> Any:
        if is_embedding:
            return super().format_response(payload, is_embedding=True)

        choices = payload.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        if (
            self.allowed_finish_reasons
            and choice.get("finish_reason") not in self.allowed_finish_reasons
        ):
            return None

        message = choice.get("message") or {}
        content_value = message.get("content")
        content = "" if content_value is None else str(content_value).strip()
        if self.require_content and not content:
            return None
        if self.response_mode == "content":
            return content

        if self.accept_preformatted and re.search(
            r"<think>.*?</think>", content, re.DOTALL
        ):
            return content

        reasoning = ""
        for field in self.reasoning_fields:
            value = message.get(field)
            if value is not None and str(value).strip():
                reasoning = str(value).strip()
                break
        if self.require_reasoning and not reasoning:
            return None
        if not reasoning:
            return content
        return self.reasoning_template.format(
            reasoning=reasoning,
            content=content,
            answer=content,
        )


def _as_tuple(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
