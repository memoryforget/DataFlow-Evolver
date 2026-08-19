"""Lightweight OpenAI-compatible serving for generated DataFlow operators.

Only the stable ``dataflow.core.LLMServingABC`` contract is imported.  This
avoids importing DataFlow's aggregate serving package, whose ``__init__``
eagerly imports every local, multimodal, audio, and cloud backend.
"""
from __future__ import annotations

import logging
import os
import re
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import requests
from requests.adapters import HTTPAdapter

from dataflow.core import LLMServingABC


ResponseMode = Literal["content", "reasoning_and_content", "auto"]


class EvolverAPILLMServing(LLMServingABC):
    """OpenAI-compatible HTTP transport with configurable response composition."""

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

        deprecated_timeout = configs.pop("timeout", None)
        if deprecated_timeout is not None:
            warnings.warn(
                "The `timeout` parameter is deprecated; use read_timeout.",
                DeprecationWarning,
                stacklevel=2,
            )
            read_timeout = float(deprecated_timeout)

        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max(1, int(max_workers))
        self.max_retries = max(1, int(max_retries))
        self.timeout = (float(connect_timeout), float(read_timeout))
        self.configs = dict(configs)
        self.configs["temperature"] = temperature
        self.response_mode = response_mode
        self.reasoning_fields = _as_tuple(
            reasoning_fields or ("reasoning", "reasoning_content")
        )
        self.reasoning_template = reasoning_template
        self.allowed_finish_reasons = _as_tuple(allowed_finish_reasons)
        self.require_reasoning = bool(require_reasoning)
        self.require_content = bool(require_content)
        self.accept_preformatted = bool(accept_preformatted)
        self.logger = logging.getLogger("dataflow_evolver.compat.serving")

        api_key = os.environ.get(key_name_of_api_key)
        if api_key is None:
            raise ValueError(
                f"Missing API key environment variable `{key_name_of_api_key}` "
                f"for {api_url}"
            )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers,
            max_retries=0,
            pool_block=True,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def start_serving(self) -> None:
        return None

    def cleanup(self) -> None:
        self.session.close()

    def format_response(self, payload: dict[str, Any], is_embedding: bool = False) -> Any:
        if is_embedding:
            return payload.get("data", [{}])[0].get("embedding", [])
        choices = payload.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        if self.allowed_finish_reasons and finish_reason not in self.allowed_finish_reasons:
            return None

        content = message.get("content")
        content = "" if content is None else str(content).strip()
        if self.require_content and not content:
            return None
        if self.response_mode == "content":
            return content
        if self.accept_preformatted and re.search(r"<think>.*?</think>", content, re.DOTALL):
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

    def _request_once(
        self,
        request_id: int,
        payload: Any,
        *,
        is_embedding: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        if is_embedding:
            body = {"model": self.model_name, "input": payload}
        else:
            body = {"model": self.model_name, "messages": payload}
            if json_schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "custom_response",
                        "strict": True,
                        "schema": json_schema,
                    },
                }
        body.update(self.configs)
        response = self.session.post(
            self.api_url,
            headers=self.headers,
            json=body,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            self.logger.warning(
                "LLM request failed id=%s status=%s body=%s",
                request_id,
                response.status_code,
                response.text[:500],
            )
            return request_id, None
        return request_id, self.format_response(response.json(), is_embedding)

    def _request_with_retry(self, request_id: int, payload: Any, **kwargs: Any) -> tuple[int, Any]:
        for attempt in range(self.max_retries):
            try:
                result = self._request_once(request_id, payload, **kwargs)
            except requests.exceptions.ConnectTimeout as exc:
                raise RuntimeError(f"Cannot connect to LLM server: {exc}") from exc
            except requests.exceptions.ConnectionError as exc:
                raise RuntimeError(f"LLM server connection failed: {exc}") from exc
            except requests.exceptions.Timeout as exc:
                self.logger.warning("LLM request timeout id=%s: %s", request_id, exc)
                result = (request_id, None)
            except (requests.RequestException, ValueError) as exc:
                self.logger.warning("LLM request error id=%s: %s", request_id, exc)
                result = (request_id, None)
            if result[1] is not None:
                return result
            if attempt + 1 < self.max_retries:
                time.sleep(2**attempt)
        return request_id, None

    def _run_parallel(self, tasks: list[tuple[Any, dict[str, Any]]]) -> list[Any]:
        responses: list[Any] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._request_with_retry, index, payload, **kwargs): index
                for index, (payload, kwargs) in enumerate(tasks)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    response_id, value = future.result()
                except RuntimeError:
                    for pending in futures:
                        pending.cancel()
                    raise
                responses[response_id if response_id is not None else index] = value
        return responses

    def generate_from_input(
        self,
        user_inputs: list[str],
        system_prompt: str = "You are a helpful assistant",
        json_schema: dict[str, Any] | None = None,
    ) -> list[Any]:
        tasks = [
            (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                {"json_schema": json_schema},
            )
            for user_input in user_inputs
        ]
        return self._run_parallel(tasks)

    def generate_from_conversations(self, conversations: list[list[dict[str, Any]]]) -> list[Any]:
        return self._run_parallel([(conversation, {}) for conversation in conversations])

    def generate_embedding_from_input(self, texts: list[str]) -> list[Any]:
        return self._run_parallel([(text, {"is_embedding": True}) for text in texts])


def _as_tuple(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
