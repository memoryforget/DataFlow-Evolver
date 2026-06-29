"""AgentABC：非-SDK 辅助 Agent 的统一基类。

封装"拼 prompt → 调 LLM → 解析（可重试）"的样板。TaskProfiler / Review / Diagnostic
继承它；PipelineAgent 不继承（它走 Agent SDK）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dataflow_evolve.llm.serving import LLMServingABC
from dataflow_evolve.utils.logging import get_logger
from dataflow_evolve.utils.parsing import extract_json


class AgentABC(ABC):
    """带 LLM 句柄、重试与 JSON 解析的辅助 Agent 基类。"""

    name: str = "agent"

    def __init__(self, serving: LLMServingABC, max_retries: int = 2) -> None:
        self.serving = serving
        self.max_retries = max_retries
        self.logger = get_logger(f"agents.{self.name}")

    @abstractmethod
    def build_prompt(self, *args: Any, **kwargs: Any) -> str:
        """子类据输入拼出发给 LLM 的 prompt。"""
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw: str, *args: Any, **kwargs: Any) -> Any:
        """子类把 LLM 原始文本解析为结构化产物（如某个 dataclass）。"""
        raise NotImplementedError

    def run_json(self, prompt: str) -> Any:
        """调用 LLM 并要求 JSON；解析失败则重试，最终失败返回 None。"""
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            raw = self.serving.generate_one(prompt)
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
            last_err = ValueError("无法从 LLM 输出中解析出 JSON")
            self.logger.warning(
                "[%s] JSON 解析失败（第 %d/%d 次），重试中。原始输出片段：%s",
                self.name, attempt + 1, self.max_retries + 1, raw[:200],
            )
        self.logger.error("[%s] 多次重试后仍无法解析 JSON：%s", self.name, last_err)
        return None
