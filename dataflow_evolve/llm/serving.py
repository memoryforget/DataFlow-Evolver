"""非-SDK 模型后端：供辅助 Agent（Profiler/Review/Diagnostic）与 DataFlow 运行时调用。

提供统一的 `LLMServingABC.generate(prompts) -> list[str]` 抽象，按 YAML 配置在
OpenAI-compatible API 与本地 vLLM 之间切换（vLLM 路径惰性依赖 DataFlow）。

注意：PipelineAgent 不走这里，它走 Claude Agent SDK（见 llm/agent_sdk.py）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dataflow_evolve.utils.config import Config
from dataflow_evolve.utils.logging import get_logger

logger = get_logger("llm.serving")


class LLMServingABC(ABC):
    """统一的批量文本生成接口。"""

    @abstractmethod
    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """对一批 prompt 返回一批补全文本，顺序与输入一一对应。"""
        raise NotImplementedError

    def generate_one(self, prompt: str, **kwargs: Any) -> str:
        """单条便捷封装。"""
        return self.generate([prompt], **kwargs)[0]


class APILLMServing(LLMServingABC):
    """OpenAI-compatible HTTP API 后端。用线程池并发跑一批 prompt。"""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        api_key: str,
        max_workers: int = 10,
        temperature: float = 0.7,
        request_timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI  # 惰性导入，避免无谓依赖

        self.model_name = model_name
        self.max_workers = max_workers
        self.temperature = temperature
        base_url = _to_base_url(api_url)
        # timeout / max_retries 交给 openai 客户端：网络抖动在 HTTP 层自动退避重试（见 review H1）
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=request_timeout,
            max_retries=max_retries,
        )
        logger.info("APILLMServing 就绪：model=%s base_url=%s", model_name, base_url)

    def _one(self, prompt: str, **kwargs: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
        )
        return resp.choices[0].message.content or ""

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self._one(prompts[0], **kwargs)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda p: self._one(p, **kwargs), prompts))


class VLLMServing(LLMServingABC):
    """本地 vLLM 后端，惰性桥接 DataFlow 的 LocalModelLLMServing_vllm。"""

    def __init__(self, hf_model_name_or_path: str, tensor_parallel_size: int = 1) -> None:
        from dataflow.serving import LocalModelLLMServing_vllm  # 惰性导入（重依赖）

        self._impl = LocalModelLLMServing_vllm(
            hf_model_name_or_path=hf_model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
        )
        logger.info("VLLMServing 就绪：model=%s tp=%d", hf_model_name_or_path, tensor_parallel_size)

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        return self._impl.generate_from_input(prompts)


def build_serving(cfg: Config) -> LLMServingABC:
    """按 `llm` 配置段构造后端。`cfg` 是 `llm` 子配置（含 backend / api / vllm）。"""
    backend = cfg.get("backend", "api")
    if backend == "api":
        api = cfg.api
        # `api_key` 直接是密钥明文；想避免入库可在 YAML 写 ${ENV_VAR}，加载时已展开（见 utils/config.py）。
        api_key = api.get("api_key", "")
        if not api_key:
            logger.warning("未配置 llm.api.api_key（可填明文或 ${ENV_VAR}）；API 调用很可能 401。")
        return APILLMServing(
            api_url=api.api_url,
            model_name=api.model_name,
            api_key=api_key,
            max_workers=api.get("max_workers", 10),
            request_timeout=api.get("request_timeout", 120.0),
            max_retries=api.get("max_retries", 2),
        )
    if backend == "vllm":
        vllm = cfg.vllm
        return VLLMServing(
            hf_model_name_or_path=vllm.hf_model_name_or_path,
            tensor_parallel_size=vllm.get("tensor_parallel_size", 1),
        )
    raise ValueError(f"未知的 llm.backend: {backend!r}（应为 'api' 或 'vllm'）")


def _to_base_url(api_url: str) -> str:
    """把完整 chat/completions URL 归一化为 openai 客户端要的 base_url。"""
    suffix = "/chat/completions"
    if api_url.endswith(suffix):
        return api_url[: -len(suffix)]
    return api_url.rstrip("/")
