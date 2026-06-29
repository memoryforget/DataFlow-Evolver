"""ReviewAgent：Phase-1 快反馈。训练前抽样评审生成数据集的 schema 符合度与任务相关性。"""
from __future__ import annotations

import json
from pathlib import Path

from dataflow_evolve.agents.base import AgentABC
from dataflow_evolve.core.schemas import ReviewResult, TaskProfile
from dataflow_evolve.llm.prompts import build_review_prompt


class ReviewAgent(AgentABC):
    name = "review"

    def __init__(
        self,
        serving,
        schema_threshold: float = 0.6,
        relevance_threshold: float = 0.6,
        sample_size: int = 20,
        max_retries: int = 2,
    ) -> None:
        super().__init__(serving, max_retries=max_retries)
        self.schema_threshold = schema_threshold
        self.relevance_threshold = relevance_threshold
        self.sample_size = sample_size

    def build_prompt(self, profile: TaskProfile, samples_text: str) -> str:
        return build_review_prompt(profile, samples_text)

    def parse(self, raw: dict) -> ReviewResult:
        schema_score = float(raw.get("schema_score", 0.0))
        relevance_score = float(raw.get("relevance_score", 0.0))
        passed = (
            schema_score >= self.schema_threshold
            and relevance_score >= self.relevance_threshold
        )
        return ReviewResult(
            schema_score=schema_score,
            relevance_score=relevance_score,
            passed=passed,
            issues=raw.get("issues", []),
        )

    def review(self, dataset_path: str, profile: TaskProfile) -> ReviewResult:
        """抽样数据集 → LLM 打分 → ReviewResult。解析失败时返回未通过（保守）。"""
        samples_text = _sample_jsonl(dataset_path, self.sample_size)
        if not samples_text:
            return ReviewResult(0.0, 0.0, False, ["数据集为空或不可读"])
        prompt = self.build_prompt(profile, samples_text)
        raw = self.run_json(prompt)
        if raw is None:
            return ReviewResult(0.0, 0.0, False, ["评审 LLM 未返回有效 JSON"])
        result = self.parse(raw)
        self.logger.info(
            "Review：schema=%.2f relevance=%.2f passed=%s",
            result.schema_score, result.relevance_score, result.passed,
        )
        return result


def _sample_jsonl(path: str, n: int) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    out: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                out.append(line)
    return "\n".join(out)
