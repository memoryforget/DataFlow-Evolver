"""TaskProfilerAgent：高层任务描述 → 形式化 TaskProfile（初始化时运行一次）。"""
from __future__ import annotations

from pathlib import Path

from dataflow_evolve.agents.base import AgentABC
from dataflow_evolve.core.schemas import TaskProfile
from dataflow_evolve.llm.prompts import build_profiler_prompt


class TaskProfilerAgent(AgentABC):
    name = "task_profiler"

    def build_prompt(self, task_description: str, sample_text: str | None = None) -> str:
        return build_profiler_prompt(task_description, sample_text)

    def parse(self, raw: dict) -> TaskProfile:
        return TaskProfile(
            task_description=raw.get("task_description", ""),
            target_schema=raw.get("target_schema", {}),
            domain=raw.get("domain", "general"),
            quality_criteria=raw.get("quality_criteria", []),
            downstream_task=raw.get("downstream_task", ""),
        )

    def run(self, task_description: str, seed_path: str | None = None) -> TaskProfile | None:
        """主入口：可选读取种子数据头几行作为样例，调用 LLM 得到 TaskProfile。"""
        sample_text = _read_sample_lines(seed_path) if seed_path else None
        prompt = self.build_prompt(task_description, sample_text)
        raw = self.run_json(prompt)
        if raw is None:
            self.logger.error("TaskProfiler 未能得到有效 JSON。")
            return None
        profile = self.parse(raw)
        self.logger.info(
            "TaskProfile 就绪：domain=%s downstream=%s schema=%s",
            profile.domain, profile.downstream_task, profile.target_schema,
        )
        return profile


def _read_sample_lines(seed_path: str, n: int = 3) -> str | None:
    path = Path(seed_path)
    if not path.exists():
        return None
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines) if lines else None
