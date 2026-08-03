"""TaskProfilerAgent：高层任务描述 → 形式化 TaskProfile（初始化时运行一次）。"""
from __future__ import annotations

from dataflow_evolver.agents.base import AgentABC
from dataflow_evolver.core.schemas import TaskProfile
from dataflow_evolver.llm.prompts import build_profiler_prompt


class TaskProfilerAgent(AgentABC):
    name = "task_profiler"

    def build_prompt(
        self,
        task_description: str,
        sample_text: str | None = None,
        fixed_schema: dict | None = None,
    ) -> str:
        return build_profiler_prompt(task_description, sample_text, fixed_schema)

    def parse(self, raw: dict) -> TaskProfile:
        return TaskProfile(
            task_description=raw.get("task_description", ""),
            target_schema=raw.get("target_schema", {}),
            quality_criteria=raw.get("quality_criteria", []),
            downstream_task=raw.get("downstream_task", ""),
        )

    def run(
        self,
        task_description: str,
        sample_text: str | None = None,
        fixed_schema: dict | None = None,
    ) -> TaskProfile | None:
        """主入口：把任务描述（可选附输入语料样例）交给 LLM，得到 TaskProfile。

        样例由调用方从 `InputCorpus` 取，本 Agent 不碰文件系统——语料的发现、格式识别与
        截断规则集中在 core/input_corpus.py 一处，避免两处各自解析同一批文件。

        `fixed_schema`（`trainer.dataset_schema`）非空时，它是下游训练脚本的**真实契约**，
        解析后直接覆盖模型的输出：prompt 里已经说了不许改，但模型仍可能改名或补字段，而这份
        schema 决定流水线该产出什么、评审按什么打分，猜错要到训练阶段才炸。
        """
        prompt = self.build_prompt(task_description, sample_text, fixed_schema)
        raw = self.run_json(prompt)
        if raw is None:
            self.logger.error("TaskProfiler 未能得到有效 JSON。")
            return None
        profile = self.parse(raw)
        if fixed_schema:
            if profile.target_schema != fixed_schema:
                self.logger.warning(
                    "模型给的 schema 与下游契约不符，按契约覆盖：%s → %s",
                    profile.target_schema, fixed_schema,
                )
            profile.target_schema = dict(fixed_schema)
        self.logger.info(
            "TaskProfile 就绪：downstream=%s schema=%s",
            profile.downstream_task, profile.target_schema,
        )
        return profile
