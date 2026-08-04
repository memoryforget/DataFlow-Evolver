"""DiagnosticAgent：Phase-2 慢反馈。真实评估的 Bad Case → 可执行诊断报告，指导下一轮变异。"""
from __future__ import annotations

import json

from dataflow_evolver.agents.base import AgentABC
from dataflow_evolver.core.schemas import BadCase, DiagnosticReport


class DiagnosticAgent(AgentABC):
    name = "diagnostic"

    def __init__(self, serving, max_bad_cases: int = 30, max_retries: int = 2) -> None:
        super().__init__(serving, max_retries=max_retries)
        self.max_bad_cases = max_bad_cases

    def build_prompt(self, downstream_task: str, bad_cases: list[BadCase]) -> str:
        from dataflow_evolver.llm.prompts import build_diagnostic_prompt

        sampled = bad_cases[: self.max_bad_cases]
        bad_cases_text = "\n".join(
            json.dumps(
                {
                    "input": bc.input,
                    "prediction": bc.prediction,
                    "extracted_answer": bc.extracted_answer,
                    "ground_truth": bc.ground_truth,
                    "error_type": bc.error_type,
                },
                ensure_ascii=False,
            )
            for bc in sampled
        )
        return build_diagnostic_prompt(downstream_task, bad_cases_text)

    def parse(self, raw: dict) -> DiagnosticReport:
        return DiagnosticReport(
            summary=raw.get("summary", ""),
            failure_patterns=raw.get("failure_patterns", []),
            actionable_suggestions=raw.get("actionable_suggestions", []),
        )

    def analyze(self, bad_cases: list[BadCase], downstream_task: str = "") -> DiagnosticReport:
        """对 bad cases 归因。无 bad case 或解析失败时返回空报告（不阻塞主循环）。"""
        if not bad_cases:
            # PTB official evaluation only exposes aggregate metrics by design. Do not render a
            # "no failures" diagnosis, because that would be interpreted as actionable evidence.
            return DiagnosticReport(summary="")
        prompt = self.build_prompt(downstream_task, bad_cases)
        raw = self.run_json(prompt)
        if raw is None:
            return DiagnosticReport(summary="诊断 LLM 未返回有效 JSON；本轮无可用建议。")
        report = self.parse(raw)
        self.logger.info("诊断：%s（%d 条建议）", report.summary, len(report.actionable_suggestions))
        return report
