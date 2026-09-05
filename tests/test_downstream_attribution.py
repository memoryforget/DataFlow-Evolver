import json

from dataflow_evolver.agents.downstream_attribution import DownstreamAttributionAgent
from dataflow_evolver.core.downstream_eval import (
    DownstreamBadCase,
    DownstreamBenchmarkResult,
    DownstreamEvaluationResult,
)
from dataflow_evolver.core.schemas import TaskSpec
from dataflow_evolver.llm.prompts import build_downstream_attribution_prompt


class _Serving:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate_one(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(self.payload, ensure_ascii=False)


def _feedback():
    return DownstreamEvaluationResult(
        status="ok",
        checkpoint_iteration=2,
        incumbent_iteration=1,
        dataset_path="/run/dataset.jsonl",
        model="test-model",
        benchmarks={
            "gsm8k": DownstreamBenchmarkResult(
                score=0.4,
                metric="accuracy",
                incorrect_count=2,
                wrong_answer_count=2,
                parse_failure_count=0,
                runaway_count=0,
                bad_cases=[
                    DownstreamBadCase("q1", "r1", "p1", "a1"),
                    DownstreamBadCase("q2", "r2", "p2", "a2"),
                ],
            )
        },
    )


def test_attribution_is_structured_and_audited(tmp_path):
    serving = _Serving(
        {
            "summary": "上下文经常不完整",
            "findings": [
                {
                    "category": "missing_context",
                    "data_side": True,
                    "confidence": 0.9,
                    "evidence_ids": ["gsm8k#1"],
                    "diagnosis": "前置条件被切掉",
                    "recommended_actions": ["保留完整题干"],
                },
                {
                    "category": "bogus",
                    "data_side": False,
                    "confidence": "nan",
                    "evidence_ids": "bad",
                    "diagnosis": "模型原因",
                },
            ],
            "non_data_causes": ["解析器可能造成误差"],
            "recommended_pipeline_changes": ["增加上下文完整性过滤"],
        }
    )
    agent = DownstreamAttributionAgent(serving, max_findings=2, max_actions=3)
    result = agent.attribute(
        task=TaskSpec("math", {"instruction": "str", "output": "str"}, []),
        feedback=_feedback(),
        pipeline_operators=[{"name": "Chunker"}],
        evidence_path=tmp_path / "attribution.json",
    )

    assert result.status == "ok"
    assert result.findings[0]["category"] == "missing_context"
    assert result.findings[0]["confidence"] == 0.9
    assert result.findings[1]["category"] == "other"
    assert result.findings[1]["confidence"] == 0.0
    record = json.loads((tmp_path / "attribution.json").read_text())
    assert record["result"]["status"] == "ok"
    assert "Question: q1" in serving.prompts[0]
    assert "不要把单个 benchmark 题目当作训练数据" in serving.prompts[0]
    assert "findings 最多 2 条" in serving.prompts[0]
    assert "recommended_actions 最多 3 条" in serving.prompts[0]


def test_attribution_prompt_uses_dynamic_benchmark_evidence_id():
    feedback = _feedback()
    feedback.benchmarks = {"summarization": feedback.benchmarks["gsm8k"]}
    prompt = build_downstream_attribution_prompt(
        task=TaskSpec("summarization", {"instruction": "str", "output": "str"}, []),
        feedback=feedback,
        pipeline_operators=[{"name": "Summarizer"}],
    )

    assert '"evidence_ids": ["summarization#1"]' in prompt
    assert '"evidence_ids": ["gsm8k#1"]' not in prompt


def test_attribution_rejects_empty_response():
    agent = DownstreamAttributionAgent(_Serving({}))
    result = agent.parse({}, _feedback())
    assert result.status == "failed"
    assert result.error
