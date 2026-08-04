import json

import pytest

from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.core.diversity import scan_diversity
from dataflow_evolver.core.schemas import TaskProfile
from dataflow_evolver.llm.serving import LLMServingABC


class StubServing(LLMServingABC):
    def generate(self, prompts, **kwargs):
        return ["{}" for _ in prompts]


def _recipe():
    return {
        "mode": "synthetic",
        "coverage_buckets": [{"name": "a", "quota": 2}, {"name": "b", "quota": 2}],
    }


def test_normalized_duplicates_ignore_surface_numbers(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {"instruction": "Alice has 3 apples", "output": "ANSWER: 3"},
        {"instruction": "Alice has 8 apples", "output": "ANSWER: 8"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    metrics = scan_diversity(path, ["instruction", "output"])
    assert metrics.normalized_dup_rate == pytest.approx(0.5)


def test_recipe_coverage_and_metadata_are_scored(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = []
    for bucket, graph in zip(("a", "a", "b", "b"), ("add", "subtract", "multiply", "divide")):
        rows.append({
            "instruction": f"different problem {len(rows)} long enough",
            "output": f"reasoning {len(rows)} ANSWER: {len(rows)}",
            "_bucket": bucket,
            "_generation_strategy": "program_first",
            "_operation_graph": graph,
        })
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    metrics = scan_diversity(path, ["instruction", "output"], _recipe())
    assert metrics.coverage_score == pytest.approx(1.0)
    assert metrics.largest_bucket_fraction == pytest.approx(0.5)
    assert metrics.metadata_complete
    assert metrics.structural_dup_rate == 0.0


def test_single_operation_graph_is_structural_collapse(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {
            "instruction": f"problem {i}", "output": f"answer {i}", "_bucket": "a",
            "_generation_strategy": "program_first", "_operation_graph": "add->format",
        }
        for i in range(4)
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    metrics = scan_diversity(path, ["instruction", "output"], _recipe())
    assert metrics.structural_dup_rate == pytest.approx(1.0)


def test_missing_recipe_metadata_fails_review(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")
    profile = TaskProfile("t", {"instruction": "str", "output": "str"}, [], "sft")
    agent = ReviewAgent(StubServing())
    result = agent.parse({
        "correctness_score": 1, "relevance_score": 1, "difficulty_score": 1,
        "diversity_score": 1, "schema_score": 1,
    })
    agent.apply_hard_metrics(result, str(path), profile, recipe=_recipe())
    assert not result.passed
    assert result.coverage_score == 0.0
    assert result.tier1_score == 0.0
