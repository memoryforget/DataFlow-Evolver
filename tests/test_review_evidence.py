import json
from pathlib import Path

from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.core.schemas import DatasetQualityResult, TaskSpec
from dataflow_evolver.llm.serving import LLMServingABC

TASK = TaskSpec("QA", {"q": "str", "a": "str"}, [])
GOOD_JSON = json.dumps({
    "schema_score": 0.9,
    "relevance_score": 0.8,
    "correctness_score": 0.7,
    "difficulty_score": 0.6,
    "issues": ["难度偏低"],
    "failure_signals": ["F3"],
}, ensure_ascii=False)


class FakeServing:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_one(self, prompt):
        return self.responses.pop(0) if self.responses else GOOD_JSON


def _dataset(tmp_path: Path, n: int = 5) -> str:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps({"q": f"q{i}", "a": str(i)}) for i in range(n)),
        encoding="utf-8",
    )
    return str(path)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_review_evidence_contains_exact_inputs_and_result(tmp_path):
    evidence = tmp_path / "iteration_001" / "review.jsonl"
    result = ReviewAgent(FakeServing([GOOD_JSON]), sample_size=5, seed=0).review(
        _dataset(tmp_path), TASK, evidence_path=str(evidence), phase="iteration_0"
    )
    (record,) = _records(evidence)
    assert record["phase"] == "iteration_0"
    assert record["raw_response"] == GOOD_JSON
    assert record["result"]["review_score"] == result.review_score
    assert record["result"]["failure_signals"] == ["F3"]
    assert all(json.loads(line) for line in record["samples_jsonl"].splitlines())


def test_review_evidence_appends(tmp_path):
    evidence = tmp_path / "review.jsonl"
    agent = ReviewAgent(FakeServing([GOOD_JSON, GOOD_JSON]), sample_size=5, seed=0)
    dataset = _dataset(tmp_path)
    agent.review(dataset, TASK, evidence_path=str(evidence), phase="iteration_0")
    agent.review(dataset, TASK, evidence_path=str(evidence), phase="iteration_1")
    assert [row["phase"] for row in _records(evidence)] == ["iteration_0", "iteration_1"]


def test_invalid_response_is_still_recorded(tmp_path):
    evidence = tmp_path / "review.jsonl"
    agent = ReviewAgent(FakeServing(["bad", "bad", "last bad"]), sample_size=5, seed=0)
    result = agent.review(_dataset(tmp_path), TASK, evidence_path=str(evidence))
    (record,) = _records(evidence)
    assert not result.passed
    assert record["parsed"] is None and record["raw_response"] == "last bad"


def test_empty_dataset_is_recorded(tmp_path):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("", encoding="utf-8")
    evidence = tmp_path / "review.jsonl"
    ReviewAgent(FakeServing([])).review(str(dataset), TASK, evidence_path=str(evidence))
    (record,) = _records(evidence)
    assert record["samples_jsonl"] == ""
    assert record["result"]["issues"] == ["数据集为空或不可读"]


def test_evidence_write_failure_does_not_break_review(tmp_path):
    dataset = _dataset(tmp_path)
    impossible = str(Path(dataset) / "sub" / "review.jsonl")
    result = ReviewAgent(FakeServing([GOOD_JSON]), sample_size=5).review(
        dataset, TASK, evidence_path=impossible
    )
    assert result.review_score > 0


def test_review_writes_llm_embedding_and_combined_usage(tmp_path):
    class MeteredServing(LLMServingABC):
        def generate(self, prompts, **_kwargs):
            self._record_usage(
                model="review-model",
                raw_usage={
                    "prompt_tokens": 30,
                    "completion_tokens": 3,
                    "total_tokens": 33,
                },
                success=True,
                latency_seconds=0.01,
                item_count=len(prompts),
            )
            return [GOOD_JSON for _ in prompts]

    class MeteredDatasetEvaluator:
        def evaluate(self, *_args, **_kwargs):
            return DatasetQualityResult(
                enabled=True,
                status="ok",
                mmd=0.2,
                das=-0.2,
                embedding_diversity={
                    "cosine_vendi_ratio": 0.5,
                    "nearest_neighbor_cosine_p95": 0.8,
                },
                embedding_usage={
                    "schema_version": 1,
                    "kind": "embedding",
                    "usage_status": "exact",
                    "request_count": 5,
                    "tokens": {
                        "input_tokens": 77,
                        "output_tokens": 0,
                        "total_tokens": 77,
                    },
                },
            )

    iteration = tmp_path / "iteration_001"
    iteration.mkdir()
    reviewer = ReviewAgent(
        MeteredServing(),
        sample_size=5,
        dataset_quality_evaluator=MeteredDatasetEvaluator(),
        dataset_quality_fields=("q", "a"),
    )

    reviewer.review(
        _dataset(tmp_path),
        TASK,
        evidence_path=str(iteration / "review.jsonl"),
        phase="iteration_001",
        usage_dir=str(iteration),
    )

    review_usage = json.loads(
        (iteration / "review_llm_token_usage.json").read_text(encoding="utf-8")
    )
    embedding_usage = json.loads(
        (iteration / "embedding_token_usage.json").read_text(encoding="utf-8")
    )
    combined = json.loads((iteration / "token_usage.json").read_text(encoding="utf-8"))
    assert review_usage["tokens"]["total_tokens"] == 33
    assert embedding_usage["tokens"]["total_tokens"] == 77
    assert combined["review_llm"]["tokens"]["total_tokens"] == 33
    assert combined["embedding"]["tokens"]["total_tokens"] == 77
