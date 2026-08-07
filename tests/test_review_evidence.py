import json
from pathlib import Path

from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.core.schemas import TaskSpec

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
