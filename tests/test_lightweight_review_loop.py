import json
from pathlib import Path

import pytest

from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult, TaskProfile
from dataflow_evolver.engine.execution_wrapper import ExecutionResult
from dataflow_evolver.lightweight_main import Candidate, ReviewOnlyLoop, select_best


def _config(code: str) -> PipelineConfig:
    return PipelineConfig(
        operators=[{"name": f"op-{code}"}],
        field_flow="raw -> instruction/output",
        code=code,
        rationale=f"iteration {code}",
    )


def _review(score: float, passed: bool, issue: str) -> ReviewResult:
    return ReviewResult(
        schema_score=0.9,
        relevance_score=0.9,
        passed=passed,
        issues=[issue],
        correctness_score=score,
        difficulty_score=0.7,
        diversity_score=0.6,
        llm_composite=score,
        review_score=score,
    )


class FakePipelineAgent:
    def __init__(self):
        self.feedback = []

    def generate_initial(self, **kwargs):
        return _config("0")

    def generate(self, **kwargs):
        self.feedback.append(kwargs["parent_review"])
        return _config(str(len(self.feedback)))

    def self_correct(self, node_name, traceback):
        raise AssertionError("execution succeeds in this test")


class FakeExecutor:
    def __init__(self, root: Path):
        self.root = root
        self.calls = []

    def run(self, config, node_name, env_overrides=None):
        self.calls.append((config.code, node_name, env_overrides))
        output = self.root / node_name / "output.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")
        return ExecutionResult(True, dataset_path=str(output))


class FakeReviewer:
    def __init__(self):
        self.results = [
            _review(0.3, False, "improve correctness"),
            _review(0.8, True, "good"),
            _review(0.6, True, "less diverse"),
        ]

    def review(self, dataset_path, profile, evidence_path=None, phase="search"):
        return self.results.pop(0)


def test_review_feedback_is_passed_directly_to_next_pipeline_prompt(tmp_path):
    agent = FakePipelineAgent()
    loop = ReviewOnlyLoop(
        pipeline_agent=agent,
        executor=FakeExecutor(tmp_path),
        reviewer=FakeReviewer(),
        profile=TaskProfile("clean raw data", {"instruction": "str", "output": "str"}, []),
        workspace=tmp_path / "reports",
        agent_root=tmp_path,
        run_name="run_a",
        max_iterations=3,
        execution_retries=0,
    )

    best = loop.run()

    assert best is not None and best.iteration == 1
    assert [review.review_score for review in agent.feedback] == [0.3, 0.8]
    assert agent.feedback[0].issues == ["improve correctness"]
    manifest = json.loads((tmp_path / "reports" / "run_a_best.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "review_only"
    assert manifest["completed_iterations"] == 3
    assert manifest["best"]["iteration"] == 1
    assert "train" not in manifest and "tier2" not in manifest


def test_best_selection_prefers_passed_candidate_before_raw_score():
    failed = Candidate(0, "node_0", _config("0"), review=_review(0.99, False, "bad schema"))
    passed = Candidate(1, "node_1", _config("1"), review=_review(0.70, True, "ok"))
    assert select_best([failed, passed]) is passed


def test_existing_run_directory_is_rejected(tmp_path):
    run_dir = tmp_path / "same_run"
    run_dir.mkdir()
    (run_dir / "stale.txt").write_text("old", encoding="utf-8")
    loop = ReviewOnlyLoop(
        pipeline_agent=FakePipelineAgent(),
        executor=FakeExecutor(tmp_path),
        reviewer=FakeReviewer(),
        profile=TaskProfile("task", {"instruction": "str"}, []),
        workspace=tmp_path / "reports",
        agent_root=tmp_path,
        run_name="same_run",
        max_iterations=1,
    )
    with pytest.raises(FileExistsError, match="旧节点产物污染"):
        loop.run()


@pytest.mark.parametrize("name", ["", "../escape", "/absolute"])
def test_run_name_cannot_escape_agent_root(tmp_path, name):
    with pytest.raises(ValueError):
        ReviewOnlyLoop(
            pipeline_agent=FakePipelineAgent(),
            executor=FakeExecutor(tmp_path),
            reviewer=FakeReviewer(),
            profile=TaskProfile("task", {"instruction": "str"}, []),
            workspace=tmp_path / "reports",
            agent_root=tmp_path,
            run_name=name,
            max_iterations=1,
        )
