import json
from pathlib import Path

import pytest

from dataflow_evolver.core.downstream_eval import (
    DownstreamBenchmarkResult,
    DownstreamEvaluationResult,
)
from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult, TaskSpec
from dataflow_evolver.engine.execution_wrapper import ExecutionResult
from dataflow_evolver.main import Candidate, ReviewOnlyLoop, select_best


def _config(code: str) -> PipelineConfig:
    return PipelineConfig(
        operators=[{"name": f"op-{code}"}], field_flow="raw -> output",
        code=code, rationale=f"iteration {code}",
    )


def _review(score: float, passed: bool, issue: str) -> ReviewResult:
    return ReviewResult(
        schema_score=0.9, relevance_score=0.9, passed=passed, issues=[issue],
        correctness_score=score, difficulty_score=0.7,
        llm_composite=score, review_score=score,
    )


class FakePipelineAgent:
    def __init__(self):
        self.feedback = []
        self.parent_codes = []
        self.histories = []
        self.downstream_feedback = []
        self.initial_downstream_feedback = None
        self.repairs = []

    def generate_initial(self, **kwargs):
        self.initial_downstream_feedback = kwargs.get("downstream_feedback")
        return _config("0")

    def generate(self, **kwargs):
        self.feedback.append(kwargs["parent_review"])
        self.parent_codes.append(kwargs["parent_code"])
        self.histories.append(kwargs["evolution_history"])
        self.downstream_feedback.append(kwargs.get("downstream_feedback"))
        return _config(str(len(self.feedback)))

    def self_correct(self, iteration_dir, traceback, diagnostics="", failure_stage="execution"):
        self.repairs.append((iteration_dir, traceback, diagnostics, failure_stage))
        return _config("repaired")


class FakeExecutor:
    def __init__(self, failures: int = 0):
        self.failures = failures
        self.calls = []

    def run(self, config, iteration_dir, env_overrides=None):
        iteration_dir = Path(iteration_dir)
        self.calls.append((config.code, iteration_dir, env_overrides))
        if self.failures:
            self.failures -= 1
            return ExecutionResult(
                False,
                traceback="boom",
                diagnostics='[{"rows": 0}]',
                failure_stage="compile",
            )
        output = iteration_dir / "output.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")
        return ExecutionResult(True, dataset_path=str(output))


class FakeReviewer:
    def __init__(self, results=None):
        self.results = results or [
            _review(0.3, False, "improve correctness"),
            _review(0.8, True, "good"),
            _review(0.6, True, "less diverse"),
        ]

    def review(
        self,
        dataset_path,
        task,
        evidence_path=None,
        phase="search",
        parent_review=None,
        usage_dir=None,
    ):
        return self.results.pop(0)


class FakeDownstreamEvaluator:
    expected_benchmarks = ["gsm8k"]
    bad_cases_per_benchmark = 3

    def __init__(self, *, baseline_enabled=False):
        self.requests = []
        self.baseline_enabled = baseline_enabled

    def run(self, request, checkpoint_dir):
        self.requests.append((request, Path(checkpoint_dir)))
        return DownstreamEvaluationResult(
            status="ok",
            checkpoint_iteration=request.checkpoint_iteration,
            incumbent_iteration=request.incumbent_iteration,
            dataset_path=request.dataset_path,
            stage=request.stage,
            training_performed=request.training_performed,
            model="downstream-model",
            benchmarks={
                "gsm8k": DownstreamBenchmarkResult(
                    score=0.5 + 0.1 * len(self.requests),
                    metric="accuracy",
                    incorrect_count=0,
                    bad_cases=[],
                )
            },
        )


def _loop(
    tmp_path,
    *,
    max_iterations=3,
    execution_repairs=0,
    failures=0,
    reviewer=None,
    downstream_evaluator=None,
    downstream_interval=3,
):
    workspace = tmp_path / "workspace"
    return ReviewOnlyLoop(
        pipeline_agent=FakePipelineAgent(), executor=FakeExecutor(failures),
        reviewer=reviewer or FakeReviewer(),
        task=TaskSpec("clean raw data", {"instruction": "str", "output": "str"}, []),
        workspace=workspace, run_name="run_a",
        max_iterations=max_iterations, execution_repairs=execution_repairs,
        downstream_evaluator=downstream_evaluator,
        downstream_interval=downstream_interval,
    )


def test_review_feedback_is_passed_directly_to_next_pipeline_prompt(tmp_path):
    loop = _loop(tmp_path)
    best = loop.run()
    assert best is not None and best.iteration == 1
    assert [review.review_score for review in loop.pipeline_agent.feedback] == [0.3, 0.8]
    manifest = json.loads((loop.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed_iterations"] == 3
    assert manifest["data_preparation_wall_time_seconds"] >= 0.0
    assert manifest["total_execution_repairs"] == 0
    assert manifest["best"]["iteration"] == 1
    assert [step["outcome"] for step in manifest["evolution_history"]] == [
        "accepted", "accepted", "rejected"
    ]
    assert (loop.run_dir / "iteration_001").is_dir()
    assert manifest["best"]["iteration_dir"].endswith("iteration_002")


def test_rejected_challenger_does_not_become_next_parent(tmp_path):
    reviewer = FakeReviewer(
        [
            _review(0.8, True, "strong baseline"),
            _review(0.3, True, "bad strategy"),
            _review(0.9, True, "successful alternative"),
        ]
    )
    loop = _loop(tmp_path, reviewer=reviewer)

    best = loop.run()

    assert best is not None and best.iteration == 2
    assert loop.pipeline_agent.parent_codes == ["0", "0"]
    rejected = loop.pipeline_agent.histories[1][-1]
    assert rejected["outcome"] == "rejected"
    assert rejected["decision"] == "iteration 1"
    assert rejected["issues"] == ["bad strategy"]


def test_execution_repairs_are_configurable_and_receive_diagnostics(tmp_path):
    loop = _loop(tmp_path, max_iterations=1, execution_repairs=2, failures=2)
    assert loop.run() is not None
    assert len(loop.pipeline_agent.repairs) == 2
    assert loop.pipeline_agent.repairs[0][2] == '[{"rows": 0}]'
    assert loop.pipeline_agent.repairs[0][3] == "compile"
    manifest = json.loads((loop.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_execution_repairs"] == 2
    assert manifest["best"]["execution_repairs"] == 2
    assert manifest["best"]["wall_time_seconds"] >= 0.0


def test_downstream_evaluation_runs_every_three_rounds_on_current_incumbent(tmp_path):
    downstream = FakeDownstreamEvaluator()
    reviewer = FakeReviewer(
        [_review(score, True, f"review {index}") for index, score in enumerate(
            [0.4, 0.8, 0.6, 0.7, 0.9, 0.85, 0.88], start=1
        )]
    )
    loop = _loop(
        tmp_path,
        max_iterations=7,
        reviewer=reviewer,
        downstream_evaluator=downstream,
    )

    best = loop.run()

    assert best is not None and best.iteration == 4
    assert [item[0].checkpoint_iteration for item in downstream.requests] == [3, 6]
    assert [item[0].incumbent_iteration for item in downstream.requests] == [2, 5]
    feedback = loop.pipeline_agent.downstream_feedback
    assert feedback[:2] == [None, None]
    assert [item.checkpoint_iteration for item in feedback[2:5]] == [3, 3, 3]
    assert feedback[5].checkpoint_iteration == 6
    assert feedback[5].score_deltas["gsm8k"] == pytest.approx(0.1)
    manifest = json.loads((loop.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["downstream_evaluation"]["completed_checkpoints"] == 2
    assert manifest["downstream_evaluation_wall_time_seconds"] >= 0.0
    assert manifest["total_wall_time_seconds"] >= manifest["data_preparation_wall_time_seconds"]
    assert (loop.run_dir / "downstream_evaluations.jsonl").is_file()


def test_downstream_baseline_runs_before_initial_pipeline_and_anchors_deltas(tmp_path):
    downstream = FakeDownstreamEvaluator(baseline_enabled=True)
    reviewer = FakeReviewer(
        [_review(score, True, f"review {index}") for index, score in enumerate(
            [0.4, 0.8, 0.6, 0.7], start=1
        )]
    )
    loop = _loop(
        tmp_path,
        max_iterations=4,
        reviewer=reviewer,
        downstream_evaluator=downstream,
    )

    assert loop.run() is not None

    requests = [item[0] for item in downstream.requests]
    assert [item.checkpoint_iteration for item in requests] == [0, 3]
    assert requests[0].stage == "baseline"
    assert requests[0].training_performed is False
    assert requests[0].incumbent_iteration is None
    assert requests[0].dataset_path is None
    assert requests[0].pipeline_path is None
    initial = loop.pipeline_agent.initial_downstream_feedback
    assert initial is not None and initial.stage == "baseline"
    assert [item.stage for item in loop.pipeline_agent.downstream_feedback] == [
        "baseline", "baseline", "periodic"
    ]
    periodic = loop.downstream_evaluations[-1]
    assert periodic.score_deltas["gsm8k"] == pytest.approx(0.1)
    assert periodic.score_deltas_vs_baseline["gsm8k"] == pytest.approx(0.1)
    manifest = json.loads((loop.run_dir / "manifest.json").read_text(encoding="utf-8"))
    downstream_manifest = manifest["downstream_evaluation"]
    assert downstream_manifest["completed_checkpoints"] == 2
    assert downstream_manifest["completed_periodic_checkpoints"] == 1
    assert downstream_manifest["baseline_enabled"] is True
    assert downstream_manifest["baseline_succeeded"] is True


def test_best_selection_prefers_passed_candidate_before_raw_score(tmp_path):
    failed = Candidate(0, tmp_path / "iteration_001", _config("0"), review=_review(0.99, False, "bad"))
    passed = Candidate(1, tmp_path / "iteration_002", _config("1"), review=_review(0.70, True, "ok"))
    assert select_best([failed, passed]) is passed


def test_existing_run_directory_is_rejected(tmp_path):
    loop = _loop(tmp_path, max_iterations=1)
    loop.run_dir.mkdir(parents=True)
    (loop.run_dir / "stale.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        loop.run()


@pytest.mark.parametrize("name", ["", "../escape", "/absolute", "nested/run"])
def test_run_name_cannot_escape_workspace(tmp_path, name):
    with pytest.raises(ValueError):
        ReviewOnlyLoop(
            pipeline_agent=FakePipelineAgent(), executor=FakeExecutor(), reviewer=FakeReviewer(),
            task=TaskSpec("task", {"instruction": "str"}, []), workspace=tmp_path / "workspace",
            run_name=name, max_iterations=1,
        )
