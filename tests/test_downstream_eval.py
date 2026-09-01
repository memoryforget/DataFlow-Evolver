import json
import sys
from pathlib import Path

from dataflow_evolver.core.downstream_eval import (
    CommandDownstreamEvaluator,
    DownstreamAttributionResult,
    DownstreamBadCase,
    DownstreamBenchmarkResult,
    DownstreamEvaluationRequest,
    DownstreamEvaluationResult,
)
from dataflow_evolver.core.schemas import TaskSpec
from dataflow_evolver.llm.prompts import _downstream_feedback_block
from dataflow_evolver.main import _build_downstream_evaluator
from dataflow_evolver.utils.config import Config


def _request(tmp_path: Path) -> DownstreamEvaluationRequest:
    return DownstreamEvaluationRequest(
        checkpoint_iteration=3,
        incumbent_iteration=2,
        dataset_path=str(tmp_path / "candidate.jsonl"),
        pipeline_path=str(tmp_path / "pipeline.py"),
        task=TaskSpec("math SFT", {"instruction": "str", "output": "str"}, []),
        expected_benchmarks=["gsm8k", "math500"],
        bad_cases_per_benchmark=3,
    )


def _runner(
    tmp_path: Path,
    *,
    bad_case_count: int = 3,
    incorrect_count: int | None = None,
) -> Path:
    script = tmp_path / "runner.py"
    reported_incorrect = bad_case_count if incorrect_count is None else incorrect_count
    script.write_text(
        f"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
parser.add_argument('--result', required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text(encoding='utf-8'))
assert request['checkpoint_iteration'] == 3
cases = [
    {{'question': f'question {{i}}', 'model_response': f'wrong output {{i}}',
      'parsed_answer': f'wrong parsed answer {{i}}',
      'correct_answer': f'answer {{i}}'}}
    for i in range({bad_case_count})
]
result = {{
    'model': 'Qwen2.5-4B-Base',
    'benchmarks': {{
        'gsm8k': {{'score': 0.61, 'metric': 'accuracy',
                   'incorrect_count': {reported_incorrect}, 'bad_cases': cases}},
        'math500': {{'score': 0.32, 'metric': 'accuracy',
                    'incorrect_count': {reported_incorrect}, 'bad_cases': cases}},
    }},
    'summary': 'algebra and counting remain weak',
    'artifacts': {{'checkpoint': '/external/checkpoint'}},
}}
Path(args.result).write_text(json.dumps(result), encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )
    return script


def test_command_evaluator_uses_json_protocol_and_validates_results(tmp_path):
    evaluator = CommandDownstreamEvaluator(
        [sys.executable, str(_runner(tmp_path))],
        expected_benchmarks=["gsm8k", "math500"],
        bad_cases_per_benchmark=3,
        timeout_sec=30,
    )

    result = evaluator.run(_request(tmp_path), tmp_path / "checkpoint_003")

    assert result.succeeded
    assert result.model == "Qwen2.5-4B-Base"
    assert result.benchmarks["gsm8k"].score == 0.61
    assert result.benchmarks["gsm8k"].bad_cases[0].parsed_answer == (
        "wrong parsed answer 0"
    )
    assert len(result.benchmarks["math500"].bad_cases) == 3
    assert Path(result.request_path).is_file()
    assert Path(result.result_path).is_file()
    assert (tmp_path / "checkpoint_003/stdout.log").is_file()
    request = json.loads(Path(result.request_path).read_text(encoding="utf-8"))
    assert request["protocol_version"] == 3
    assert request["stage"] == "periodic"
    assert request["training_performed"] is True
    assert request["dataset_path"].endswith("candidate.jsonl")


def test_command_evaluator_accepts_fewer_cases_when_fewer_answers_are_wrong(tmp_path):
    evaluator = CommandDownstreamEvaluator(
        [sys.executable, str(_runner(tmp_path, bad_case_count=2))],
        expected_benchmarks=["gsm8k", "math500"],
        bad_cases_per_benchmark=3,
        timeout_sec=30,
    )

    result = evaluator.run(_request(tmp_path), tmp_path / "checkpoint_003")

    assert result.succeeded
    assert result.benchmarks["gsm8k"].incorrect_count == 2
    assert len(result.benchmarks["gsm8k"].bad_cases) == 2


def test_command_evaluator_rejects_missing_cases_when_more_answers_are_wrong(tmp_path):
    evaluator = CommandDownstreamEvaluator(
        [
            sys.executable,
            str(_runner(tmp_path, bad_case_count=2, incorrect_count=8)),
        ],
        expected_benchmarks=["gsm8k", "math500"],
        bad_cases_per_benchmark=3,
        timeout_sec=30,
    )

    result = evaluator.run(_request(tmp_path), tmp_path / "checkpoint_003")

    assert not result.succeeded
    assert "exactly 3 bad_cases" in (result.error or "")


def test_baseline_request_is_eval_only_and_has_no_candidate_paths(tmp_path):
    request = DownstreamEvaluationRequest(
        checkpoint_iteration=0,
        incumbent_iteration=None,
        dataset_path=None,
        pipeline_path=None,
        task=TaskSpec("math SFT", {"instruction": "str", "output": "str"}, []),
        expected_benchmarks=["aime24"],
        bad_cases_per_benchmark=3,
        stage="baseline",
        training_performed=False,
    )

    payload = request.to_dict()

    assert payload["protocol_version"] == 3
    assert payload["checkpoint_iteration"] == 0
    assert payload["training_performed"] is False
    assert payload["dataset_path"] is None
    assert payload["pipeline_path"] is None


def test_prompt_uses_attribution_report_and_never_exposes_raw_bad_cases():
    full_output = "BEGIN-REASONING\n" + ("very long reasoning; " * 1000) + "\nFINAL-ANSWER"
    result = DownstreamEvaluationResult(
        status="ok",
        checkpoint_iteration=3,
        incumbent_iteration=2,
        dataset_path="/run/iteration_002/output.jsonl",
        result_path="/run/downstream_eval/checkpoint_003/result.json",
        model="Qwen2.5-4B-Base",
        benchmarks={
            "gsm8k": DownstreamBenchmarkResult(
                score=0.5,
                metric="accuracy",
                incorrect_count=3,
                wrong_answer_count=0,
                parse_failure_count=0,
                runaway_count=3,
                bad_cases=[
                    DownstreamBadCase(
                        question=f"question {index}",
                        model_response=full_output,
                        parsed_answer=f"parsed {index}",
                        correct_answer=f"answer {index}",
                        response_token_count=16383,
                        finish_reason="length",
                        length_truncated=True,
                    )
                    for index in range(3)
                ],
            )
        },
        attribution=DownstreamAttributionResult(
            status="ok",
            summary="多个样本显示上下文缺失。",
            findings=[
                {
                    "category": "missing_context",
                    "data_side": True,
                    "confidence": 0.86,
                    "evidence_ids": ["gsm8k#1", "gsm8k#2"],
                    "diagnosis": "题干依赖未保留的前置条件。",
                    "recommended_actions": ["过滤上下文不完整的 chunk"],
                }
            ],
            non_data_causes=["部分失败可能来自模型解析器"],
            recommended_pipeline_changes=["保留题干与必要条件"],
        ),
    )

    prompt = _downstream_feedback_block(result)

    assert full_output not in prompt
    assert "下游 bad case 归因报告" in prompt
    assert "多个样本显示上下文缺失" in prompt
    assert "missing_context" in prompt
    assert "gsm8k#1" in prompt
    assert "保留题干与必要条件" in prompt
    assert "/run/downstream_eval/checkpoint_003/result.json" not in prompt
    assert "Question:" not in prompt
    assert "Model response (bounded prompt projection):" not in prompt


def test_downstream_config_is_disabled_without_loading_external_dependencies():
    evaluator, interval = _build_downstream_evaluator(
        Config({"enabled": False, "interval": 3})
    )
    assert evaluator is None
    assert interval == 3


def test_downstream_config_requires_argv_and_benchmark_lists():
    config = Config(
        {
            "enabled": True,
            "interval": 3,
            "command": "python runner.py",
            "benchmarks": ["gsm8k"],
        }
    )
    try:
        _build_downstream_evaluator(config)
    except ValueError as exc:
        assert "YAML argv list" in str(exc)
    else:
        raise AssertionError("string commands must not be passed through a shell")


def test_downstream_config_exposes_baseline_policy():
    evaluator, interval = _build_downstream_evaluator(
        Config(
            {
                "enabled": True,
                "interval": 3,
                "command": [sys.executable, "runner.py"],
                "benchmarks": ["gsm8k", "aime24"],
                "baseline": {"enabled": True},
            }
        )
    )

    assert interval == 3
    assert evaluator is not None
    assert evaluator.baseline_enabled is True
