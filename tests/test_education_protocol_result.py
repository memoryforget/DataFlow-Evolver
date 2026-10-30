import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "downstream_runtime/education/scripts/build_protocol_result.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("education_protocol_result_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finish_prioritizes_wrong_answers_before_parse_and_runaway():
    builder = load_builder()
    candidates = [
        {"parsed_answer": "runaway", "_category": "runaway"},
        {"parsed_answer": "parse", "_category": "parse"},
        {"parsed_answer": "wrong-1", "_category": "wrong"},
        {"parsed_answer": "wrong-2", "_category": "wrong"},
    ]

    result = builder.finish(0.0, "test", candidates, bad_case_count=3)

    assert [item["parsed_answer"] for item in result["bad_cases"]] == [
        "wrong-1",
        "wrong-2",
        "parse",
    ]
    assert all("_category" not in item for item in result["bad_cases"])
    assert result["wrong_answer_count"] == 2
    assert result["parse_failure_count"] == 1
    assert result["runaway_count"] == 1


def test_finish_covers_distinct_dimensions_before_repeating_one():
    builder = load_builder()
    candidates = [
        {"parsed_answer": "a1", "_category": "wrong", "_dimension": "a"},
        {"parsed_answer": "a2", "_category": "wrong", "_dimension": "a"},
        {"parsed_answer": "b1", "_category": "wrong", "_dimension": "b"},
        {"parsed_answer": "c1", "_category": "wrong", "_dimension": "c"},
        {"parsed_answer": "parse", "_category": "parse", "_dimension": "d"},
    ]

    result = builder.finish(0.0, "test", candidates, bad_case_count=3)

    assert [item["parsed_answer"] for item in result["bad_cases"]] == [
        "a1", "b1", "c1",
    ]
    assert [item["dimension"] for item in result["bad_cases"]] == ["a", "b", "c"]


def test_finish_keeps_parse_only_dimension_before_repeating_wrong_dimension():
    builder = load_builder()
    candidates = [
        {"parsed_answer": "a1", "_category": "wrong", "_dimension": "a"},
        {"parsed_answer": "a2", "_category": "wrong", "_dimension": "a"},
        {"parsed_answer": "b1", "_category": "parse", "_dimension": "b"},
        {"parsed_answer": "c1", "_category": "runaway", "_dimension": "c"},
    ]

    result = builder.finish(0.0, "test", candidates, bad_case_count=3)

    assert [item["parsed_answer"] for item in result["bad_cases"]] == [
        "a1", "b1", "c1",
    ]


def test_gaokao_objective_only_score_preserves_objective_result(tmp_path):
    builder = load_builder()
    gaokao_dir = tmp_path / "gaokao"
    gaokao_dir.mkdir()
    payload = {
        "score_kind": "official",
        "objective_only": True,
        "note": "GAOKAO subjective questions are excluded; objective questions only.",
        "metrics": {
            "objective": {
                "total_score": 100,
                "correct_score": 80,
                "question_num": 20,
            },
            "subjective": None,
        },
    }
    (gaokao_dir / "official_metrics.json").write_text(
        builder.json.dumps(payload), encoding="utf-8"
    )
    (tmp_path / "gaokao.jsonl").write_text("", encoding="utf-8")

    result = builder.gaokao_result(tmp_path, 3)

    assert result["score"] == 0.8
    assert result["score_kind"] == "official"
    assert result["objective_only"] is True
    assert result["subjective_coverage"] is None
    assert result["note"] == payload["note"]


def test_mathtutor_bad_cases_cover_all_nine_tasks(tmp_path):
    builder = load_builder()
    mtb = tmp_path / "mathtutorbench_official"
    scores = mtb / "scaffolding_scores"
    scores.mkdir(parents=True)
    metric_paths = {
        "problem_solving": "accuracy_flexible-extract",
        "socratic_questioning": "bleu",
        "solution_correctness": "f1",
        "mistake_location": "f1_micro",
        "mistake_correction": "accuracy",
    }
    reward_tasks = [
        "scaffolding_generation", "scaffolding_generation_hard",
        "pedagogy_following", "pedagogy_following_hard",
    ]
    summary = {task: {metric: 0.5} for task, metric in metric_paths.items()}
    summary.update({task: {"results": {"win_rate": 0.5}} for task in reward_tasks})
    (mtb / "results-model.yaml").write_text(
        builder.yaml.safe_dump(summary), encoding="utf-8"
    )
    audit_rows = {
        "problem_solving": ("1", "2"),
        "socratic_questioning": (["wrong question"], "right question"),
        "solution_correctness": (False, "Yes"),
        "mistake_location": (1, 2),
        "mistake_correction": (1.0, 2.0),
    }
    for task, (prediction, target) in audit_rows.items():
        rows = [{
            "question": f"question-{task}",
            "model_response": f"response-{task}",
            "parsed_prediction": prediction,
            "target": target,
        }]
        (mtb / f"examples-model-{task}.json").write_text(
            builder.json.dumps(rows), encoding="utf-8"
        )
    for task in reward_tasks:
        rows = [{
            "problem": f"question-{task}",
            "generated_teacher_utterance": f"response-{task}",
            "chosen_score": 0.0,
            "rejected_score": 1.0,
        }]
        (scores / f"generations-model-{task}.json_enriched_data.json").write_text(
            builder.json.dumps(rows), encoding="utf-8"
        )

    result = builder.mathtutor_result(tmp_path, 9)

    assert {item["dimension"] for item in result["bad_cases"]} == {
        *metric_paths, *reward_tasks,
    }
    assert len(result["bad_cases"]) == 9
