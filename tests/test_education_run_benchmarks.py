import importlib.util
import json
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "downstream_runtime/education/scripts/run_benchmarks.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("education_benchmark_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gaokao_keeps_official_prefix_separate_from_user_question(tmp_path):
    runner = load_runner()
    keyword = "2010-2022_Math_I_MCQs"
    data_dir = tmp_path / "benchmarks/gaokao/Data/Objective_Questions"
    data_dir.mkdir(parents=True)
    (data_dir / f"{keyword}.json").write_text(
        json.dumps(
            {"example": [{"index": 7, "question": "What is 1 + 1?", "answer": "B"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runner.ROOT = tmp_path
    runner.DATA_SOURCE = "official"

    rows = runner.load_gaokao()

    assert len(rows) == 1
    assert rows[0]["prompt"] == "What is 1 + 1?"
    assert rows[0]["official"]["_system_prompt"] == runner._GAOKAO_OBJ_PREFIX[keyword]
    assert runner._GAOKAO_OBJ_PREFIX[keyword] not in rows[0]["prompt"]


def test_gaokao_loader_excludes_subjective_questions(tmp_path):
    runner = load_runner()
    root = tmp_path / "benchmarks/gaokao/Data"
    objective = root / "Objective_Questions"
    subjective = root / "Subjective_Questions"
    objective.mkdir(parents=True)
    subjective.mkdir(parents=True)
    row = {"example": [{"index": 1, "question": "objective", "answer": "A"}]}
    (objective / "2010-2022_Math_I_MCQs.json").write_text(
        json.dumps(row), encoding="utf-8"
    )
    (subjective / "2010-2022_Physics.json").write_text(
        json.dumps({"example": [{"index": 2, "question": "subjective"}]}),
        encoding="utf-8",
    )
    runner.ROOT = tmp_path
    runner.DATA_SOURCE = "official"

    rows = runner.load_gaokao()

    assert len(rows) == 1
    assert rows[0]["official"]["question_type"] == "Objective_Questions"
