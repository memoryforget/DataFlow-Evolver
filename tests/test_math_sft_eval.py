import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "llamafactory_math_sft_eval",
    ROOT / "scripts" / "llamafactory_math_sft_eval.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

from math_benchmarks import additional_benchmarks as additional_benchmarks_module
from math_benchmarks.additional_benchmarks import gpqa_score, load_chmath, load_gpqa
from math_benchmarks.aime2024 import load_aime2024
from math_benchmarks.gsm8k import load_gsm8k

TEST_LARGE_BENCHMARKS = _MODULE.TEST_LARGE_BENCHMARKS
TEST_SMALL_BENCHMARKS = _MODULE.TEST_SMALL_BENCHMARKS
EVAL_SYSTEM_PROMPT_BOXED = _MODULE.EVAL_SYSTEM_PROMPT_BOXED
_build_remote_worker_ssh_cmd = _MODULE._build_remote_worker_ssh_cmd
_should_launch_remote_worker = _MODULE._should_launch_remote_worker
_system_prompt_for = _MODULE._system_prompt_for
_user_prompt_for = _MODULE._user_prompt_for


def test_test_benchmark_suite_matches_expected_contract():
    assert TEST_SMALL_BENCHMARKS == ("AIME2024", "AMC23", "CHMath")
    assert TEST_LARGE_BENCHMARKS == (
        "GSM8K",
        "MATH500",
        "OlympiadBench",
        "Gaokao",
        "Kaoyan",
        "GradeSchool",
        "MinervaMath",
        "GPQA",
    )


def test_user_prompt_appends_choices():
    prompt = _user_prompt_for(
        {
            "question": "Which one is correct?",
            "choices": [("A", "first"), ("B", "second")],
        },
        lambda question: question,
    )
    assert "Options:" in prompt
    assert "A. first" in prompt
    assert "B. second" in prompt


def test_eval_prompt_contract_matches_limo():
    assert EVAL_SYSTEM_PROMPT_BOXED == (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    assert _system_prompt_for("GPQA", {"question": "q"}) == EVAL_SYSTEM_PROMPT_BOXED
    assert _system_prompt_for("GSM8K", {"question": "q"}) == EVAL_SYSTEM_PROMPT_BOXED


def test_should_launch_remote_worker_depends_on_rank_and_nnodes(monkeypatch):
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("NODE_RANK", "0")
    assert _should_launch_remote_worker() is True

    monkeypatch.setenv("NODE_RANK", "1")
    assert _should_launch_remote_worker() is False

    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("NODE_RANK", "0")
    assert _should_launch_remote_worker() is False


def test_build_remote_worker_ssh_cmd_contains_expected_env_and_args(tmp_path, monkeypatch):
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text('{"instruction":"q","answer":"a"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("NODE_RANK", "0")
    monkeypatch.setenv("NODE1_HOST", "user@10.0.0.2")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("SFT_NUM_GPUS", "16")
    monkeypatch.setenv("REMOTE_CWD", "/shared/DataFlow-Evolve")
    monkeypatch.setenv("SSH_OPTS", "-p 2222")
    monkeypatch.setenv("SFT_REMOTE_PYTHON", "/opt/conda/envs/df-train/bin/python")

    cmd = _build_remote_worker_ssh_cmd(dataset, output_dir, "test")
    assert cmd[0] == "ssh"
    assert "-p" in cmd and "2222" in cmd
    assert "user@10.0.0.2" in cmd
    remote_str = cmd[-1]
    assert "NODE_RANK=1" in remote_str
    assert "MASTER_ADDR=10.0.0.1" in remote_str
    assert "NNODES=2" in remote_str
    assert "--mode" in remote_str and "test" in remote_str
    assert str(dataset.resolve()) in remote_str


def test_project_chmath_loader_uses_fixed_repo_layout(tmp_path, monkeypatch):
    chmath_dir = tmp_path / "cn_math_2024"
    chmath_dir.mkdir(parents=True)
    (chmath_dir / "test.jsonl").write_text(
        json.dumps({"question": "1+1=?", "answer": "2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(additional_benchmarks_module, "_PROJECT_LIMO_EVAL_DATA_DIR", tmp_path)
    examples = load_chmath()
    assert len(examples) == 1
    assert examples[0]["question"] == "1+1=?"
    assert examples[0]["gold_answer"] == "2"


def test_project_gpqa_loader_builds_choices_and_scores_letter_answer(tmp_path, monkeypatch):
    gpqa_dir = tmp_path / "gpqa"
    gpqa_dir.mkdir(parents=True)
    row = {
        "Question": "What color is the daytime sky on Earth?",
        "Correct Answer": "Blue",
        "Incorrect Answer 1": "Green",
        "Incorrect Answer 2": "Red",
        "Incorrect Answer 3": "Yellow",
    }
    (gpqa_dir / "test.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(additional_benchmarks_module, "_PROJECT_LIMO_EVAL_DATA_DIR", tmp_path)
    examples = load_gpqa()
    assert len(examples) == 1
    assert "choices" in examples[0]
    gold_letter = examples[0]["gold_letter"]
    score, bad_cases = gpqa_score([f"The answer is {gold_letter}."], examples)
    assert score == 1.0
    assert bad_cases == []


def test_project_benchmark_dir_autoloads_chmath_and_aime(tmp_path, monkeypatch):
    chmath_dir = tmp_path / "cn_math_2024"
    chmath_dir.mkdir(parents=True)
    (chmath_dir / "test.jsonl").write_text(
        json.dumps({"problem": "1+1=?", "answer": "2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    aime_dir = tmp_path / "aime"
    aime_dir.mkdir(parents=True)
    (aime_dir / "test.jsonl").write_text(
        json.dumps({"question": "2+3=?", "answer": "5"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(additional_benchmarks_module, "_PROJECT_LIMO_EVAL_DATA_DIR", tmp_path)
    chmath_examples = load_chmath()
    aime_examples = load_aime2024()
    assert chmath_examples[0]["gold_answer"] == "2"
    assert aime_examples[0]["gold_answer"] == "5"


def test_gsm8k_loader_supports_project_jsonl(tmp_path, monkeypatch):
    gsm8k_dir = tmp_path / "gsm8k"
    gsm8k_dir.mkdir(parents=True)
    (gsm8k_dir / "test.jsonl").write_text(
        json.dumps({"question": "3+4=?", "answer": "reasoning #### 7"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(additional_benchmarks_module, "_PROJECT_LIMO_EVAL_DATA_DIR", tmp_path)
    examples = load_gsm8k()
    assert examples == [{"question": "3+4=?", "gold_answer": "7"}]
