import importlib.util
import json
import sys
import types
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

_EVAL_SPEC = importlib.util.spec_from_file_location(
    "vllm_eval_standalone",
    ROOT / "scripts" / "vllm_eval_standalone.py",
)
assert _EVAL_SPEC is not None and _EVAL_SPEC.loader is not None
_EVAL_MODULE = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(_EVAL_MODULE)

from math_benchmarks import additional_benchmarks as additional_benchmarks_module
from math_benchmarks.additional_benchmarks import load_chmath, load_gpqa
from math_benchmarks.aime2024 import load_aime2024
from math_benchmarks.gsm8k import load_gsm8k

_score_benchmark = _EVAL_MODULE._score_benchmark


def test_dev_set_is_evaluated_whole_and_loader_never_samples(tmp_path):
    """框架下传的题集必须原样评完，任何规模都不许被脚本改写。

    评多少题是框架按 (source_sha, seed, size) 定死并写进 manifest 的决策。脚本侧曾经并存一套
    MATH_TRAIN_SAMPLE_SIZE 抽样，导致 dev_set.size 调到 500 以上就被静默抽回 500——评的题数
    与 manifest 对不上，"所有节点评在同一批题上"退化成"两个默认值恰好相等"。
    这里用几个跨越旧档位的规模钉住：loader 只读文件，不做任何取舍。
    """
    from math_benchmarks import math_train as math_train_module

    assert not hasattr(math_train_module, "_SAMPLE_SIZE"), "脚本侧不应再有抽样档位"

    for size in (10, 500, 800):
        dev_set = tmp_path / f"dev_{size}.jsonl"
        with dev_set.open("w", encoding="utf-8") as f:
            for i in range(size):
                f.write(json.dumps({"problem": f"q{i}", "answer": str(i)}) + "\n")
        assert len(math_train_module.load_math_train(str(dev_set))) == size


def test_dev_mode_without_dev_set_is_refused_before_any_gpu_work():
    """缺 dev 题集要当场报错，不能回退到"自己挑题"。

    退路一旦存在，框架的 dev 口径就只是建议。而这条链路是先 SFT 再评测，等到评测阶段才发现
    题集没给，一整轮训练已经烧掉了——所以校验必须发生在训练之前。
    """
    import pytest

    with pytest.raises(ValueError, match="--dev-set"):
        _EVAL_MODULE._load_benchmarks("dev", dev_set="")

    # test 模式不受影响：终评走固定的 11 份 benchmark，与 dev 题集无关。
    assert len(_EVAL_MODULE._load_benchmarks("test", dev_set="")) == 11


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


def _no_sympy_grader() -> types.ModuleType:
    """替身 math_benchmarks.grader：真品依赖 latex2sympy2，与 DataFlow 算子用的
    math-verify 争 antlr 版本，装不进本环境（正因如此线上判分路径一直没有单测）。
    选择题判分本就不该走数学等价比较，被调到即判定实现走错了分支。
    """
    stub = types.ModuleType("math_benchmarks.grader")
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("MCQ 判分不应调用 sympy grader")
    stub.check_is_correct = _unexpected
    return stub


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
    monkeypatch.setitem(sys.modules, "math_benchmarks.grader", _no_sympy_grader())

    examples = load_gpqa()
    assert len(examples) == 1
    assert "choices" in examples[0]
    gold_letter = examples[0]["gold_letter"]
    # 走终评真正调用的那条判分路径：MCQ 的 gold_answer 是选项原文（"Blue"），
    # 只有比字母才判得对。此前这里测的是没接线的 gpqa_score，线上路径一直是错的。
    bad_cases: list[dict] = []
    score = _score_benchmark(
        "gpqa", examples, [[f"The answer is {gold_letter}."]], bad_cases, 10
    )
    assert score == 1.0
    assert bad_cases == []

    wrong_letter = next(c for c in "ABCD" if c != gold_letter)
    bad_cases = []
    score = _score_benchmark(
        "gpqa", examples, [[f"The answer is {wrong_letter}."]], bad_cases, 10
    )
    assert score == 0.0
    assert bad_cases[0]["extracted_answer"] == wrong_letter
    assert bad_cases[0]["ground_truth"] == gold_letter


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
