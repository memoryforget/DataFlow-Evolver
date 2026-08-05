"""去污染：n-gram 重合检测本身，以及它在 ReviewAgent 里的扣分/否决行为。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.core import decontamination
from dataflow_evolver.core.schemas import TaskProfile
from dataflow_evolver.llm.serving import LLMServingABC

PROFILE = TaskProfile("t", {"instruction": "str", "answer": "str"}, [])

TEST_Q = (
    "Natalia sold clips to 48 of her friends in April and then she sold half "
    "as many clips in May how many clips did Natalia sell altogether"
)


class StubServing(LLMServingABC):
    def __init__(self, response):
        self.response = response

    def generate(self, prompts, **kwargs):
        return [self.response for _ in prompts]


def _jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def _bench(tmp_path: Path, questions=(TEST_Q,)) -> str:
    return str(_jsonl(tmp_path / "bench" / "test.jsonl", [{"question": q} for q in questions]))


# ---- 索引与扫描 ------------------------------------------------------------
def test_no_reference_files_means_disabled_and_always_clean(tmp_path):
    """没配基准参考集时必须完全无感：扫描恒返回干净，不影响任何既有行为。"""
    index = decontamination.build([])
    assert index.enabled is False
    ds = _jsonl(tmp_path / "ds.jsonl", [{"instruction": TEST_Q, "answer": "72"}])
    report = index.scan(ds)
    assert report.rate == 0.0 and report.hits == 0


def test_verbatim_copy_of_test_item_is_caught(tmp_path):
    index = decontamination.build([_bench(tmp_path)])
    ds = _jsonl(tmp_path / "ds.jsonl", [{"instruction": TEST_Q, "answer": "72"}])
    report = index.scan(ds)
    assert report.hits == 1 and report.total == 1
    assert report.rate == pytest.approx(1.0)
    assert report.examples                      # 命中片段要能带回去给 Agent 定位


def test_surface_edits_do_not_evade_detection(tmp_path):
    """改大小写/标点/空格就能绕过的检测等于没有检测。"""
    index = decontamination.build([_bench(tmp_path)])
    disguised = "NATALIA sold clips to 48 of her friends in April, and then she SOLD half as many clips in May!!  How many clips did Natalia sell altogether?"
    ds = _jsonl(tmp_path / "ds.jsonl", [{"instruction": disguised, "answer": "72"}])
    assert index.scan(ds).hits == 1


def test_unrelated_data_is_clean(tmp_path):
    index = decontamination.build([_bench(tmp_path)])
    ds = _jsonl(tmp_path / "ds.jsonl", [
        {"instruction": "A train travels 60 km in 45 minutes what is its average speed in km per hour", "answer": "80"},
        {"instruction": "Compute the derivative of x squared plus three x with respect to x", "answer": "2x+3"},
    ])
    report = index.scan(ds)
    assert report.hits == 0 and report.rate == 0.0


def test_rate_is_fraction_of_rows(tmp_path):
    index = decontamination.build([_bench(tmp_path)])
    ds = _jsonl(tmp_path / "ds.jsonl", [
        {"instruction": TEST_Q, "answer": "72"},
        {"instruction": "What is the capital city of the country located south of Canada", "answer": "DC"},
    ])
    assert index.scan(ds).rate == pytest.approx(0.5)


def test_short_text_falls_back_to_exact_match(tmp_path):
    """短题目凑不出 13-gram；若不兜底，最容易泄露的短题反而永远检不出来。"""
    index = decontamination.build([_bench(tmp_path, questions=("What is 2 + 2?",))])
    ds = _jsonl(tmp_path / "ds.jsonl", [{"instruction": "what is 2 + 2"}])
    assert index.scan(ds).hits == 1


def test_chinese_text_is_tokenized_by_character(tmp_path):
    """中文没有空格，按空白切词会得到一个巨型 token，n-gram 恒为空。"""
    zh = "某工厂五月份生产零件三千个六月份比五月份多生产两成问六月份生产了多少个零件"
    index = decontamination.build([_bench(tmp_path, questions=(zh,))])
    ds = _jsonl(tmp_path / "ds.jsonl", [{"instruction": zh, "answer": "3600"}])
    assert index.scan(ds).hits == 1


def test_missing_reference_file_warns_instead_of_raising(tmp_path):
    index = decontamination.build([str(tmp_path / "nope.jsonl")])
    assert index.enabled is False               # 建不起来就是不生效，但不该拖垮启动


def test_plain_text_reference_file(tmp_path):
    ref = tmp_path / "bench.txt"
    ref.write_text(TEST_Q + "\n", encoding="utf-8")
    index = decontamination.build([str(ref)])
    ds = _jsonl(tmp_path / "ds.jsonl", [{"instruction": TEST_Q}])
    assert index.scan(ds).hits == 1


# ---- 在 ReviewAgent 里的行为 -----------------------------------------------
_GOOD_REVIEW = json.dumps({
    "schema_score": 1.0, "relevance_score": 1.0, "correctness_score": 1.0,
    "difficulty_score": 1.0, "diversity_score": 1.0, "issues": [],
})


def _review(tmp_path, rows, index):
    ds = _jsonl(tmp_path / "ds.jsonl", rows)
    agent = ReviewAgent(StubServing(_GOOD_REVIEW), contamination=index)
    return agent.review(str(ds), PROFILE)


def test_heavy_contamination_fails_the_gate(tmp_path):
    """泄露不是"质量差一点"，是把评测答案抄进训练集，必须判不通过。"""
    index = decontamination.build([_bench(tmp_path)], max_rate=0.02)
    result = _review(tmp_path, [{"instruction": TEST_Q, "answer": "72"}], index)

    assert result.contamination_rate == pytest.approx(1.0)
    assert result.passed is False
    assert any("数据泄露" in i for i in result.issues)   # 证据要喂回 Agent，否则它会继续这么干


def test_light_contamination_only_discounts_score(tmp_path):
    """低于阈值的疑似重合可能是假阳，按比例扣分即可，不该一票否决。"""
    clean_rows = [
        {"instruction": f"Compute the sum of the first {i} positive odd integers and explain the pattern", "answer": str(i * i)}
        for i in range(2, 102)
    ]
    index = decontamination.build([_bench(tmp_path)], max_rate=0.05)
    result = _review(tmp_path, [{"instruction": TEST_Q, "answer": "72"}] + clean_rows, index)

    assert result.contamination_rate == pytest.approx(1 / 101, abs=1e-6)
    assert result.passed is True
    assert result.review_score < result.llm_composite     # 但确实被扣了分


def test_clean_data_is_not_penalized(tmp_path):
    index = decontamination.build([_bench(tmp_path)])
    result = _review(tmp_path, [{"instruction": "Explain why the sum of two even integers is always even", "answer": "proof"}], index)
    assert result.contamination_rate == 0.0
    assert result.passed is True
    assert result.review_score == pytest.approx(result.llm_composite)
