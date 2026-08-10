"""ReviewAgent 单测：四维样本复合分、硬指标惩罚、随机蓄水池采样。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.agents.review_agent import (
    DEFAULT_REVIEW_WEIGHTS,
    ReviewAgent,
    _normalize_signals,
    _sample_jsonl,
    _scan_hard_metrics,
)
from dataflow_evolver.core.schemas import TaskSpec
from dataflow_evolver.llm.prompts import FAILURE_SIGNALS, build_review_prompt
from dataflow_evolver.llm.serving import LLMServingABC


class StubServing(LLMServingABC):
    def __init__(self, responses):
        self._responses = responses
        self._i = 0

    def generate(self, prompts, **kwargs):
        out = []
        for _ in prompts:
            out.append(self._responses[min(self._i, len(self._responses) - 1)])
            self._i += 1
        return out


def _agent(weights=None, hard_metrics=True, seed=0, **kwargs):
    return ReviewAgent(
        StubServing(["{}"]),
        weights=weights,
        hard_metrics=hard_metrics,
        seed=seed,
        **kwargs,
    )


def test_composite_uses_fixed_default_weights():
    agent = _agent()
    raw = {
        "correctness_score": 1.0,
        "relevance_score": 1.0,
        "difficulty_score": 1.0,
        "schema_score": 1.0,
    }
    r = agent.parse(raw)
    # 权重之和 = 1.0，全 1 → composite = 1.0
    assert r.llm_composite == pytest.approx(1.0)
    # 单独验证加权
    raw2 = {"correctness_score": 1.0, "relevance_score": 0, "difficulty_score": 0,
            "schema_score": 0}
    assert agent.parse(raw2).llm_composite == pytest.approx(DEFAULT_REVIEW_WEIGHTS["correctness"])


def test_composite_custom_weights():
    agent = _agent(weights={"correctness": 0.5, "relevance": 0.5, "difficulty": 0.0,
                            "schema": 0.0})
    raw = {"correctness_score": 1.0, "relevance_score": 0.0, "difficulty_score": 1.0,
           "schema_score": 1.0}
    assert agent.parse(raw).llm_composite == pytest.approx(0.5)


def test_clip_out_of_range_scores():
    agent = _agent()
    r = agent.parse({"correctness_score": 5, "relevance_score": -3, "difficulty_score": 0.5,
                     "schema_score": 0.5})
    assert r.correctness_score == 1.0 and r.relevance_score == 0.0


def test_passed_requires_configured_correctness_threshold():
    raw = {
        "correctness_score": 0.59,
        "relevance_score": 1.0,
        "difficulty_score": 1.0,
        "schema_score": 1.0,
    }
    assert _agent().parse(raw).passed is False

    raw["correctness_score"] = 0.6
    assert _agent().parse(raw).passed is True

    assert _agent(correctness_threshold=0.8).parse(raw).passed is False


def test_hard_metric_penalty_math(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        '{"instruction":"a","answer":"b"}\n'   # dup 1
        '{"instruction":"a","answer":"b"}\n'   # dup 2（与上重复）
        '{"instruction":"c","answer":""}\n'    # answer 空 → null
        'not-json\n'                             # 不可解析
        , encoding="utf-8",
    )
    m = _scan_hard_metrics(str(ds), ["instruction", "answer"])
    assert m.total == 4
    assert m.parse_valid_rate == pytest.approx(0.75)     # 3/4 可解析
    assert m.dup_rate == pytest.approx(0.25)             # unique=3 / total=4
    assert m.null_rate == pytest.approx(1 / 6, abs=1e-6)  # 空槽 1 / (3 行 × 2 字段)


def _long(text: str) -> str:
    """凑到内容字段门槛之上，同时保持不同 text 之间内容不同。"""
    return f"{text} " + "padding text to push this field above the content-field length threshold."


def test_field_dup_catches_same_question_with_varied_cot(tmp_path):
    """整记录哈希看不见「同一道题配不同 CoT」，字段级必须抓到。"""
    ds = tmp_path / "d.jsonl"
    rows = [
        {"question": _long("同一道题"), "cot": _long(f"推理链 {i}")}
        for i in range(3)
    ] + [
        {"question": _long(f"独立题目 {i}"), "cot": _long(f"推理链 {i}")}
        for i in range(3, 5)
    ]
    ds.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    m = _scan_hard_metrics(str(ds), ["question", "cot"])
    assert m.record_dup_rate == 0.0          # 5 行各不相同
    assert m.field_dup_rate == pytest.approx(0.4)   # question 唯一 3/5
    assert m.dup_field == "question"
    assert m.dup_rate == pytest.approx(0.4)


def test_configured_duplicate_fields_ignore_fixed_instruction(tmp_path):
    ds = tmp_path / "fixed_instruction.jsonl"
    fixed = _long("Solve the following grade-school math problem.")
    rows = [
        {
            "instruction": fixed,
            "input": _long(f"Distinct problem {index}"),
            "output": _long(f"Distinct solution {index}"),
        }
        for index in range(5)
    ]
    ds.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    default_metrics = _scan_hard_metrics(
        str(ds), ["instruction", "input", "output"]
    )
    configured_metrics = _scan_hard_metrics(
        str(ds),
        ["instruction", "input", "output"],
        duplicate_fields=["input", "output"],
    )

    assert default_metrics.dup_field == "instruction"
    assert default_metrics.field_dup_rate == pytest.approx(0.8)
    assert configured_metrics.field_dup_rate == 0.0
    assert configured_metrics.dup_rate == 0.0


def test_short_label_fields_do_not_inflate_dup(tmp_path):
    """answer/difficulty 这类短标签天然重复，纳入会把每份数据都判成重灾区。"""
    ds = tmp_path / "d.jsonl"
    rows = [
        {"question": _long(f"题目 {i}"), "answer": "42", "difficulty": "easy"}
        for i in range(5)
    ]
    ds.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    m = _scan_hard_metrics(str(ds), ["question", "answer", "difficulty"])
    assert m.field_dup_rate == 0.0   # answer 唯一率只有 1/5，但它不是内容字段
    assert m.dup_rate == 0.0


def test_record_dup_still_counted_without_content_fields(tmp_path):
    """字段全是短标签时退化回整记录口径，不能因为跳过字段级就漏报。"""
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        '{"q":"1+1","a":"2"}\n'
        '{"q":"1+1","a":"2"}\n'
        '{"q":"2+2","a":"4"}\n', encoding="utf-8",
    )
    m = _scan_hard_metrics(str(ds), ["q", "a"])
    assert m.field_dup_rate == 0.0
    assert m.dup_rate == pytest.approx(1 / 3, abs=1e-6)


def test_non_string_field_values_do_not_crash(tmp_path):
    """字段值可能是 dict/list/数字/缺失，折成文本后照常参与去重。"""
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        '{"messages":[{"role":"user","content":"x"}],"n":1}\n'
        '{"messages":[{"role":"user","content":"x"}],"n":2}\n'
        '{"n":3}\n', encoding="utf-8",
    )
    m = _scan_hard_metrics(str(ds), ["messages", "n"])
    assert m.total == 3
    assert 0.0 <= m.dup_rate <= 1.0


def test_apply_hard_metrics_review_score(tmp_path):
    ds = tmp_path / "d.jsonl"
    # 全部合法、无重复、字段齐全 → 无惩罚，review_score == composite
    ds.write_text(
        '{"instruction":"q1","answer":"a1"}\n'
        '{"instruction":"q2","answer":"a2"}\n', encoding="utf-8",
    )
    task = TaskSpec("t", {"instruction": "str", "answer": "str"}, [])
    agent = _agent()
    r = agent.parse({"correctness_score": 0.8, "relevance_score": 0.8, "difficulty_score": 0.8,
                     "schema_score": 0.8})
    agent.apply_hard_metrics(r, str(ds), task)
    assert r.dup_rate == 0.0 and r.null_rate == 0.0 and r.parse_valid_rate == 1.0
    assert r.review_score == pytest.approx(r.llm_composite)
    assert r.review_score == pytest.approx(0.8)


def test_review_score_penalized_by_dup_and_null(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        '{"instruction":"a","answer":"x"}\n'
        '{"instruction":"a","answer":"x"}\n'   # 重复
        '{"instruction":"b","answer":""}\n', encoding="utf-8",
    )
    task = TaskSpec("t", {"instruction": "str", "answer": "str"}, [])
    agent = _agent()
    r = agent.parse({"correctness_score": 1.0, "relevance_score": 1.0, "difficulty_score": 1.0,
                     "schema_score": 1.0})
    agent.apply_hard_metrics(r, str(ds), task)
    expected = r.llm_composite * (1 - r.dup_rate) * (1 - r.null_rate) * r.parse_valid_rate
    assert r.review_score == pytest.approx(expected)
    assert r.review_score < r.llm_composite   # 有惩罚


def test_empty_dataset_review_score_zero(tmp_path):
    ds = tmp_path / "empty.jsonl"
    ds.write_text("", encoding="utf-8")
    task = TaskSpec("t", {"instruction": "str"}, [])
    agent = _agent()
    r = agent.parse({"correctness_score": 1.0, "relevance_score": 1.0, "difficulty_score": 1.0,
                     "schema_score": 1.0})
    agent.apply_hard_metrics(r, str(ds), task)
    assert r.review_score == 0.0


def test_reservoir_sampling_deterministic_and_subset(tmp_path):
    ds = tmp_path / "big.jsonl"
    lines = [f'{{"i":{i}}}' for i in range(100)]
    ds.write_text("\n".join(lines) + "\n", encoding="utf-8")

    a = _sample_jsonl(str(ds), 10, seed=42)
    b = _sample_jsonl(str(ds), 10, seed=42)
    assert a == b                                   # 同种子确定性
    sampled = a.split("\n")
    assert len(sampled) == 10
    assert all(s in lines for s in sampled)          # 都来自原文件
    # 不同种子通常给出不同样本（证明是随机而非取前 N）
    c = _sample_jsonl(str(ds), 10, seed=7)
    assert set(sampled) != set(range(10)) or set(c.split("\n")) != set(a.split("\n"))


def test_reservoir_returns_all_when_n_ge_count(tmp_path):
    ds = tmp_path / "small.jsonl"
    ds.write_text('{"i":0}\n{"i":1}\n{"i":2}\n', encoding="utf-8")
    out = _sample_jsonl(str(ds), 10, seed=0).split("\n")
    assert len(out) == 3


def test_reservoir_not_prefix_biased(tmp_path):
    """蓄水池采样应能抽到文件后半部分（旧的取前 N 行做不到）。"""
    ds = tmp_path / "ordered.jsonl"
    ds.write_text("\n".join(f'{{"i":{i}}}' for i in range(1000)) + "\n", encoding="utf-8")
    sampled = _sample_jsonl(str(ds), 20, seed=123).split("\n")
    idxs = [int(s.split(":")[1].rstrip("}")) for s in sampled]
    assert max(idxs) >= 100    # 至少抽到前 100 之外的样本


def test_review_end_to_end_review_score_exposed(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text('{"instruction":"q","answer":"a"}\n', encoding="utf-8")
    task = TaskSpec("t", {"instruction": "str", "answer": "str"}, [])
    resp = ('{"correctness_score":0.9,"relevance_score":0.8,"difficulty_score":0.7,'
            '"schema_score":1.0,"issues":[]}')
    agent = ReviewAgent(StubServing([resp]), seed=0)
    r = agent.review(str(ds), task)
    assert r.passed is True
    assert 0.0 < r.review_score <= 1.0
    assert r.review_score == pytest.approx(r.llm_composite)   # 单行、无重复、字段齐 → 无惩罚


# --------------------------- F1-F7 失败信号（借鉴 ANDES） ---------------------------
def test_normalize_signals_filters_and_dedups():
    # 合法编号大写去重保序；非法/越界的丢弃
    assert _normalize_signals(["F3", "f5", "F3", "F9", "junk", 3]) == ["F3", "F5"]
    assert _normalize_signals(None) == []
    assert _normalize_signals([]) == []


def test_parse_keeps_signals_separate_from_issues():
    """信号与 issues 是异质数据，必须分开保存。

    曾把 "[Fx] 描述" 追加进 issues，结果每个消费点都要再过滤回来（共享过滤函数 + 属性 +
    四处调用 + 一处"有意的不对称"）。信号文本由需要它的渲染点按编号查 FAILURE_SIGNALS 拼。
    """
    agent = _agent()
    r = agent.parse({
        "correctness_score": 0.8, "relevance_score": 0.8, "difficulty_score": 0.8,
        "schema_score": 0.8,
        "failure_signals": ["F3", "f5", "bogus"],
        "issues": ["答案与推理不一致"],
    })
    assert r.failure_signals == ["F3", "F5"]
    assert r.issues == ["答案与推理不一致"]                  # 未被信号污染
    assert not any(s.startswith("[F") for s in r.issues)
    assert FAILURE_SIGNALS["F3"] not in "".join(r.issues)
    # 打分不受影响（仍是四维加权）
    assert r.llm_composite == pytest.approx(0.8)


def test_parse_no_signals_leaves_issues_untouched():
    agent = _agent()
    r = agent.parse({
        "correctness_score": 1.0, "relevance_score": 1.0, "difficulty_score": 1.0,
        "schema_score": 1.0, "issues": [],
    })
    assert r.failure_signals == []
    assert r.issues == []


def test_review_prompt_lists_all_signals():
    task = TaskSpec("t", {"instruction": "str"}, ["q1"])
    prompt = build_review_prompt(task, '{"instruction":"x"}')
    for sid in FAILURE_SIGNALS:
        assert sid in prompt
    assert "failure_signals" in prompt
    assert "diversity_score" not in prompt
    assert "dataset embedding" in prompt
