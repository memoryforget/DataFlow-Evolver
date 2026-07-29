"""Tier-1 ReviewAgent 升级单测：五维复合分、硬指标惩罚、随机蓄水池采样。"""
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
from dataflow_evolver.core.schemas import TaskProfile
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


def _agent(weights=None, hard_metrics=True, seed=0):
    return ReviewAgent(StubServing(["{}"]), weights=weights, hard_metrics=hard_metrics, seed=seed)


def test_composite_uses_fixed_default_weights():
    agent = _agent()
    raw = {
        "correctness_score": 1.0,
        "relevance_score": 1.0,
        "difficulty_score": 1.0,
        "diversity_score": 1.0,
        "schema_score": 1.0,
    }
    r = agent.parse(raw)
    # 权重之和 = 1.0，全 1 → composite = 1.0
    assert r.llm_composite == pytest.approx(1.0)
    # 单独验证加权
    raw2 = {"correctness_score": 1.0, "relevance_score": 0, "difficulty_score": 0,
            "diversity_score": 0, "schema_score": 0}
    assert agent.parse(raw2).llm_composite == pytest.approx(DEFAULT_REVIEW_WEIGHTS["correctness"])


def test_composite_custom_weights():
    agent = _agent(weights={"correctness": 0.5, "relevance": 0.5, "difficulty": 0.0,
                            "diversity": 0.0, "schema": 0.0})
    raw = {"correctness_score": 1.0, "relevance_score": 0.0, "difficulty_score": 1.0,
           "diversity_score": 1.0, "schema_score": 1.0}
    assert agent.parse(raw).llm_composite == pytest.approx(0.5)


def test_clip_out_of_range_scores():
    agent = _agent()
    r = agent.parse({"correctness_score": 5, "relevance_score": -3, "difficulty_score": 0.5,
                     "diversity_score": 0.5, "schema_score": 0.5})
    assert r.correctness_score == 1.0 and r.relevance_score == 0.0


def test_hard_metric_penalty_math(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        '{"instruction":"a","answer":"b"}\n'   # dup 1
        '{"instruction":"a","answer":"b"}\n'   # dup 2（与上重复）
        '{"instruction":"c","answer":""}\n'    # answer 空 → null
        'not-json\n'                             # 不可解析
        , encoding="utf-8",
    )
    dup, null, parse_valid, total = _scan_hard_metrics(str(ds), ["instruction", "answer"])
    assert total == 4
    assert parse_valid == pytest.approx(0.75)        # 3/4 可解析
    assert dup == pytest.approx(0.25)                # unique=3 / total=4
    assert null == pytest.approx(1 / 6, abs=1e-6)    # 空槽 1 / (3 行 × 2 字段)


def test_apply_hard_metrics_tier1(tmp_path):
    ds = tmp_path / "d.jsonl"
    # 全部合法、无重复、字段齐全 → 无惩罚，tier1 == composite
    ds.write_text(
        '{"instruction":"q1","answer":"a1"}\n'
        '{"instruction":"q2","answer":"a2"}\n', encoding="utf-8",
    )
    profile = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")
    agent = _agent()
    r = agent.parse({"correctness_score": 0.8, "relevance_score": 0.8, "difficulty_score": 0.8,
                     "diversity_score": 0.8, "schema_score": 0.8})
    agent.apply_hard_metrics(r, str(ds), profile)
    assert r.dup_rate == 0.0 and r.null_rate == 0.0 and r.parse_valid_rate == 1.0
    assert r.tier1_score == pytest.approx(r.llm_composite)
    assert r.tier1_score == pytest.approx(0.8)


def test_tier1_penalized_by_dup_and_null(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        '{"instruction":"a","answer":"x"}\n'
        '{"instruction":"a","answer":"x"}\n'   # 重复
        '{"instruction":"b","answer":""}\n', encoding="utf-8",
    )
    profile = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")
    agent = _agent()
    r = agent.parse({"correctness_score": 1.0, "relevance_score": 1.0, "difficulty_score": 1.0,
                     "diversity_score": 1.0, "schema_score": 1.0})
    agent.apply_hard_metrics(r, str(ds), profile)
    expected = r.llm_composite * (1 - r.dup_rate) * (1 - r.null_rate) * r.parse_valid_rate
    assert r.tier1_score == pytest.approx(expected)
    assert r.tier1_score < r.llm_composite   # 有惩罚


def test_empty_dataset_tier1_zero(tmp_path):
    ds = tmp_path / "empty.jsonl"
    ds.write_text("", encoding="utf-8")
    profile = TaskProfile("t", {"instruction": "str"}, [], "sft")
    agent = _agent()
    r = agent.parse({"correctness_score": 1.0, "relevance_score": 1.0, "difficulty_score": 1.0,
                     "diversity_score": 1.0, "schema_score": 1.0})
    agent.apply_hard_metrics(r, str(ds), profile)
    assert r.tier1_score == 0.0


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


def test_review_end_to_end_tier1_exposed(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text('{"instruction":"q","answer":"a"}\n', encoding="utf-8")
    profile = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")
    resp = ('{"correctness_score":0.9,"relevance_score":0.8,"difficulty_score":0.7,'
            '"diversity_score":0.6,"schema_score":1.0,"issues":[]}')
    agent = ReviewAgent(StubServing([resp]), seed=0)
    r = agent.review(str(ds), profile)
    assert r.passed is True
    assert 0.0 < r.tier1_score <= 1.0
    assert r.tier1_score == pytest.approx(r.llm_composite)   # 单行、无重复、字段齐 → 无惩罚


# --------------------------- F1-F7 失败信号（借鉴 ANDES） ---------------------------
def test_normalize_signals_filters_and_dedups():
    # 合法编号大写去重保序；非法/越界的丢弃
    assert _normalize_signals(["F3", "f5", "F3", "F9", "junk", 3]) == ["F3", "F5"]
    assert _normalize_signals(None) == []
    assert _normalize_signals([]) == []


def test_parse_folds_signals_into_issues():
    agent = _agent()
    r = agent.parse({
        "correctness_score": 0.8, "relevance_score": 0.8, "difficulty_score": 0.8,
        "diversity_score": 0.8, "schema_score": 0.8,
        "failure_signals": ["F3", "f5", "bogus"],
        "issues": ["答案与推理不一致"],
    })
    assert r.failure_signals == ["F3", "F5"]
    # 原有 issue 保留，且失败信号以 [Fx] 描述 追加进 issues
    assert "答案与推理不一致" in r.issues
    assert any(s.startswith("[F3]") and FAILURE_SIGNALS["F3"] in s for s in r.issues)
    assert any(s.startswith("[F5]") for s in r.issues)
    # 打分不受影响（仍是五维加权）
    assert r.llm_composite == pytest.approx(0.8)


def test_parse_no_signals_leaves_issues_untouched():
    agent = _agent()
    r = agent.parse({
        "correctness_score": 1.0, "relevance_score": 1.0, "difficulty_score": 1.0,
        "diversity_score": 1.0, "schema_score": 1.0, "issues": [],
    })
    assert r.failure_signals == []
    assert r.issues == []


def test_review_prompt_lists_all_signals():
    profile = TaskProfile("t", {"instruction": "str"}, ["q1"], "sft")
    prompt = build_review_prompt(profile, '{"instruction":"x"}')
    for sid in FAILURE_SIGNALS:
        assert sid in prompt
    assert "failure_signals" in prompt
