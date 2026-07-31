"""评审证据落盘 + Tier-2 重评回写。

结构化终态在 checkpoint、过程在 run.log，但抽样与 LLM 原始响应此前用完即弃，事后无法核验
tier1 是否可信、也无法离线换权重复算。Tier-2 的重评更是算完即弃——而它与 train.score 出自
同一份放大数据，是唯一能检验「tier1 能否预测下游真分」的样本对。
"""
from __future__ import annotations

import json
from pathlib import Path

from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.core import checkpoint
from dataflow_evolver.core.baseline import RewardPolicy
from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult, TaskProfile
from dataflow_evolver.core.tree_manager import TreeManager

PROFILE = TaskProfile("QA", {"q": "str", "a": "str"}, [], "sft")

GOOD_JSON = json.dumps(
    {
        "schema_score": 0.9,
        "relevance_score": 0.8,
        "correctness_score": 0.7,
        "difficulty_score": 0.6,
        "diversity_score": 0.5,
        "passed": True,
        "issues": ["难度偏低"],
        "failure_signals": ["F3"],
    },
    ensure_ascii=False,
)


class _FakeServing:
    """按序返回预设响应，并记录收到的 prompt。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate_one(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else GOOD_JSON


def _dataset(tmp_path: Path, n: int = 5) -> str:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps({"q": f"问题{i}", "a": str(i)}, ensure_ascii=False) for i in range(n)),
        encoding="utf-8",
    )
    return str(path)


def _agent(responses: list[str]) -> ReviewAgent:
    return ReviewAgent(_FakeServing(responses), sample_size=5, seed=0)


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---- 证据落盘 --------------------------------------------------------------
def test_evidence_not_written_without_path(tmp_path):
    """不传 evidence_path 时行为完全不变（向后兼容旧调用）。"""
    _agent([GOOD_JSON]).review(_dataset(tmp_path), PROFILE)
    assert list(tmp_path.glob("*.jsonl")) == [tmp_path / "data.jsonl"]


def test_evidence_records_samples_raw_and_parsed(tmp_path):
    evidence = tmp_path / "node_1" / "review.jsonl"
    result = _agent([GOOD_JSON]).review(
        _dataset(tmp_path), PROFILE, evidence_path=str(evidence), phase="search"
    )

    (rec,) = _read_records(evidence)
    assert rec["phase"] == "search"
    assert rec["raw_response"] == GOOD_JSON
    assert rec["parsed"]["difficulty_score"] == 0.6
    assert rec["result"]["tier1_score"] == result.tier1_score
    assert rec["result"]["failure_signals"] == ["F3"]
    assert rec["sample_size"] == 5 and rec["seed"] == 0
    assert "correctness" in rec["weights"]


def test_evidence_samples_are_parseable_jsonl(tmp_path):
    """抽样字段存的是 LLM 实际看到的原文，本身即合法 JSONL，可直接离线复算。"""
    evidence = tmp_path / "review.jsonl"
    _agent([GOOD_JSON]).review(_dataset(tmp_path, n=4), PROFILE, evidence_path=str(evidence))

    (rec,) = _read_records(evidence)
    rows = [json.loads(line) for line in rec["samples_jsonl"].splitlines()]
    assert len(rows) == 4
    assert {"q", "a"} == set(rows[0])


def test_evidence_appends_across_reviews(tmp_path):
    """同一节点多次评审必须追加，否则丢掉自纠轨迹（哪一维改后真的涨了）。"""
    evidence = tmp_path / "review.jsonl"
    agent = _agent([GOOD_JSON, GOOD_JSON])
    dataset = _dataset(tmp_path)
    agent.review(dataset, PROFILE, evidence_path=str(evidence), phase="search")
    agent.review(dataset, PROFILE, evidence_path=str(evidence), phase="search_recheck_1")

    phases = [r["phase"] for r in _read_records(evidence)]
    assert phases == ["search", "search_recheck_1"]


def test_evidence_written_when_llm_returns_invalid_json(tmp_path):
    """LLM 没给出有效 JSON 时更需要留证：原始响应必须落盘，parsed 为 None。"""
    evidence = tmp_path / "review.jsonl"
    result = _agent(["抱歉，我无法完成", "还是不行", "依旧不行"]).review(
        _dataset(tmp_path), PROFILE, evidence_path=str(evidence)
    )

    (rec,) = _read_records(evidence)
    assert rec["parsed"] is None
    assert rec["raw_response"] == "依旧不行"          # 重试耗尽后的最后一次原文
    assert result.passed is False


def test_evidence_written_for_empty_dataset(tmp_path):
    evidence = tmp_path / "review.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    _agent([]).review(str(empty), PROFILE, evidence_path=str(evidence))

    (rec,) = _read_records(evidence)
    assert rec["samples_jsonl"] == ""
    assert rec["result"]["issues"] == ["数据集为空或不可读"]


def test_evidence_failure_does_not_break_review(tmp_path):
    """落盘出错只告警，不能影响评审主流程（这里让父目录落在一个普通文件下，mkdir 必失败）。"""
    dataset = _dataset(tmp_path)
    unwritable = str(Path(dataset) / "sub" / "review.jsonl")

    result = _agent([GOOD_JSON]).review(dataset, PROFILE, evidence_path=unwritable)
    assert result.tier1_score > 0


# ---- Tier-2 重评回写 -------------------------------------------------------
def _review(tier1: float) -> ReviewResult:
    return ReviewResult(
        schema_score=0.8, relevance_score=0.8, passed=True, tier1_score=tier1
    )


def _cfg() -> PipelineConfig:
    return PipelineConfig(operators=[{"name": "Gen"}], field_flow="f", code="x", rationale="")


def _serialize(tree: TreeManager) -> dict:
    return checkpoint.serialize(
        tree, RewardPolicy(), None, completed_iterations=1, profile=PROFILE
    )


def test_tier2_review_defaults_to_none_and_is_separate_field():
    tree = TreeManager()
    node = tree.root.add_child(_cfg())
    assert node.tier2_review_result is None

    node.review_result = _review(0.40)                 # 搜索期探针小样本
    node.tier2_review_result = _review(0.31)           # 放大后全量数据
    assert node.review_result.tier1_score == 0.40      # 互不覆盖：两个分都要留着比对
    assert node.tier2_review_result.tier1_score == 0.31


def test_checkpoint_roundtrips_tier2_review():
    tree = TreeManager()
    node = tree.root.add_child(_cfg())
    node.review_result = _review(0.40)
    node.tier2_review_result = _review(0.31)

    restored = TreeManager()
    checkpoint.restore_tree(_serialize(tree), restored)

    got = next(n for n in restored.root.children if n.id == node.id)
    assert got.tier2_review_result is not None
    assert got.tier2_review_result.tier1_score == 0.31
    assert got.review_result.tier1_score == 0.40


def test_restore_tolerates_checkpoint_without_tier2_key():
    """旧 checkpoint 没有该键，续跑不应报错。"""
    tree = TreeManager()
    node = tree.root.add_child(_cfg())
    node.review_result = _review(0.40)

    state = _serialize(tree)
    for nd in state["nodes"]:
        nd.pop("tier2_review_result", None)

    restored = TreeManager()
    checkpoint.restore_tree(state, restored)
    got = next(n for n in restored.root.children if n.id == node.id)
    assert got.tier2_review_result is None
