"""GlobalMemory 的 Tier-1 信号透传。

reward_source=review 时搜索期不训练，score 恒为 None、诊断恒为 None，因此 ReviewAgent 的
tier1/硬指标/失败信号/issues 是唯一可跨节点传递的质量信号，必须进入记忆并被渲染出来，
否则下游节点只能看到"别人试过什么结构"，无法判断好坏。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.core.global_memory import GlobalMemory  # noqa: E402
from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult, TrainResult  # noqa: E402
from dataflow_evolver.core.tree_node import PipelineNode  # noqa: E402


def _cfg(flow: str = "a -> b", op: str = "OpA") -> PipelineConfig:
    return PipelineConfig(
        operators=[{"name": op, "params": {}}],
        field_flow=flow,
        code="# code",
        rationale="r",
    )


def _review(tier1: float, *, passed: bool = True, signals=None, issues=None) -> ReviewResult:
    r = ReviewResult(
        schema_score=0.9,
        relevance_score=0.9,
        passed=passed,
        issues=list(issues or []),
        dup_rate=0.12,
        null_rate=0.0,
        parse_valid_rate=1.0,
        failure_signals=list(signals or []),
    )
    r.tier1_score = tier1
    return r


def _node(node_id: int, *, tier1=None, score=None, **review_kw) -> PipelineNode:
    node = PipelineNode(config=_cfg(op=f"Op{node_id}"))
    node.id = node_id
    if tier1 is not None:
        node.review_result = _review(tier1, **review_kw)
    if score is not None:
        node.train_result = TrainResult(score=score, metric_name="dev")
    return node


# --------------------------- 渲染：Tier-1 信号必须可见 ---------------------------
def test_render_exposes_tier1_hard_metrics_and_signals():
    mem = GlobalMemory()
    mem.update(_node(2, tier1=0.77, signals=["F1", "F3"]))
    text = mem.render()

    assert "tier1=0.770" in text
    assert "dup=0.12" in text and "null=0.00" in text and "parse=1.00" in text
    assert "信号: F1,F3" in text
    assert "评审=通过" in text


def test_render_marks_review_rejected():
    mem = GlobalMemory()
    mem.update(_node(2, tier1=0.21, passed=False))
    assert "评审=未通过" in mem.render()


def test_header_reflects_actual_sort_basis():
    """无下游真分时表头不能宣称"按下游得分降序"（会误导 LLM）。"""
    mem = GlobalMemory()
    mem.update(_node(2, tier1=0.5))
    assert "Tier-1 复合分" in mem.render().splitlines()[0]

    mem.update(_node(3, tier1=0.5, score=0.42))
    assert "下游得分" in mem.render().splitlines()[0]


def test_issues_are_truncated_by_count_and_length():
    long_issue = "重" * 400
    mem = GlobalMemory(max_issues_per_entry=2, issue_maxlen=50)
    mem.update(_node(2, tier1=0.4, issues=[long_issue, "第二条", "第三条", "第四条"]))
    text = mem.render()

    assert "第二条" in text                 # 前 2 条保留
    assert "第三条" not in text             # 超出条数上限被丢
    assert "第四条" not in text
    assert "重" * 50 + "…" in text          # 单条按长度截断
    assert "重" * 51 not in text


def test_issues_can_be_disabled():
    mem = GlobalMemory(max_issues_per_entry=0)
    mem.update(_node(2, tier1=0.4, issues=["某个问题"]))
    assert "评审反馈" not in mem.render()


def test_all_issues_injected_when_unlimited():
    """max_issues_per_entry=-1 且 issue_maxlen<=0：完整反馈一字不删（PTB 模式的配置）。"""
    long_issue = "重" * 400
    mem = GlobalMemory(max_issues_per_entry=-1, issue_maxlen=0)
    mem.update(_node(2, tier1=0.4, issues=[long_issue, "第二条", "第三条", "第四条"]))
    text = mem.render()

    assert long_issue in text and "…" not in text     # 未截断
    for s in ("第二条", "第三条", "第四条"):
        assert s in text


def test_signals_and_issues_render_in_separate_fields():
    """信号走 `信号:` 字段、具体证据走 `评审反馈:` 字段，两者在 ReviewResult 里本就分开存。"""
    mem = GlobalMemory(max_issues_per_entry=-1)
    mem.update(_node(
        2, tier1=0.4,
        signals=["F3"],
        issues=["真正的问题甲", "真正的问题乙"],
    ))
    text = mem.render()

    assert "信号: F3" in text
    feedback = text.split("评审反馈:")[1]
    assert "真正的问题甲" in feedback and "真正的问题乙" in feedback
    assert "[F3]" not in text                         # 不再有 issues 里夹带的信号转述


# --------------------------- 排序：无真分时退回 tier1 ---------------------------
def test_sorting_falls_back_to_tier1_when_untrained():
    """review 模式下全是 None score，必须按 tier1 降序而非插入序。"""
    mem = GlobalMemory(top_k=2)
    mem.update(_node(2, tier1=0.30))
    mem.update(_node(3, tier1=0.90))
    mem.update(_node(4, tier1=0.60))

    assert [e.node_id for e in mem.top_entries()] == [3, 4]


def test_trained_nodes_rank_above_untrained():
    mem = GlobalMemory()
    mem.update(_node(2, tier1=0.99))              # 仅 tier1
    mem.update(_node(3, tier1=0.10, score=0.05))  # 有下游真分
    assert [e.node_id for e in mem.top_entries()] == [3, 2]


def test_top_k_is_honoured():
    mem = GlobalMemory(top_k=2)
    for i, t in enumerate([0.1, 0.2, 0.3, 0.4], start=2):
        mem.update(_node(i, tier1=t))
    assert len(mem.top_entries()) == 2


# --------------------------- 配置保持 ---------------------------
def test_clear_keeps_render_config():
    """续跑重建记忆时不能把 top_k 等配置重置回默认值。"""
    mem = GlobalMemory(top_k=9, max_issues_per_entry=1, issue_maxlen=20)
    mem.update(_node(2, tier1=0.5))
    mem.clear()

    assert len(mem) == 0
    assert (mem.top_k, mem.max_issues_per_entry, mem.issue_maxlen) == (9, 1, 20)


def test_tree_manager_wires_memory_top_k():
    from dataflow_evolver.core.tree_manager import TreeManager

    assert TreeManager().global_memory.top_k == 5                  # 默认不变
    assert TreeManager(memory_top_k=8).global_memory.top_k == 8


def test_checkpoint_restore_preserves_memory_top_k(tmp_path):
    """restore_tree 曾整体替换 GlobalMemory 实例，会静默把 top_k 打回默认 5。"""
    from dataflow_evolver.core import checkpoint
    from dataflow_evolver.core.schemas import TaskProfile
    from dataflow_evolver.core.tree_manager import TreeManager

    src = TreeManager(memory_top_k=8)
    child = src.root.add_child(_cfg())
    child.review_result = _review(0.66)
    src.backpropagate(child, 0.66)

    state = checkpoint.serialize(
        src, completed_iterations=1,
        profile=TaskProfile("生成", {"instruction": "str"}, ["非空"], "qa-sft"),
    )
    dst = TreeManager(memory_top_k=8)
    checkpoint.restore_tree(state, dst)

    assert dst.global_memory.top_k == 8
    assert len(dst.global_memory) == 1
    assert "tier1=0.660" in dst.global_memory.render()
