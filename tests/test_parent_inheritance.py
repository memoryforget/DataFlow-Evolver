"""父子继承通道的注入测试（rationale / 父节点定向质量块 / 兄弟负面清单）。

搜索期（reward_source=review）没有下游真分，父节点的 Tier-1 归因是唯一可行动的质量信号。
历史实现里这三样都没进 prompt：rationale 解析后从未使用，评审块只在"父节点被打回"时渲染
（而被打回的节点会置 dead、永远不会被 select 选为父节点，故是死代码），兄弟信息完全缺失。
"""
from __future__ import annotations

from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult, TaskProfile, TrainResult
from dataflow_evolver.core.tree_node import PipelineNode
from dataflow_evolver.llm.prompts import build_pipeline_prompt
from dataflow_evolver.main import _sibling_summaries

PROFILE = TaskProfile("QA", {"q": "str", "a": "str"}, [], "sft")


def _review(**kw) -> ReviewResult:
    base = dict(
        passed=True,
        issues=[],
        correctness_score=0.9,
        relevance_score=0.8,
        difficulty_score=0.7,
        diversity_score=0.2,
        schema_score=0.85,
        dup_rate=0.3,
        null_rate=0.0,
        parse_valid_rate=1.0,
        tier1_score=0.42,
        failure_signals=[],
    )
    base.update(kw)
    return ReviewResult(**base)


# ---- 1. rationale ----------------------------------------------------------
def test_parent_rationale_injected_with_parent_code():
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2",
        parent_code="print('parent')",
        parent_rationale="故意压低温度以保证答案正确性",
    )
    assert "父节点设计意图" in prompt
    assert "故意压低温度以保证答案正确性" in prompt


def test_no_rationale_block_when_absent():
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2", parent_code="print('parent')"
    )
    assert "父节点设计意图" not in prompt


def test_rationale_injected_even_without_parent_code():
    """code 为空（续跑时 node 目录已清理）时子节点信息最匮乏，意图反而是仅存的方向信息。"""
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2", parent_code="", parent_rationale="压低温度保正确性"
    )
    assert "父节点设计意图" in prompt
    assert "压低温度保正确性" in prompt
    assert "【变异起点】" not in prompt
    # 措辞不能悬空："在此基础上改进"的"此"指那段代码，无代码时须换成中性表述
    assert "在此基础上改进" not in prompt
    assert "可作为方向参考" in prompt


def test_rationale_wording_switches_with_code_presence():
    with_code = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2", parent_code="print(1)", parent_rationale="理由"
    )
    assert "在此基础上改进" in with_code
    assert "可作为方向参考" not in with_code


# ---- 2. 父节点定向质量块 ---------------------------------------------------
def test_parent_review_block_injected_even_when_passed():
    """关键回归：父节点评审**通过**时也必须注入归因（这正是旧实现漏掉的唯一现实分支）。"""
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2", parent_review=_review(passed=True)
    )
    assert "父节点数据质量归因" in prompt
    assert "评审通过" in prompt


def test_parent_review_block_exposes_dims_and_weakest():
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2", parent_review=_review()
    )
    assert "多样性 0.20" in prompt and "正确性 0.90" in prompt
    # 最弱两维应被点名：diversity(0.20) 与 difficulty(0.70)
    assert "最弱维度：多样性(0.20)、难度(0.70)" in prompt
    assert "Tier-1 复合分：0.420" in prompt
    assert "重复率 0.30" in prompt and "可解析率 1.00" in prompt


def test_parent_review_block_renders_failure_signals_with_description():
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2",
        parent_review=_review(failure_signals=["F3"]),
    )
    assert "命中的失败信号" in prompt
    assert "F3：" in prompt


def test_parent_review_block_renders_signals_and_issues_separately():
    """信号段按编号查表拼出，issues 段照实渲染——两者在 ReviewResult 里本就是两个字段。"""
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2",
        parent_review=_review(
            failure_signals=["F3"],
            issues=["80 条里约 60 条是同一个购物找零模板"],
        ),
    )
    assert "F3：" in prompt                                  # 信号段
    assert "80 条里约 60 条是同一个购物找零模板" in prompt      # 证据段
    assert "[F3]" not in prompt                              # issues 里不该夹带信号转述


def test_parent_review_issues_not_truncated():
    """父节点是最相关节点，其 issues 原文完整注入（不做长度截断）。"""
    long_issue = "非常具体的证据" * 40
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2", parent_review=_review(issues=[long_issue])
    )
    assert long_issue in prompt


def test_parent_review_marks_rejected():
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_2", parent_review=_review(passed=False)
    )
    assert "评审未通过" in prompt


# ---- 3. 兄弟负面清单 -------------------------------------------------------
def test_sibling_block_injected():
    prompt = build_pipeline_prompt(
        PROFILE, "empty.jsonl", "node_4",
        siblings=["node_2: ops=[A] | tier1=0.310", "node_3: ops=[B] | tier1=0.440"],
    )
    assert "兄弟节点已试过的方向" in prompt
    assert "node_2: ops=[A] | tier1=0.310" in prompt
    assert "node_3: ops=[B] | tier1=0.440" in prompt


def test_no_sibling_block_when_empty():
    prompt = build_pipeline_prompt(PROFILE, "empty.jsonl", "node_2", siblings=[])
    assert "兄弟节点已试过的方向" not in prompt


def _node(node_id: int, **kw) -> PipelineNode:
    node = PipelineNode()
    node.id = node_id                        # id 由全局计数器分配，测试里固定以便断言 node_dir_name
    for k, v in kw.items():
        setattr(node, k, v)
    return node


def test_sibling_summaries_collects_ops_scores_and_intent():
    parent = _node(1)
    sib = _node(
        2,
        config=PipelineConfig(
            operators=[{"name": "Gen"}, {"name": "DupFilter"}],
            field_flow="empty -> qa",
            code="x",
            rationale="加近重复过滤提升多样性",
        ),
        review_result=_review(tier1_score=0.31),
    )
    pending = _node(3)
    parent.children = [sib, pending]

    out = _sibling_summaries(parent, exclude=pending)
    assert len(out) == 1
    node_id, line = out[0]
    assert node_id == 2                        # 连 id 一起返回，供调用方排除记忆里的重复条目
    assert "node_2" in line
    assert "ops=[Gen, DupFilter]" in line
    assert "tier1=0.310" in line
    assert "加近重复过滤提升多样性" in line


def test_sibling_summaries_skips_excluded_and_contentless():
    parent = _node(1)
    empty = _node(2)                                     # 预留但还没生成内容
    parent.children = [empty]
    assert _sibling_summaries(parent, exclude=None) == []


def test_sibling_summaries_marks_dead_and_downstream_score():
    parent = _node(1)
    sib = _node(
        2,
        config=PipelineConfig(operators=[{"name": "Gen"}], field_flow="f", code="x", rationale=""),
        review_result=_review(tier1_score=0.20),
        train_result=TrainResult(score=0.55, metric_name="accuracy"),
        dead=True,
    )
    parent.children = [sib]
    ((_, line),) = _sibling_summaries(parent)
    assert "dev=0.550" in line and "已判失败" in line


def test_sibling_summaries_truncates_long_rationale():
    parent = _node(1)
    sib = _node(
        2,
        config=PipelineConfig(
            operators=[{"name": "Gen"}], field_flow="f", code="x", rationale="理" * 200
        ),
    )
    parent.children = [sib]
    ((_, line),) = _sibling_summaries(parent)
    assert "…" in line
    assert len(line) < 300


def test_sibling_summaries_caps_item_count():
    parent = _node(1)
    parent.children = [
        _node(
            i,
            config=PipelineConfig(
                operators=[{"name": f"Op{i}"}], field_flow="f", code="x", rationale=""
            ),
        )
        for i in range(2, 12)
    ]
    assert len(_sibling_summaries(parent, max_items=6)) == 6


# ---- 4. 与 global memory 去重 ---------------------------------------------
def test_global_memory_render_can_exclude_parent_and_siblings():
    """父与已列出的兄弟都有专属块：父避免重复，兄弟避免"参考它/别学它"的角色矛盾。"""
    from dataflow_evolver.core.global_memory import GlobalMemory

    memory = GlobalMemory()
    for node_id, tier1 in ((2, 0.5), (3, 0.4), (4, 0.3)):
        node = _node(
            node_id,
            config=PipelineConfig(
                operators=[{"name": f"Op{node_id}"}], field_flow="f", code="x", rationale=""
            ),
            review_result=_review(tier1_score=tier1),
        )
        memory.update(node)

    assert "[node 2]" in memory.render()
    excluded = memory.render(exclude_node_ids={2, 3})
    assert "[node 2]" not in excluded and "[node 3]" not in excluded
    assert "[node 4]" in excluded
    assert memory.render(exclude_node_ids=None) == memory.render()


# ---- 5. 块编号不撞号 / 过滤规则单一实现 -----------------------------------
def test_diagnostic_block_has_no_duplicate_numbering():
    """要求列表第 7 条是 compile 校验；诊断块曾也带 "7."，train 模式下会出现两个 7.。"""
    from dataflow_evolver.core.schemas import DiagnosticReport

    diag = DiagnosticReport(summary="答案格式错误率高", actionable_suggestions=["加格式校验算子"])
    prompt = build_pipeline_prompt(PROFILE, "empty.jsonl", "node_2", diagnostic=diag)

    assert "【父节点诊断反馈】" in prompt
    assert "加格式校验算子" in prompt
    assert len([ln for ln in prompt.splitlines() if ln.startswith("7. ")]) == 1


def test_selfcorrect_prompt_renders_signals_from_ids():
    """自修正 prompt 自己按编号查表拼信号段落——不需要 ReviewAgent 预先把文本混进 issues。"""
    from dataflow_evolver.llm.prompts import FAILURE_SIGNALS, build_review_selfcorrect_prompt

    prompt = build_review_selfcorrect_prompt(
        "node_2", ["80 条里 60 条同一模板"], failure_signals=["F3", "bogus"]
    )
    assert "命中的失败信号" in prompt
    assert f"F3：{FAILURE_SIGNALS['F3']}" in prompt
    assert "bogus" not in prompt                      # 非法编号忽略
    assert "80 条里 60 条同一模板" in prompt


def test_selfcorrect_prompt_without_signals_has_no_block():
    from dataflow_evolver.llm.prompts import build_review_selfcorrect_prompt

    assert "命中的失败信号" not in build_review_selfcorrect_prompt("node_2", ["qa 为空"])
