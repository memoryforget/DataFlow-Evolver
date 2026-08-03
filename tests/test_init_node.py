"""Init node（node_0 = 树根）单测：与普通节点同口径评分、三档启动模式、失败即终止。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.schemas import TaskProfile, TrainResult
from dataflow_evolver.llm.prompts import build_pipeline_prompt

SCRATCH = InputCorpus(entry_path="empty.jsonl")   # 无输入语料 = 从零合成

from _ptb_stubs import (
    StubReviewer,
    StubTrainer,
    bare_orchestrator,
    make_review_result,
)

PROFILE = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")


def test_root_is_node_0(tmp_path):
    """根的目录名必须是 node_0——文档、prompt、日志都这么叫它。

    节点 id 来自全局计数器，若不钉死，建树前多造一个节点就会让根漂成 node_3 之类，
    而这只体现在目录名上，几乎不可能在跑起来之后发现。
    """
    from dataflow_evolver.core.tree_manager import TreeManager

    assert TreeManager().root.node_dir_name == "node_0"
    o = bare_orchestrator(tmp_path)
    assert o.tree.root.node_dir_name == "node_0"


def test_init_node_becomes_root_with_first_score(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="review")
    o.reviewer = StubReviewer([make_review_result(tier1=0.61)])

    assert o._run_init_node(PROFILE) is True

    root = o.tree.root
    assert root.config is not None                    # 根不再是虚拟空壳
    assert root.visits == 1
    assert root.mean_reward == pytest.approx(0.61)    # 分数就是 tier1，没有任何平移
    assert root.parent is None
    assert len(o.tree.global_memory) == 1             # 立即进全局记忆供首层子节点参考


def test_init_node_uses_same_scoring_path_as_children(tmp_path):
    """train 模式下 Init node 一样要跑训练 + 诊断——它不享受特殊评分口径。"""
    o = bare_orchestrator(tmp_path, reward_source="train")
    o.reviewer = StubReviewer([make_review_result(passed=True, tier1=0.1)])
    o.trainer = StubTrainer([TrainResult(score=0.73, metric_name="math")])

    assert o._run_init_node(PROFILE) is True
    assert o.tree.root.mean_reward == pytest.approx(0.73)
    assert o.trainer.calls == 1
    assert o.diagnostician.calls == 1


def test_init_node_failure_aborts_run(tmp_path):
    """起点跑不出分 = 整棵树没有可变异的基础，应当直接失败而不是硬着头皮搜索。"""
    o = bare_orchestrator(tmp_path, reward_source="review")
    o.reviewer = StubReviewer([make_review_result(passed=False, tier1=0.0)])

    assert o._run_init_node(PROFILE) is False
    assert o.tree.root.dead is True


def test_init_node_generation_failure_aborts_run(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="review")
    o.pipeline_agent.generate_initial = lambda **kwargs: None
    assert o._run_init_node(PROFILE) is False


# ---- 三档启动模式的 prompt 渲染 -------------------------------------------
def _init_prompt(**kwargs) -> str:
    return build_pipeline_prompt(
        profile=PROFILE, corpus=SCRATCH, node_dir="node_1",
        is_init=True, **kwargs,
    )


def test_init_prompt_reference_mode_demands_faithful_refactor():
    prompt = _init_prompt(init_reference_code="print('reference')")
    assert "print('reference')" in prompt
    assert "忠实改写" in prompt
    assert "保持其数据处理逻辑与算法取舍不变" in prompt


def test_init_prompt_instruction_mode():
    prompt = _init_prompt(init_instruction="先合成题目再蒸馏 CoT")
    assert "先合成题目再蒸馏 CoT" in prompt
    assert "忠实改写" not in prompt


def test_init_prompt_scratch_mode():
    prompt = _init_prompt()
    assert "自主设计" in prompt
    assert "忠实改写" not in prompt


def test_non_init_prompt_has_no_init_block():
    prompt = build_pipeline_prompt(
        profile=PROFILE, corpus=SCRATCH, node_dir="node_2",
        parent_code="print('parent')",
    )
    assert "初始节点" not in prompt
    assert "变异起点" in prompt
