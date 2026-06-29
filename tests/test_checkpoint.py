"""断点续跑 checkpoint 的序列化/恢复测试（纯内存往返 + id 计数器恢复 + 读容错）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolve.core import checkpoint
from dataflow_evolve.core.baseline import REWARD_RELATIVE, RewardPolicy
from dataflow_evolve.core.data_pool import DataPool, PoolConfig
from dataflow_evolve.core.schemas import (
    BadCase,
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TaskProfile,
    TrainResult,
)
from dataflow_evolve.core.tree_manager import TreeManager


def _profile():
    return TaskProfile("生成", {"instruction": "str"}, "qa", ["非空"], "qa-sft")


def _build_tree():
    tm = TreeManager(exploration_constant=1.0)
    tm.set_baseline_root(PipelineConfig([{"name": "B"}], "f", "BASE_CODE", "r"), 0.5)
    c1 = tm.expand(tm.root, PipelineConfig([{"name": "Gen"}], "f1", "C1_CODE", "r1"))
    c1.review_result = ReviewResult(0.9, 0.95, True, ["ok"])
    c1.train_result = TrainResult(0.8, "acc", [BadCase("i", "p", "g", "t")], "log", False, None)
    c1.diagnostic_report = DiagnosticReport("摘要", ["fp1"], ["建议1"])
    tm.backpropagate(c1, 0.3)
    return tm, c1


def test_checkpoint_roundtrip_tree(tmp_path):
    tm, c1 = _build_tree()
    rp = RewardPolicy(REWARD_RELATIVE, baseline_dev_score=0.5)
    pool = DataPool(PoolConfig(enabled=True))
    pool.set_warm_gate(0.5, 0.0)
    pool.stage(c1.id, [{"a": 1}], dev_score=0.8, review_passed=True)

    path = str(tmp_path / "ckpt.json")
    checkpoint.write(path, checkpoint.serialize(tm, rp, pool, completed_iterations=1, profile=_profile()))
    state = checkpoint.read(path)
    assert state is not None

    tm2 = TreeManager(exploration_constant=1.0)
    checkpoint.restore_tree(state, tm2)

    # 根：基线 config、visits、reward、baseline_dev
    assert tm2.root.id == tm.root.id
    assert tm2.root.config.code == "BASE_CODE"
    assert tm2.root.visits == 2 and abs(tm2.root.total_reward - 0.3) < 1e-9
    assert abs(tm2.root.baseline_dev_score - 0.5) < 1e-9
    # 子节点：结构 + 双反馈结果完整恢复
    assert len(tm2.root.children) == 1
    rc1 = tm2.root.children[0]
    assert rc1.id == c1.id and rc1.parent is tm2.root
    assert rc1.config.code == "C1_CODE"
    assert rc1.visits == 1 and abs(rc1.total_reward - 0.3) < 1e-9
    assert rc1.review_result.passed is True
    assert rc1.train_result.score == 0.8
    assert rc1.train_result.bad_cases[0].error_type == "t"
    assert rc1.diagnostic_report.summary == "摘要"
    # step_count（衰减用）与 global_memory 一并恢复
    assert tm2._step_count == tm._step_count == 1
    assert len(tm2.global_memory) == 2


def test_checkpoint_reseeds_id_counter(tmp_path):
    tm, c1 = _build_tree()
    rp = RewardPolicy(REWARD_RELATIVE, baseline_dev_score=0.5)
    path = str(tmp_path / "c.json")
    checkpoint.write(path, checkpoint.serialize(tm, rp, None, 1, _profile()))
    state = checkpoint.read(path)

    tm2 = TreeManager()
    checkpoint.restore_tree(state, tm2)
    # 续跑时新建的节点 id 必须大于所有已载入节点（否则 node_dir 名冲突）
    new = tm2.reserve_child(tm2.root)
    assert new.id > max(c1.id, tm.root.id)


def test_checkpoint_restore_reward_policy_and_pool(tmp_path):
    tm, c1 = _build_tree()
    rp = RewardPolicy(REWARD_RELATIVE, baseline_dev_score=0.5)
    pool = DataPool(PoolConfig(enabled=True))
    pool.set_warm_gate(0.5, 0.0)
    pool.stage(c1.id, [{"a": 1}, {"b": 2}], dev_score=0.8, review_passed=True)

    path = str(tmp_path / "c.json")
    checkpoint.write(path, checkpoint.serialize(tm, rp, pool, 1, _profile()))
    state = checkpoint.read(path)

    rp2 = checkpoint.restore_reward_policy(state)
    assert rp2.reward_mode == REWARD_RELATIVE
    assert abs(rp2.baseline_dev_score - 0.5) < 1e-9

    pool2 = DataPool(PoolConfig(enabled=True))
    checkpoint.restore_pool(state, pool2)
    assert pool2.gate_mode == "relative_baseline"
    assert len(pool2._staged) == 1 and len(pool2._staged[0].rows) == 2
    # 恢复后 finalize 仍按严格 > 基线工作（0.8 > 0.5 → 2 行入池）
    pool2.config.max_pool_fraction_per_pipeline = 0
    pool2.finalize()
    assert len(pool2) == 2


def test_checkpoint_read_missing_and_bad(tmp_path):
    assert checkpoint.read(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert checkpoint.read(str(bad)) is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"schema_version": 999}', encoding="utf-8")
    assert checkpoint.read(str(wrong)) is None
