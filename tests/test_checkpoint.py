"""断点续跑 checkpoint 的序列化/恢复测试（纯内存往返 + id 计数器恢复 + 读容错）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.core import checkpoint
from dataflow_evolver.core.schemas import (
    BadCase,
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TaskProfile,
    TrainResult,
)
from dataflow_evolver.core.tree_manager import TreeManager


def _profile():
    return TaskProfile("生成", {"instruction": "str"}, ["非空"], "qa-sft")


def _write_node_code(agent_root: Path, node_id: int, code: str) -> None:
    """把节点流水线源码写到磁盘（checkpoint 不再存 code，恢复时从这里重读）。"""
    d = agent_root / f"node_{node_id}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pipeline.py").write_text(code, encoding="utf-8")


def _build_tree(agent_root: Path | None = None):
    """Init node 作根（config + visits=1 + 首个分数），下面挂一个已评估的变异节点。"""
    tm = TreeManager(exploration_constant=1.0)
    tm.root.config = PipelineConfig([{"name": "Init"}], "f", "INIT_CODE", "r")
    tm.root.visits = 1
    tm.root.total_reward = 0.5
    tm.global_memory.update(tm.root)

    c1 = tm.expand(tm.root, PipelineConfig([{"name": "Gen"}], "f1", "C1_CODE", "r1"))
    c1.review_result = ReviewResult(0.9, 0.95, True, ["ok"])
    c1.train_result = TrainResult(0.8, "acc", [BadCase("i", "p", "g", "t")], "log", False, None)
    c1.diagnostic_report = DiagnosticReport("摘要", ["fp1"], ["建议1"])
    tm.backpropagate(c1, 0.3)
    if agent_root is not None:                       # 落盘 pipeline.py 供 code 重读
        _write_node_code(agent_root, tm.root.id, "INIT_CODE")
        _write_node_code(agent_root, c1.id, "C1_CODE")
    return tm, c1


def test_checkpoint_roundtrip_tree(tmp_path):
    agent_root = tmp_path / "_agent_root"
    tm, c1 = _build_tree(agent_root)

    path = str(tmp_path / "ckpt.json")
    checkpoint.write(path, checkpoint.serialize(tm, completed_iterations=1, profile=_profile()))
    # code 不应写进 checkpoint 文件（体积优化）
    assert "INIT_CODE" not in Path(path).read_text(encoding="utf-8")
    state = checkpoint.read(path)
    assert state is not None

    tm2 = TreeManager(exploration_constant=1.0)
    checkpoint.restore_tree(state, tm2, agent_root=str(agent_root))

    # 根（Init node）：config、visits、reward
    assert tm2.root.id == tm.root.id
    assert tm2.root.config.code == "INIT_CODE"          # 从磁盘 pipeline.py 重读
    assert tm2.root.visits == 2 and abs(tm2.root.total_reward - 0.8) < 1e-9
    # 子节点：结构 + 双反馈结果完整恢复
    assert len(tm2.root.children) == 1
    rc1 = tm2.root.children[0]
    assert rc1.id == c1.id and rc1.parent is tm2.root
    assert rc1.config.code == "C1_CODE"                 # 从磁盘 pipeline.py 重读
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
    path = str(tmp_path / "c.json")
    checkpoint.write(path, checkpoint.serialize(tm, 1, _profile()))
    state = checkpoint.read(path)

    tm2 = TreeManager()
    checkpoint.restore_tree(state, tm2)
    # 续跑时新建的节点 id 必须大于所有已载入节点（否则 node_dir 名冲突）
    new = tm2.reserve_child(tm2.root)
    assert new.id > max(c1.id, tm.root.id)


def test_checkpoint_read_missing_and_bad(tmp_path):
    assert checkpoint.read(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert checkpoint.read(str(bad)) is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"schema_version": 999}', encoding="utf-8")
    assert checkpoint.read(str(wrong)) is None
