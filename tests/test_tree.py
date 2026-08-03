import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.core.tree_manager import TreeManager
from dataflow_evolver.core.tree_node import PipelineNode


def _cfg(name: str = "op") -> PipelineConfig:
    return PipelineConfig(
        operators=[{"name": name, "params": {}}],
        field_flow="a -> b",
        code="# code",
        rationale="r",
    )


def test_unvisited_node_has_infinite_uct():
    root = PipelineNode()
    child = root.add_child(_cfg())
    assert child.uct_value(1.414) == math.inf


def test_uct_formula():
    root = PipelineNode()
    child = root.add_child(_cfg())
    root.visits = 10
    child.visits = 3
    child.total_reward = 1.5  # mean 0.5
    c = 1.414
    expected = 0.5 + c * math.sqrt(math.log(10) / 3)
    assert abs(child.uct_value(c) - expected) < 1e-9


def test_backpropagate_accumulates_along_ancestors():
    tm = TreeManager()
    n1 = tm.expand(tm.root, _cfg("a"))
    n2 = n1.add_child(_cfg("b"))
    tm.backpropagate(n2, reward=0.8)
    assert n2.visits == 1 and abs(n2.total_reward - 0.8) < 1e-9
    assert n1.visits == 1 and abs(n1.total_reward - 0.8) < 1e-9
    assert tm.root.visits == 1 and abs(tm.root.total_reward - 0.8) < 1e-9


def test_select_returns_root_until_first_layer_full():
    tm = TreeManager(max_children_per_node=2)
    assert tm.select() is tm.root
    tm.expand(tm.root, _cfg("a"))
    assert tm.select() is tm.root  # still room for 1 more
    tm.expand(tm.root, _cfg("b"))
    # root now full -> select must descend to an expandable child
    selected = tm.select()
    assert selected in tm.root.children


def test_select_prefers_higher_reward_when_visits_equal():
    tm = TreeManager(max_children_per_node=2, exploration_constant=0.1)
    a = tm.expand(tm.root, _cfg("a"))
    b = tm.expand(tm.root, _cfg("b"))
    # give both one visit so UCT is finite; a has higher reward
    tm.backpropagate(a, 0.9)
    tm.backpropagate(b, 0.1)
    # root visited twice now; both children visited once -> equal explore term
    selected = tm.select()
    assert selected is a


def test_best_node_picks_highest_mean_reward():
    tm = TreeManager(max_children_per_node=3)
    a = tm.expand(tm.root, _cfg("a"))
    b = tm.expand(tm.root, _cfg("b"))
    tm.backpropagate(a, 0.3)
    tm.backpropagate(b, 0.7)
    assert tm.best_node() is b


def test_dead_child_not_counted_in_expandable():
    """dead 子节点不占扩展配额：被失败塞满后父节点仍可扩展（review M1）。"""
    tm = TreeManager(max_children_per_node=2)
    a = tm.reserve_child(tm.root)
    b = tm.reserve_child(tm.root)
    a.dead = True
    b.dead = True
    # 两个子节点都 dead → 存活数 0 < 2 → 仍可扩展，select 回到 root
    assert tm.is_expandable(tm.root) is True
    assert tm.select() is tm.root


def test_dead_child_skipped_in_selection_and_best_node():
    tm = TreeManager(max_children_per_node=2, exploration_constant=0.1)
    a = tm.expand(tm.root, _cfg("a"))
    b = tm.expand(tm.root, _cfg("b"))
    tm.backpropagate(a, 0.9)   # a 分高但标记 dead
    tm.backpropagate(b, 0.1)
    a.dead = True
    assert tm.root.best_child(0.1) is b      # 跳过 dead 的 a
    assert tm.best_node() is b               # best_node 也跳过 dead


def _init_root(tm: TreeManager, score: float) -> None:
    """按 Orchestrator._run_init_node 的记账方式把 root 填成 Init node。"""
    tm.root.config = _cfg("init")
    tm.root.visits = 1
    tm.root.total_reward = score


def test_best_node_includes_init_root():
    """没有变异超过初始流水线时，交付的应是初始流水线本身而非"最不差的变异"。"""
    tm = TreeManager(max_children_per_node=3)
    _init_root(tm, 0.8)
    worse = tm.reserve_child(tm.root)
    worse.config = _cfg("worse")
    tm.backpropagate(worse, 0.6)
    assert tm.best_node() is tm.root

    better = tm.reserve_child(tm.root)
    better.config = _cfg("better")
    tm.backpropagate(better, 0.95)
    assert tm.best_node() is better


def test_all_nodes_excludes_virtual_root_but_keeps_init_root():
    tm = TreeManager()
    a = tm.expand(tm.root, _cfg("a"))
    assert tm.all_nodes() == [a]             # 虚拟根（config=None）不是一次实验

    _init_root(tm, 0.5)
    assert set(tm.all_nodes()) == {tm.root, a}


def test_global_memory_updates_on_backprop():
    tm = TreeManager()
    a = tm.expand(tm.root, _cfg("gen"))
    tm.backpropagate(a, 0.5)
    assert len(tm.global_memory) == 1
    rendered = tm.global_memory.render()
    assert "gen" in rendered


def test_exploration_decay_default_is_no_op():
    tm = TreeManager(exploration_constant=1.5)  # decay=1.0 默认
    assert tm.current_exploration() == 1.5
    tm.backpropagate(tm.expand(tm.root, _cfg("a")), 0.1)
    tm.backpropagate(tm.expand(tm.root, _cfg("b")), 0.1)
    assert tm.current_exploration() == 1.5  # 不衰减


def test_exploration_decays_with_steps_and_floors():
    tm = TreeManager(exploration_constant=1.0, exploration_decay=0.5, exploration_min=0.2)
    assert tm.current_exploration() == 1.0           # step 0: 1.0 * 0.5^0
    tm.backpropagate(tm.expand(tm.root, _cfg("a")), 0.1)
    assert abs(tm.current_exploration() - 0.5) < 1e-9  # step 1: 0.5
    tm.backpropagate(tm.expand(tm.root, _cfg("b")), 0.1)
    assert abs(tm.current_exploration() - 0.25) < 1e-9  # step 2: 0.25
    tm.backpropagate(tm.expand(tm.root, _cfg("c")), 0.1)
    assert tm.current_exploration() == 0.2           # step 3: 0.125 → 触地板 0.2
