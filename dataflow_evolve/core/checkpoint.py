"""断点续跑：把 MCTS 运行状态序列化到磁盘，崩溃/中断后从上次进度继续。

动机：真实跑每个节点都耗真实 API/SDK 预算；进程在第 k/N 轮挂掉若从头再来 = 重复烧钱。
这里把【树状态 + 奖励策略 + 数据池暂存 + 进度 + TaskProfile】落盘成一个 JSON checkpoint，
每完成一轮覆盖写一次；启动时若存在则载入、从下一轮续跑。

节点产物（pipeline.py / 数据集 / cache）本就落在各 node 目录，无需在此重复；这里只存
"树结构 + 统计 + 进度"这层内存态。默认关闭（`search.checkpoint` 为空）。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dataflow_evolve.core.baseline import RewardPolicy
from dataflow_evolve.core.data_pool import DataPool, StagedNode
from dataflow_evolve.core.schemas import (
    BadCase,
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TaskProfile,
    TrainResult,
)
from dataflow_evolve.core.tree_manager import TreeManager
from dataflow_evolve.core.tree_node import PipelineNode, reseed_id_counter
from dataflow_evolve.utils.logging import get_logger

logger = get_logger("core.checkpoint")

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------
def _node_to_dict(node: PipelineNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "parent_id": node.parent.id if node.parent is not None else None,
        "config": asdict(node.config) if node.config is not None else None,
        "visits": node.visits,
        "total_reward": node.total_reward,
        "baseline_dev_score": node.baseline_dev_score,
        "review_result": asdict(node.review_result) if node.review_result else None,
        "train_result": asdict(node.train_result) if node.train_result else None,
        "diagnostic_report": asdict(node.diagnostic_report) if node.diagnostic_report else None,
    }


def _all_nodes(tree: TreeManager) -> list[PipelineNode]:
    """前序遍历（含 root），按 id 排序保证确定性。"""
    out: list[PipelineNode] = []
    stack = [tree.root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return sorted(out, key=lambda n: n.id)


def serialize(
    tree: TreeManager,
    reward_policy: RewardPolicy,
    pool: DataPool | None,
    completed_iterations: int,
    profile: TaskProfile,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "completed_iterations": completed_iterations,
        "step_count": tree._step_count,
        "profile": asdict(profile),
        "reward_policy": {
            "reward_mode": reward_policy.reward_mode,
            "baseline_dev_score": reward_policy.baseline_dev_score,
        },
        "nodes": [_node_to_dict(n) for n in _all_nodes(tree)],
    }
    if pool is not None:
        state["pool"] = {
            "gate_mode": pool.gate_mode,
            "baseline_dev": pool.baseline_dev,
            "margin": pool.margin,
            "staged": [
                {"node_id": s.node_id, "dev_score": s.dev_score, "rows": s.rows}
                for s in pool._staged
            ],
        }
    return state


def write(path: str, state: dict[str, Any]) -> None:
    """原子写：先写临时文件再 os.replace（Windows 上亦原子）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def read(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("checkpoint 读取失败，忽略并从头跑：%s", exc)
        return None
    if state.get("schema_version") != SCHEMA_VERSION:
        logger.warning("checkpoint schema 版本不符（%s≠%s），忽略。",
                       state.get("schema_version"), SCHEMA_VERSION)
        return None
    return state


# ---------------------------------------------------------------------------
# 反序列化 / 恢复
# ---------------------------------------------------------------------------
def restore_profile(state: dict[str, Any]) -> TaskProfile:
    return TaskProfile(**state["profile"])


def restore_reward_policy(state: dict[str, Any]) -> RewardPolicy:
    rp = state["reward_policy"]
    return RewardPolicy(reward_mode=rp["reward_mode"], baseline_dev_score=rp["baseline_dev_score"])


def _config_from_dict(d: dict[str, Any] | None) -> PipelineConfig | None:
    return PipelineConfig(**d) if d else None


def _review_from_dict(d: dict[str, Any] | None) -> ReviewResult | None:
    return ReviewResult(**d) if d else None


def _train_from_dict(d: dict[str, Any] | None) -> TrainResult | None:
    if not d:
        return None
    bad = [BadCase(**bc) for bc in d.get("bad_cases", [])]
    return TrainResult(
        score=d["score"], metric_name=d["metric_name"], bad_cases=bad,
        raw_log_path=d.get("raw_log_path", ""), failed=d.get("failed", False),
        error_msg=d.get("error_msg"),
    )


def _diag_from_dict(d: dict[str, Any] | None) -> DiagnosticReport | None:
    return DiagnosticReport(**d) if d else None


def restore_tree(state: dict[str, Any], tree: TreeManager) -> None:
    """把序列化的节点重建进 `tree`：恢复 root、统计、global_memory、step_count、id 计数器。"""
    built: dict[int, PipelineNode] = {}
    parent_of: dict[int, int | None] = {}
    for nd in state["nodes"]:
        node = PipelineNode(config=_config_from_dict(nd["config"]), parent=None)
        node.id = nd["id"]
        node.visits = nd["visits"]
        node.total_reward = nd["total_reward"]
        node.baseline_dev_score = nd["baseline_dev_score"]
        node.review_result = _review_from_dict(nd["review_result"])
        node.train_result = _train_from_dict(nd["train_result"])
        node.diagnostic_report = _diag_from_dict(nd["diagnostic_report"])
        built[node.id] = node
        parent_of[node.id] = nd["parent_id"]

    root: PipelineNode | None = None
    for nid in sorted(built):                  # 按 id 链接，children 顺序确定
        node = built[nid]
        pid = parent_of[nid]
        if pid is None:
            root = node
        else:
            parent = built[pid]
            node.parent = parent
            parent.children.append(node)
    if root is None:
        raise ValueError("checkpoint 无根节点，无法恢复树")

    tree.root = root
    tree._step_count = state.get("step_count", 0)
    tree.global_memory = type(tree.global_memory)()   # 清空后从节点重建
    for nid in sorted(built):
        if built[nid].config is not None:
            tree.global_memory.update(built[nid])
    reseed_id_counter(max(built) + 1 if built else 1)


def restore_pool(state: dict[str, Any], pool: DataPool) -> None:
    pd = state.get("pool")
    if not pd:
        return
    pool.gate_mode = pd["gate_mode"]
    pool.baseline_dev = pd["baseline_dev"]
    pool.margin = pd["margin"]
    pool._staged = [
        StagedNode(node_id=s["node_id"], dev_score=s["dev_score"], rows=s["rows"])
        for s in pd["staged"]
    ]
