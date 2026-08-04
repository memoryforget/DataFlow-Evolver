"""断点续跑：把 MCTS 运行状态序列化到磁盘，崩溃/中断后从上次进度继续。

动机：真实跑每个节点都耗真实 API/SDK 预算；进程在第 k/N 轮挂掉若从头再来 = 重复烧钱。
这里把【树状态 + 进度 + TaskProfile】落盘成一个 JSON checkpoint，
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

from dataflow_evolver.core.schemas import (
    BadCase,
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TaskProfile,
    TrainResult,
)
from dataflow_evolver.core.tree_manager import TreeManager
from dataflow_evolver.core.tree_node import PipelineNode, reseed_id_counter
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("core.checkpoint")

# v2：TaskProfile 移除 domain 字段（旧 v1 checkpoint 含 domain，恢复会失败）→ 提版本使其被安全丢弃重跑
# v3：移除 baseline / data_pool 模块，reward_policy 与 pool 两段不再存在，root 改为 Init node
SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------
def _config_to_dict(config) -> dict[str, Any] | None:
    """序列化 PipelineConfig，但**不含 `code`**：完整流水线源码已落盘于节点目录的
    pipeline.py，恢复时按需从磁盘重读（见 restore_tree），避免 checkpoint 随代码体积膨胀。
    """
    if config is None:
        return None
    d = asdict(config)
    d.pop("code", None)
    return d


def _node_to_dict(node: PipelineNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "parent_id": node.parent.id if node.parent is not None else None,
        "config": _config_to_dict(node.config),
        "visits": node.visits,
        "total_reward": node.total_reward,
        "dead": node.dead,
        "review_result": asdict(node.review_result) if node.review_result else None,
        "tier2_review_result": (
            asdict(node.tier2_review_result) if node.tier2_review_result else None
        ),
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
    completed_iterations: int,
    profile: TaskProfile,
    dev_set_recipe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_iterations": completed_iterations,
        "step_count": tree._step_count,
        "profile": asdict(profile),
        # 已存节点的分数都出自这份 dev 集；续跑时换了口径就不可比（见 verify_dev_set）。
        "dev_set": dev_set_recipe,
        "nodes": [_node_to_dict(n) for n in _all_nodes(tree)],
    }


def verify_dev_set(state: dict[str, Any], current_recipe: dict[str, Any] | None) -> None:
    """续跑前核对 dev 集口径；不一致则抛异常。

    `dev_set.materialize` 只能发现 workspace 里遗留的 manifest 与本次配置不符；若 workspace
    的 eval 目录被清掉，它会心安理得地重建一份新题集，而 checkpoint 里的旧节点分数是在老题集
    上得到的——两批分数混在一棵树里比较，MCTS 的排序就是错的，且没有任何迹象。
    """
    recorded = state.get("dev_set")
    if recorded is None and current_recipe is None:
        return
    if recorded is None:
        logger.warning("checkpoint 未记录 dev 集口径（旧版本落盘），跳过核对。")
        return
    if recorded != current_recipe:
        raise ValueError(
            "checkpoint 里的 dev 集口径与本次配置不一致，续跑会把两批不可比的分数混进同一棵树。\n"
            f"  checkpoint：{recorded}\n"
            f"  本次配置：  {current_recipe}\n"
            "要么改回原配置续跑，要么换一个 checkpoint 路径从头跑。"
        )


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


def _load_node_code(agent_root: str | None, node_id: int, pipeline_filename: str) -> str:
    """从节点目录重读流水线源码（checkpoint 不再存 code）。找不到则返回 ""（不阻塞续跑）。"""
    if not agent_root:
        return ""
    p = Path(agent_root) / f"node_{node_id}" / pipeline_filename
    try:
        return p.read_text(encoding="utf-8") if p.exists() else ""
    except OSError:
        return ""


def _config_from_dict(d: dict[str, Any] | None, code: str = "") -> PipelineConfig | None:
    if not d:
        return None
    return PipelineConfig(
        operators=d.get("operators", []),
        field_flow=d.get("field_flow", ""),
        code=code,
        rationale=d.get("rationale", ""),
        recipe=d.get("recipe", {}),
        recipe_hash=d.get("recipe_hash", ""),
    )


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


def restore_tree(
    state: dict[str, Any],
    tree: TreeManager,
    agent_root: str | None = None,
    pipeline_filename: str = "pipeline.py",
) -> None:
    """把序列化的节点重建进 `tree`：恢复 root、统计、global_memory、step_count、id 计数器。

    `code` 不在 checkpoint 内——传入 `agent_root` 时按 `node_<id>/<pipeline_filename>` 从磁盘
    重读回 config.code（供变异起点 parent_code 注入）；不传或文件缺失则 code 保持 ""。
    """
    built: dict[int, PipelineNode] = {}
    parent_of: dict[int, int | None] = {}
    for nd in state["nodes"]:
        code = _load_node_code(agent_root, nd["id"], pipeline_filename) if nd["config"] else ""
        node = PipelineNode(config=_config_from_dict(nd["config"], code), parent=None)
        node.id = nd["id"]
        node.visits = nd["visits"]
        node.total_reward = nd["total_reward"]
        node.dead = nd.get("dead", False)
        node.review_result = _review_from_dict(nd["review_result"])
        # .get()：旧 checkpoint 没有这个键，续跑不应因此报错
        node.tier2_review_result = _review_from_dict(nd.get("tier2_review_result"))
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
    # 就地清空后从节点重建（不可整体替换实例：那会把 top_k 等渲染配置重置回默认值）
    tree.global_memory.clear()
    for nid in sorted(built):
        if built[nid].config is not None:
            tree.global_memory.update(built[nid])
    reseed_id_counter(max(built) + 1 if built else 0)
