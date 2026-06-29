"""PipelineNode：MCTS 树节点，持有流水线配置、UCB1 统计与双反馈结果。

UCT 公式借鉴 DataMaster：Q/n + c*sqrt(ln(N)/n)，未访问节点返回 inf。
"""
from __future__ import annotations

import math
from itertools import count
from typing import Optional

from dataflow_evolve.core.schemas import (
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TrainResult,
)

_id_counter = count(1)


def reseed_id_counter(start: int) -> None:
    """重置全局节点 id 计数器（断点续跑时调用，使新节点 id 从 `start` 起，避免与已载入节点冲突）。"""
    global _id_counter
    _id_counter = count(start)


class PipelineNode:
    """一棵 MCTS 树中的节点；根节点为虚拟节点（config 为 None）。"""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        parent: Optional["PipelineNode"] = None,
    ) -> None:
        self.id: int = next(_id_counter)
        self.config = config
        self.parent = parent
        self.children: list[PipelineNode] = []

        # UCB1 统计
        self.visits: int = 0
        self.total_reward: float = 0.0

        # 暖启动时 root 记录基线 dev 分（仅 baseline root 使用；其余节点保持 None）
        self.baseline_dev_score: float | None = None

        # 双反馈结果（回写）
        self.review_result: ReviewResult | None = None
        self.train_result: TrainResult | None = None
        self.diagnostic_report: DiagnosticReport | None = None

    # ---- 树形结构 ----
    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def node_dir_name(self) -> str:
        """workspace 下的节点子目录名。"""
        return f"node_{self.id}"

    def add_child(self, config: PipelineConfig) -> "PipelineNode":
        child = PipelineNode(config=config, parent=self)
        self.children.append(child)
        return child

    # ---- UCB1 ----
    def uct_value(self, exploration_constant: float) -> float:
        """对父节点而言本节点的 UCT 分。未访问→inf（优先探索）；根节点无意义→inf。"""
        if self.visits == 0:
            return math.inf
        if self.parent is None or self.parent.visits == 0:
            return math.inf
        exploit = self.total_reward / self.visits
        explore = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploit + explore

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def best_child(self, exploration_constant: float) -> Optional["PipelineNode"]:
        """按 UCT 选最优子节点；无子节点返回 None。"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct_value(exploration_constant))

    def __repr__(self) -> str:
        return (
            f"PipelineNode(id={self.id}, visits={self.visits}, "
            f"mean_reward={self.mean_reward:.3f}, children={len(self.children)})"
        )
