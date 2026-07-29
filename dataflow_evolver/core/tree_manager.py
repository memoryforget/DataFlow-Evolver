"""TreeManager：MCTS 主循环的树操作——select(UCB1) / expand / backpropagate，并维护全局记忆。

借鉴 DataMaster 的 MCTS：从虚拟根沿最大 UCT 下行到"可扩展"节点；扩展时挂上
PipelineAgent 产出的新配置；回传时沿祖先链累加奖励与访问数。
"""
from __future__ import annotations

from dataflow_evolver.core.global_memory import GlobalMemory
from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.core.tree_node import PipelineNode
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("core.tree_manager")


class TreeManager:
    """持有虚拟根与超参，提供选点 / 扩展 / 回传。"""

    def __init__(
        self,
        exploration_constant: float = 1.414,
        max_children_per_node: int = 4,
        exploration_decay: float = 1.0,
        exploration_min: float = 0.0,
        memory_top_k: int = 5,
        memory_max_issues: int = 3,
        memory_issue_maxlen: int = 120,
    ) -> None:
        self.root = PipelineNode()  # 虚拟根，config=None
        self.exploration_constant = exploration_constant
        self.max_children_per_node = max_children_per_node
        # 探索常数 c 衰减（借鉴 DataMaster）：早期多探索、后期多利用。
        # c_eff = max(exploration_min, c0 * decay**已回传次数)。decay=1.0 即不衰减（默认，行为不变）。
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self._step_count = 0        # 已回传(评估)次数，驱动衰减
        self.global_memory = GlobalMemory(
            top_k=memory_top_k,
            max_issues_per_entry=memory_max_issues,
            issue_maxlen=memory_issue_maxlen,
        )

    def current_exploration(self) -> float:
        """当前有效探索常数 c（随已评估节点数指数衰减，下限 exploration_min）。"""
        c = self.exploration_constant * (self.exploration_decay ** self._step_count)
        return max(self.exploration_min, c)

    def set_baseline_root(
        self, config: PipelineConfig, baseline_dev_score: float
    ) -> PipelineNode:
        """暖启动：把虚拟空根替换为持有基线配置的真实根（见 docs/design-decisions.md §2.4）。

        - root 持有基线 `config`（其 `code` 即适配后的基线源码，供首层子节点作变异起点）；
        - `visits=1`，`total_reward=0.0`——基线相对自身的 Relative Gain 恒为 0，是 UCB 的零点；
        - `baseline_dev_score` 记到 root，便于复现与日志（奖励换算的真值由 RewardPolicy 持有）。
        """
        self.root.config = config
        self.root.visits = 1
        self.root.total_reward = 0.0
        self.root.baseline_dev_score = baseline_dev_score
        self.global_memory.update(self.root)
        logger.info(
            "基线作 root：%s（baseline_dev_score=%.4f, visits=1, reward=0）",
            self.root.node_dir_name, baseline_dev_score,
        )
        return self.root

    def is_expandable(self, node: PipelineNode) -> bool:
        """存活（非 dead）子节点未满配额即可扩展。

        dead 子节点（生成/执行/评审/训练失败）不占配额——否则一个父节点被几次失败
        塞满后就再也无法在它下面尝试新流水线（见 review M1）。
        """
        live = sum(1 for c in node.children if not c.dead)
        return live < self.max_children_per_node

    def reserve_child(self, parent: PipelineNode) -> PipelineNode:
        """先占一个子节点（config 暂为 None）以拿到稳定的 node_dir 名，供生成写入。

        生成成功后由调用方填回 `child.config`；失败则该节点保留为一次已用的扩展尝试。
        """
        child = PipelineNode(config=None, parent=parent)
        parent.children.append(child)
        logger.info("预留节点：parent=node_%s → %s", parent.id, child.node_dir_name)
        return child

    def discard_child(self, child: PipelineNode) -> None:
        """撤销一个已预留但因基础设施失败未成功扩展的子节点。

        与「生成/执行/评审/训练失败」（置 dead 且回传 penalty）不同：基础设施失败（429/超时/
        连接等外部原因）不反映流水线质量，故从父节点直接摘除——不置 dead、不回传奖励、不占配额，
        避免污染 UCB 统计（见 main._step 的 AgentInfraError 处理）。
        """
        parent = child.parent
        if parent is not None and child in parent.children:
            parent.children.remove(child)

    def select(self) -> PipelineNode:
        """自虚拟根沿最大 UCT 下行，返回第一个可扩展节点。

        根节点总是可扩展（直到铺满首层）；否则沿 best_child 下探，遇到可扩展节点即停。
        探索常数随已评估节点数衰减（见 current_exploration）。
        """
        c = self.current_exploration()
        node = self.root
        while True:
            if self.is_expandable(node):
                return node
            child = node.best_child(c)
            if child is None:
                return node
            node = child

    def expand(self, parent: PipelineNode, config: PipelineConfig) -> PipelineNode:
        """在 parent 下挂一个新子节点。"""
        child = parent.add_child(config)
        logger.info("扩展：parent=node_%s → child=%s", parent.id, child.node_dir_name)
        return child

    def backpropagate(self, node: PipelineNode, reward: float) -> None:
        """从 node 沿父链累加 reward 与 visits（含虚拟根）。"""
        cur: PipelineNode | None = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += reward
            cur = cur.parent
        self.global_memory.update(node)
        self._step_count += 1   # 驱动探索常数衰减

    # ---- 便捷查询 ----
    def best_leaf(self) -> PipelineNode | None:
        """返回平均奖励最高的已访问、非 dead、非根节点（最终产物挑选用）。

        暖启动兜底（review H2）：若 root 是基线节点（有 config + baseline_dev_score），
        且没有任何子节点打败基线（最佳子节点平均奖励 ≤ 0，Relative Gain 下即未超基线），
        则回退返回基线 root——避免把"比已有强基线还差"的流水线当成最优交付。
        """
        best: PipelineNode | None = None
        stack = list(self.root.children)
        while stack:
            n = stack.pop()
            stack.extend(n.children)
            if n.visits == 0 or n.dead:
                continue
            if best is None or n.mean_reward > best.mean_reward:
                best = n
        # 暖启动：基线作 root 时，没人超过基线就交付基线本身
        if self.root.config is not None and self.root.baseline_dev_score is not None:
            if best is None or best.mean_reward <= 0.0:
                logger.info("无节点打败基线，best_leaf 回退到基线 root：%s", self.root.node_dir_name)
                return self.root
        return best

    def all_nodes(self) -> list[PipelineNode]:
        """前序遍历返回除虚拟根外的所有节点。"""
        out: list[PipelineNode] = []
        stack = list(self.root.children)
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out
