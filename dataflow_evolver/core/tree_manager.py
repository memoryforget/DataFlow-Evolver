"""TreeManager：MCTS 主循环的树操作——select(UCB1) / expand / backpropagate，并维护全局记忆。

借鉴 DataMaster 的 MCTS：从根沿最大 UCT 下行到"可扩展"节点；扩展时挂上
PipelineAgent 产出的新配置；回传时沿祖先链累加奖励与访问数。

根节点（node_0）由编排器填成 Init node（持有真实流水线与首个分数），整棵树都是它的变异后代。
TreeManager 本身不负责建立 Init node，因此新建时根仍是空壳（config=None）——查询类方法
按"有没有 config"判断它是不是一次实验，不依赖任何外部标记。
"""
from __future__ import annotations

from dataflow_evolver.core.global_memory import GlobalMemory
from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.core.tree_node import PipelineNode
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("core.tree_manager")


class TreeManager:
    """持有根节点与搜索超参，提供选点 / 扩展 / 回传。"""

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
        self.root = PipelineNode()  # 空壳，待编排器填成 Init node
        # 钉死根的编号：文档、prompt 与日志都把起点称作 node_0，而计数器是全局的——只要有人
        # 在建树前多造一个节点，根就会漂成 node_3 之类，且只体现在目录名上，很难发现。
        # 子节点仍从计数器取号，不会与 0 相撞。
        self.root.id = 0
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
        """自根沿最大 UCT 下行，返回第一个可扩展节点。

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
        """从 node 沿父链累加 reward 与 visits（一路到根）。"""
        cur: PipelineNode | None = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += reward
            cur = cur.parent
        self.global_memory.update(node)
        self._step_count += 1   # 驱动探索常数衰减

    # ---- 便捷查询 ----
    def best_node(self) -> PipelineNode | None:
        """返回平均奖励最高的已评估、非 dead 节点（最终产物挑选用）。

        root（Init node）本身也是候选：若没有任何变异超过初始流水线，就该交付初始流水线
        本身，而不是"最不差的那个变异"。
        """
        best: PipelineNode | None = None
        for n in self.all_nodes():
            if n.visits == 0 or n.dead:
                continue
            if best is None or n.mean_reward > best.mean_reward:
                best = n
        return best

    def all_nodes(self) -> list[PipelineNode]:
        """前序遍历返回所有节点，含作为 Init node 的 root。

        仅当根还是空壳（config=None，即 Init node 尚未建立）时排除它——它不是一次实验，
        排除后调用方拿到的恒是"可比较、可重跑"的节点集合，无需各自再判断根的性质。
        """
        out: list[PipelineNode] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            stack.extend(n.children)
            if n is self.root and n.config is None:
                continue
            out.append(n)
        return out
