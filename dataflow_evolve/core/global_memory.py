"""GlobalMemory：跨节点共享知识，供 PipelineAgent 生成时参考（借鉴 DataMaster fetch_child_memory）。

聚合各已评估节点的 (config 摘要, score, diagnostic 摘要)，拼成一段简短文本注入 prompt，
让后续生成"站在已有结果之上"，避免重复劣质方案。
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_evolve.core.tree_node import PipelineNode


@dataclass
class MemoryEntry:
    node_id: int
    field_flow: str
    operators: list[str]
    score: float | None
    review_passed: bool | None
    diagnostic_summary: str | None


class GlobalMemory:
    """已评估节点的轻量摘要集合。"""

    def __init__(self, top_k: int = 5) -> None:
        self._entries: dict[int, MemoryEntry] = {}
        self.top_k = top_k

    def update(self, node: PipelineNode) -> None:
        """从一个节点的当前状态抽取摘要写入记忆。"""
        if node.config is None:
            return
        op_names = [op.get("name", "?") for op in node.config.operators]
        score = node.train_result.score if node.train_result else None
        review_passed = node.review_result.passed if node.review_result else None
        diag = node.diagnostic_report.summary if node.diagnostic_report else None
        self._entries[node.id] = MemoryEntry(
            node_id=node.id,
            field_flow=node.config.field_flow,
            operators=op_names,
            score=score,
            review_passed=review_passed,
            diagnostic_summary=diag,
        )

    def top_entries(self) -> list[MemoryEntry]:
        """按 score 降序取前 top_k（无分数的排在后面）。"""
        return sorted(
            self._entries.values(),
            key=lambda e: (e.score is not None, e.score or 0.0),
            reverse=True,
        )[: self.top_k]

    def render(self) -> str:
        """渲染为注入 PipelineAgent prompt 的文本摘要。"""
        entries = self.top_entries()
        if not entries:
            return "（暂无历史记忆：这是首批节点。）"
        lines = ["历史已评估流水线（按下游得分降序，供参考与避免重复）："]
        for e in entries:
            score_str = f"{e.score:.3f}" if e.score is not None else "未训练"
            line = (
                f"- [node {e.node_id}] flow={e.field_flow} | ops={e.operators} "
                f"| score={score_str}"
            )
            if e.diagnostic_summary:
                line += f" | 诊断: {e.diagnostic_summary}"
            lines.append(line)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)
