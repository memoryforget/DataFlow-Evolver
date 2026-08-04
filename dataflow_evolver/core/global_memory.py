"""GlobalMemory：跨节点共享知识，供 PipelineAgent 生成时参考（借鉴 DataMaster fetch_child_memory）。

聚合各已评估节点的 (config 摘要, 质量信号, diagnostic 摘要)，拼成一段简短文本注入 prompt，
让后续生成"站在已有结果之上"，避免重复劣质方案。

质量信号有两个来源，取决于 `search.reward_source`：
  - train 模式：`score` = 下游 dev 真分，另有 DiagnosticAgent 摘要；
  - review 模式：搜索期不训练，`score` 恒为 None，唯一信号是 ReviewAgent 的 Tier-1
    复合分与硬指标/失败信号。两者都渲染，缺的那一侧自动省略。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from dataflow_evolver.core.tree_node import PipelineNode


@dataclass
class MemoryEntry:
    node_id: int
    field_flow: str
    operators: list[dict[str, str]]   # [{"name": ..., "purpose": ...}, ...]
    score: float | None
    review_passed: bool | None
    diagnostic_summary: str | None
    # --- Tier-1 信号（review 模式下搜索期唯一的质量信号）---
    tier1_score: float | None = None
    dup_rate: float | None = None
    null_rate: float | None = None
    parse_valid_rate: float | None = None
    failure_signals: list[str] = field(default_factory=list)
    review_issues: list[str] = field(default_factory=list)

    @property
    def quality(self) -> float:
        """用于排序的有效质量分：有下游真分用真分，否则退回 Tier-1。"""
        if self.score is not None:
            return self.score
        return self.tier1_score or 0.0


class GlobalMemory:
    """已评估节点的轻量摘要集合。"""

    def __init__(
        self,
        top_k: int = 5,
        max_issues_per_entry: int = 3,
        issue_maxlen: int = 120,
    ) -> None:
        self._entries: dict[int, MemoryEntry] = {}
        self.top_k = top_k
        # issues 原文可达每节点两千余字（评审会逐样本举证）。两个旋钮控制注入粒度：
        #   max_issues_per_entry: -1=全部条目, 0=不渲染, N>0=前 N 条
        #   issue_maxlen:         <=0=不截断,   N>0=单条截断到 N 字
        # 想让记忆承载完整评审反馈就设 (-1, 0)，代价是 prompt 体积随 top_k 线性增长。
        self.max_issues_per_entry = max_issues_per_entry
        self.issue_maxlen = issue_maxlen

    def clear(self) -> None:
        """清空条目但保留渲染配置（checkpoint 续跑时从节点重建用）。"""
        self._entries.clear()

    def update(self, node: PipelineNode) -> None:
        """从一个节点的当前状态抽取摘要写入记忆。"""
        if node.config is None:
            return
        op_entries = [
            {"name": op.get("name", "?"), "purpose": op.get("purpose", "")}
            for op in node.config.operators
        ]
        review = node.review_result
        train = node.train_result
        diagnostic = node.diagnostic_report
        diagnostic_summary = None
        if diagnostic and (diagnostic.failure_patterns or diagnostic.actionable_suggestions):
            diagnostic_summary = diagnostic.summary
        self._entries[node.id] = MemoryEntry(
            node_id=node.id,
            field_flow=node.config.field_flow,
            operators=op_entries,
            score=train.score if train else None,
            review_passed=review.passed if review else None,
            diagnostic_summary=diagnostic_summary,
            tier1_score=review.tier1_score if review else None,
            dup_rate=review.dup_rate if review else None,
            null_rate=review.null_rate if review else None,
            parse_valid_rate=review.parse_valid_rate if review else None,
            failure_signals=list(review.failure_signals) if review else [],
            review_issues=list(review.issues) if review else [],
        )

    def top_entries(self, exclude_node_ids: Iterable[int] | None = None) -> list[MemoryEntry]:
        """按有效质量分降序取前 top_k（有下游真分的排在只有 Tier-1 的前面）。

        `exclude_node_ids`：排除这些节点。调用方已在 prompt 里为它们准备了专属块，若同一节点
        两处并存，会被赋予相反角色——记忆列表说"分高，参考它"，兄弟清单说"试过了，别跟它一样"。
        """
        excluded = set(exclude_node_ids or ())
        candidates = [e for e in self._entries.values() if e.node_id not in excluded]
        return sorted(
            candidates,
            key=lambda e: (e.score is not None, e.quality),
            reverse=True,
        )[: self.top_k]

    def _render_entry(self, e: MemoryEntry) -> str:
        ops_desc = ", ".join(
            f"{op['name']}({op['purpose']})" if op.get("purpose") else op["name"]
            for op in e.operators
        )
        parts = [f"- [node {e.node_id}] flow={e.field_flow} | ops=[{ops_desc}]"]
        if e.score is not None:
            parts.append(f"score={e.score:.3f}")
        if e.tier1_score is not None:
            parts.append(f"tier1={e.tier1_score:.3f}")
        if e.review_passed is not None:
            parts.append(f"评审={'通过' if e.review_passed else '未通过'}")
        hard = [
            (f"dup={e.dup_rate:.2f}" if e.dup_rate is not None else ""),
            (f"null={e.null_rate:.2f}" if e.null_rate is not None else ""),
            (f"parse={e.parse_valid_rate:.2f}" if e.parse_valid_rate is not None else ""),
        ]
        hard_desc = " ".join(p for p in hard if p)
        if hard_desc:
            parts.append(hard_desc)
        if e.failure_signals:
            parts.append(f"信号: {','.join(e.failure_signals)}")
        if e.diagnostic_summary:
            parts.append(f"诊断: {e.diagnostic_summary}")
        if self.max_issues_per_entry != 0:
            # 保持评审给出的原始顺序（该顺序并非按严重度排列，评审常先写正面观察再列问题）。
            candidates = e.review_issues
            if self.max_issues_per_entry > 0:
                candidates = candidates[: self.max_issues_per_entry]
            picked = [
                (s[: self.issue_maxlen] + "…")
                if 0 < self.issue_maxlen < len(s)
                else s
                for s in candidates
            ]
            if picked:
                parts.append("评审反馈: " + "；".join(picked))
        return " | ".join(parts)

    def render(self, exclude_node_ids: Iterable[int] | None = None) -> str:
        """渲染为注入 PipelineAgent prompt 的文本摘要。"""
        entries = self.top_entries(exclude_node_ids=exclude_node_ids)
        if not entries:
            return "（暂无历史记忆：这是首批节点。）"
        # 表头如实反映排序依据：review 模式下并没有下游得分，写"下游得分"会误导 LLM。
        basis = "下游得分" if any(e.score is not None for e in entries) else "Tier-1 复合分"
        lines = [f"历史已评估流水线（按{basis}降序，供参考与避免重复）："]
        lines.extend(self._render_entry(e) for e in entries)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)
