"""DataPool：跨节点高质量数据的累积池（与 MCTS 奖励解耦的可选副产品）。

设计与决策见 `docs/design-decisions.md` §4 及 §2（准入门槛尺度无关化）。要点：
- 池与奖励【解耦】——奖励 per-node，池只「收割」通过门槛的节点数据。
- 交付物主线是「最优 pipeline」，池是可选副产品（默认 enabled=False）。

【运行期暂存 → dump 前统一过滤】（2026-06-28 重构）：
为支持尺度无关、跨 benchmark 不重调的门槛，准入判定**不再逐节点即时进行**，而是
运行期只 `stage()` 暂存每个节点的 (dev_score, review_passed, rows)；运行结束 `dump()`
时按固定顺序统一处理：

  1. 门槛筛节点：
     - 暖启动（warm，set_warm_gate）：`node_dev > baseline_dev + margin`（**严格大于**，
       margin=0 即"严格打败基线"；等号的 baseline 自身被排除，与"baseline 不进池"一致）。
     - 冷启动（cold，默认）：`node_dev ≥ max(本轮全部节点 dev 分的 q 分位数, score_threshold)`
       （q 默认中位数 0.5；分位数从本轮分布自算 → 无魔数、跨 benchmark 同一套配置）。
  2. 跨源精确去重（content_hash）。
  3. per-pipeline 占比上限（`max_pool_fraction_per_pipeline`，默认 30%）：
     仅当来源数 ≥ ⌈1/f⌉ 时才生效（否则占比约束不可行——单一来源必占 100%）；
     用 water-filling 迭代把超额来源裁到 ≤ f×池总量。
     （可选叠加绝对上限 `max_rows_per_pipeline`，默认 0=不限。）

⚠️ 未实现（见文末 TODO）：行级 bad-case 上采样、嵌入近邻去重、去污染。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dataflow_evolve.utils.logging import get_logger

logger = get_logger("core.data_pool")

GATE_QUANTILE = "quantile"            # 冷启动：分位数门槛
GATE_RELATIVE_BASELINE = "relative_baseline"  # 暖启动：> baseline + margin


@dataclass
class PoolConfig:
    """data_pool 配置段（对应 configs/default.yaml 的 `data_pool`）。"""

    enabled: bool = False
    score_threshold: float = 0.0            # 冷启动绝对地板（与分位数取 max；0=不设地板）
    quantile: float = 0.5                   # 冷启动分位数门槛（0.5=中位数；尺度无关元参数）
    max_pool_fraction_per_pipeline: float = 0.3  # per-pipeline 占池比例上限（0=不限）
    max_rows_per_pipeline: int = 0          # 可选叠加的绝对行数上限（0=不限）
    dedup: str = "content_hash"             # content_hash | none | embedding(未实现)
    output_path: str = ""                   # 池落盘 jsonl 路径（空=不落盘）

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PoolConfig":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            score_threshold=float(d.get("score_threshold", 0.0)),
            quantile=float(d.get("quantile", 0.5)),
            max_pool_fraction_per_pipeline=float(d.get("max_pool_fraction_per_pipeline", 0.3)),
            max_rows_per_pipeline=int(d.get("max_rows_per_pipeline", 0)),
            dedup=str(d.get("dedup", "content_hash")),
            output_path=str(d.get("output_path", "")),
        )


@dataclass
class StagedNode:
    """运行期暂存的一个候选节点（dump 前才决定是否准入）。"""

    node_id: int
    dev_score: float
    rows: list[dict[str, Any]]


@dataclass
class PoolEntry:
    """池中一条数据的溯源信息（便于复现与按来源剔除）。"""

    row: dict[str, Any]
    source_node_id: int
    content_hash: str


@dataclass
class FinalizeStats:
    """一次 finalize 的统计摘要（落盘日志 / 测试用）。"""

    threshold: float = 0.0
    gate_mode: str = GATE_QUANTILE
    staged_nodes: int = 0
    admitted_nodes: int = 0
    rows_before_dedup: int = 0
    rejected_dedup: int = 0
    rejected_cap: int = 0
    rows_final: int = 0
    per_source: dict[int, int] = field(default_factory=dict)


class DataPool:
    """累积通过门槛的节点数据：运行期 stage()，结束 dump() 统一过滤+落盘。

    用法（main.py）：
        pool = DataPool(cfg)            # 默认冷启动分位数门槛
        pool.set_warm_gate(baseline_dev, margin)   # 暖启动改用 > baseline+margin
        ...
        pool.stage(node_id, rows, dev_score, review_passed)   # 每个评估过的节点
        pool.dump()                     # 运行结束统一 finalize + 落盘
    """

    def __init__(self, config: PoolConfig) -> None:
        self.config = config
        self._staged: list[StagedNode] = []
        self._entries: list[PoolEntry] = []      # finalize 后填充
        # 门槛模式：默认冷启动分位数；暖启动由 set_warm_gate 切换
        self.gate_mode: str = GATE_QUANTILE
        self.baseline_dev: float | None = None
        self.margin: float = 0.0
        self.last_stats: FinalizeStats | None = None

    # ------------------------------------------------------------------
    def set_warm_gate(self, baseline_dev: float, margin: float = 0.0) -> None:
        """暖启动：门槛改为 `node_dev > baseline_dev + margin`（严格大于）。"""
        self.gate_mode = GATE_RELATIVE_BASELINE
        self.baseline_dev = baseline_dev
        self.margin = margin

    @staticmethod
    def content_hash(row: dict[str, Any]) -> str:
        """对一行做规范化 JSON 后的 sha1，用于【精确】去重（非语义近邻）。"""
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def stage(
        self,
        node_id: int,
        rows: Iterable[dict[str, Any]],
        dev_score: float | None,
        review_passed: bool | None,
    ) -> bool:
        """暂存一个节点的数据（不做门槛判定，留到 dump 时统一处理）。

        仅暂存"已过 review 且已训练得分"的节点；失败/未训练节点不入候选。返回是否暂存。
        """
        if not self.config.enabled:
            return False
        if not review_passed or dev_score is None:
            return False
        self._staged.append(StagedNode(node_id=node_id, dev_score=float(dev_score),
                                       rows=[r for r in rows if isinstance(r, dict)]))
        return True

    # ------------------------------------------------------------------
    def _threshold(self, scores: list[float]) -> float:
        """据门槛模式计算准入阈值。"""
        if self.gate_mode == GATE_RELATIVE_BASELINE:
            return (self.baseline_dev or 0.0) + self.margin
        # 冷启动：本轮分位数与绝对地板取 max
        q = _quantile(sorted(scores), self.config.quantile) if scores else 0.0
        return max(q, self.config.score_threshold)

    def _admits(self, dev_score: float, threshold: float) -> bool:
        """暖启动严格 `>`（排除等号的 baseline）；冷启动 `>=`（并列时不空池）。"""
        if self.gate_mode == GATE_RELATIVE_BASELINE:
            return dev_score > threshold
        return dev_score >= threshold

    # ------------------------------------------------------------------
    def finalize(self) -> list[PoolEntry]:
        """对暂存节点统一执行：门槛筛 → 去重 → 占比裁剪，返回最终 entries。"""
        stats = FinalizeStats(gate_mode=self.gate_mode, staged_nodes=len(self._staged))
        scores = [s.dev_score for s in self._staged]
        threshold = self._threshold(scores)
        stats.threshold = threshold

        # 1. 门槛筛节点（按 node_id 稳定排序，保证去重"先到先得"可复现）
        admitted = sorted(
            (s for s in self._staged if self._admits(s.dev_score, threshold)),
            key=lambda s: s.node_id,
        )
        stats.admitted_nodes = len(admitted)

        # 2. 跨源精确去重，按来源归集
        seen: set[str] = set()
        per_source_rows: dict[int, list[PoolEntry]] = {}
        for node in admitted:
            for row in node.rows:
                stats.rows_before_dedup += 1
                h = self.content_hash(row)
                if self.config.dedup == "content_hash" and h in seen:
                    stats.rejected_dedup += 1
                    continue
                seen.add(h)
                per_source_rows.setdefault(node.node_id, []).append(
                    PoolEntry(row=row, source_node_id=node.node_id, content_hash=h)
                )

        # 3. 占比裁剪（+ 可选绝对上限）
        kept = self._apply_caps(per_source_rows, stats)

        self._entries = kept
        stats.rows_final = len(kept)
        stats.per_source = {nid: len(rows) for nid, rows in _group_by_source(kept).items()}
        self.last_stats = stats
        logger.info(
            "DataPool.finalize mode=%s thr=%.4f staged=%d admitted=%d "
            "rows=%d dedup=-%d cap=-%d final=%d per_source=%s",
            stats.gate_mode, stats.threshold, stats.staged_nodes, stats.admitted_nodes,
            stats.rows_before_dedup, stats.rejected_dedup, stats.rejected_cap,
            stats.rows_final, stats.per_source,
        )
        return kept

    def _apply_caps(
        self, per_source_rows: dict[int, list[PoolEntry]], stats: FinalizeStats
    ) -> list[PoolEntry]:
        """先应用可选绝对上限，再用 water-filling 应用占比上限。"""
        counts = {nid: len(rows) for nid, rows in per_source_rows.items()}

        # 3a. 可选绝对上限（0=不限）
        abs_cap = self.config.max_rows_per_pipeline
        if abs_cap and abs_cap > 0:
            for nid in counts:
                counts[nid] = min(counts[nid], abs_cap)

        # 3b. 占比上限：仅当来源数 ≥ ⌈1/f⌉ 时可行（否则单一来源必占 100%）
        f = self.config.max_pool_fraction_per_pipeline
        n_sources = len([nid for nid, c in counts.items() if c > 0])
        if f and 0 < f < 1 and n_sources >= math.ceil(1.0 / f):
            counts = _water_fill_cap(counts, f)

        # 按裁剪后的配额回填（保留每源前 N 行，确定性）
        kept: list[PoolEntry] = []
        for nid in sorted(per_source_rows.keys()):
            quota = counts.get(nid, 0)
            rows = per_source_rows[nid]
            stats.rejected_cap += max(0, len(rows) - quota)
            kept.extend(rows[:quota])
        return kept

    # ------------------------------------------------------------------
    def rows(self) -> list[dict[str, Any]]:
        return [e.row for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def dump(self, path: str | None = None) -> str | None:
        """finalize 后落盘为 jsonl；未指定且 config.output_path 为空则只 finalize 不落盘。"""
        self.finalize()
        target = path or self.config.output_path
        if not target:
            return None
        out = Path(target)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(e.row, ensure_ascii=False) + "\n")
        logger.info("DataPool 落盘 %d 行 → %s", len(self._entries), out)
        return str(out)


# ---------------------------------------------------------------------------
def _quantile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数（q∈[0,1]）；空列表返回 0.0。"""
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _water_fill_cap(counts: dict[int, int], f: float) -> dict[int, int]:
    """水填充：把各来源裁到每源占比 ≤ f（max-min 公平），解析求平衡水位。

    正确性依据：设裁剪 top-k 个高产源到统一水位 L，其余 (n-k) 个源不动。
    平衡点须满足 L = f × 总量，而总量 = (未裁源之和 U) + k×L：
        L = f × (U + k×L)  ⟹  L = f×U / (1 − f×k)
    合法当且仅当第 k 个源 ≥ L ≥ 第 (k+1) 个源。从 k=1 递增试探即得唯一解。

    k 最多到 n−1（至少留一个未裁源作平衡基准，否则 U=0 会把全部裁成 0）。
    若源数不足以让任一单源 ≤ f（如 3 源要求 ≤30% 不可行），走均分兜底，
    绝不螺旋至零。注意：调用方有 `n_sources ≥ ceil(1/f)` 守卫，兜底通常不触发。
    """
    if not counts or f <= 0 or f >= 1:
        return counts

    counts = dict(counts)
    vals = sorted(counts.values(), reverse=True)
    n = len(vals)

    # k = 被裁的高产源数，从 1 递增试探平衡水位
    for k in range(1, n):
        if 1 - f * k <= 0:
            break  # 源数不足，约束在此 k 起不可行
        U = sum(vals[k:])                     # 未裁源之和
        L = f * U / (1 - f * k)               # 平衡水位
        upper = vals[k - 1]                   # 第 k 个（被裁）源，须 ≥ L
        lower = vals[k]                       # 第 k+1 个（未裁）源，须 ≤ L
        if lower <= L <= upper:
            cap = int(L)
            return {nid: min(c, cap) for nid, c in counts.items()}

    # 不可行（源数 < ceil(1/f)）：均分兜底，避免螺旋至零
    share = max(int(f * sum(vals)), 1)
    return {nid: min(c, share) for nid, c in counts.items()}


def _group_by_source(entries: list[PoolEntry]) -> dict[int, list[PoolEntry]]:
    out: dict[int, list[PoolEntry]] = {}
    for e in entries:
        out.setdefault(e.source_node_id, []).append(e)
    return out


# ---------------------------------------------------------------------------
# TODO（见 docs/design-decisions.md §4.3/§7）：
# 1. 行级 bad-case-aware 上采样：命中 DiagnosticReport.failure_patterns 的行加权/保留
#    （注意与去重/占比裁剪冲突——失败模式行故意相似，应在去重后且豁免裁剪）。
# 2. 嵌入近邻去重（config.dedup == "embedding"）。
# 3. 去污染：finalize 前对 test benchmark 做 n-gram 重叠过滤（应最先执行）。
# ---------------------------------------------------------------------------
