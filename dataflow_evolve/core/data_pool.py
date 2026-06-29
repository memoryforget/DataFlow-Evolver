"""DataPool：跨节点高质量数据的累积池（脚手架阶段）。

设计与决策见 `docs/design-decisions.md` §4。要点：
- 池与 MCTS 奖励【解耦】——奖励保持 per-node，池只「收割」通过门槛的节点数据。
- 池的价值 = 可直接用的成品 + 沉没成本回收 + 承载诊断驱动的累积覆盖。
- 交付物主线是「最优 pipeline」，池是可选副产品（默认 enabled=False）。

⚠️ 本文件目前是【接口骨架】：
- 数据结构、内容哈希去重（确定性、无依赖）已实现；
- 【准入策略】（pipeline 级阈值 / 行级筛选 / per-pipeline 上限的最终口径）与
  【main.py 第⑧步接线】留待有 dev 数据、完成实测设计后再实现，见文末 TODO。
不要在未做受控实验（等量对比 + 多样性量化，见设计文档 §6）前，把池当作
「比单 pipeline 更强」的既定结论使用。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dataflow_evolve.utils.logging import get_logger

logger = get_logger("core.data_pool")


@dataclass
class PoolConfig:
    """data_pool 配置段（对应 configs/default.yaml 的 `data_pool`）。"""

    enabled: bool = False
    score_threshold: float = 0.0        # τ：pipeline 级准入的 dev 分门槛
    max_rows_per_pipeline: int = 0      # 0 = 不限；>0 时为单 pipeline 贡献上限
    dedup: str = "content_hash"         # content_hash | none | embedding(未实现)
    output_path: str = ""               # 池落盘 jsonl 路径（空 = 不落盘）

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PoolConfig":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            score_threshold=float(d.get("score_threshold", 0.0)),
            max_rows_per_pipeline=int(d.get("max_rows_per_pipeline", 0)),
            dedup=str(d.get("dedup", "content_hash")),
            output_path=str(d.get("output_path", "")),
        )


@dataclass
class PoolEntry:
    """池中一条数据的溯源信息（便于复现与按来源剔除）。"""

    row: dict[str, Any]
    source_node_id: int
    content_hash: str


@dataclass
class AdmissionDecision:
    """一次 admit() 的结果摘要。"""

    admitted: int = 0
    rejected_dedup: int = 0
    rejected_cap: int = 0
    skipped_reason: str | None = None    # 非空表示整批被拒（如未过门槛/未启用）
    contributed_by: int | None = None


class DataPool:
    """累积通过门槛的节点数据，做去重与 per-pipeline 上限控制。

    用法（待 main.py 接线后）：
        pool = DataPool(cfg)
        decision = pool.admit(node_id=child.id, rows=..., dev_score=train.score,
                              review_passed=review.passed)
        ...
        pool.dump()  # 运行结束统一落盘

    现阶段仅供单元测试与设计参考，未接入主循环。
    """

    def __init__(self, config: PoolConfig) -> None:
        self.config = config
        self._entries: list[PoolEntry] = []
        self._seen_hashes: set[str] = set()
        self._rows_per_node: dict[int, int] = {}
        # 暖启动时由 main.py 设为 `baseline_dev + margin`（基线自标定门槛，见设计文档 §2.4）；
        # 为 None 时回退到绝对 `config.score_threshold`（冷启动）。
        self.admission_threshold: float | None = None

    # ------------------------------------------------------------------
    # 已实现：确定性内容哈希（无第三方依赖）
    # ------------------------------------------------------------------
    @staticmethod
    def content_hash(row: dict[str, Any]) -> str:
        """对一行做规范化 JSON 后的 sha1，用于精确去重。

        注意：这是【精确】去重（字面相同才判重），不做语义近邻去重。
        语义近邻去重需引入嵌入模型，列为未实现项（config.dedup == "embedding"）。
        """
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 已实现：pipeline 级门槛判定（最小口径，待实测后细化）
    # ------------------------------------------------------------------
    def passes_pipeline_gate(self, dev_score: float | None, review_passed: bool | None) -> bool:
        """pipeline 级准入：必须 review 通过且 dev 分 ≥ 门槛。

        门槛取值：暖启动用 `admission_threshold`（= baseline_dev + margin，基线自标定，无魔数）；
        未设置（冷启动）时回退到绝对 `config.score_threshold`。
        研究级的「边际贡献 / 贪心子模选择」见设计文档 §2.3，属未实现项（需在 池∪节点 上重训）。
        """
        if review_passed is False:
            return False
        if dev_score is None:
            return False
        threshold = (
            self.admission_threshold
            if self.admission_threshold is not None
            else self.config.score_threshold
        )
        return dev_score >= threshold

    # ------------------------------------------------------------------
    # 脚手架：单批数据准入（去重 + 上限已实现；行级质量筛选为 TODO）
    # ------------------------------------------------------------------
    def admit(
        self,
        node_id: int,
        rows: Iterable[dict[str, Any]],
        dev_score: float | None,
        review_passed: bool | None,
    ) -> AdmissionDecision:
        """尝试把某节点的数据并入池。

        当前实现：pipeline 级门槛 → 内容哈希去重 → per-pipeline 上限。
        ⚠️ 未实现的行级策略（见设计文档 §2.3）：
          - 命中失败模式的行上采样（bad-case-aware，不应被简单去重抹掉）；
          - 嵌入近邻去重。
        在补齐这些之前，admit() 只做「全部行 − 精确去重 − 上限」。
        """
        if not self.config.enabled:
            return AdmissionDecision(skipped_reason="pool disabled")
        if not self.passes_pipeline_gate(dev_score, review_passed):
            return AdmissionDecision(skipped_reason="pipeline gate not passed")

        decision = AdmissionDecision(contributed_by=node_id)
        cap = self.config.max_rows_per_pipeline
        for row in rows:
            if cap and self._rows_per_node.get(node_id, 0) >= cap:
                decision.rejected_cap += 1
                continue
            h = self.content_hash(row)
            if self.config.dedup == "content_hash" and h in self._seen_hashes:
                decision.rejected_dedup += 1
                continue
            self._entries.append(PoolEntry(row=row, source_node_id=node_id, content_hash=h))
            self._seen_hashes.add(h)
            self._rows_per_node[node_id] = self._rows_per_node.get(node_id, 0) + 1
            decision.admitted += 1
        logger.info(
            "DataPool.admit node=%s admitted=%d dedup=%d cap=%d total=%d",
            node_id, decision.admitted, decision.rejected_dedup, decision.rejected_cap, len(self._entries),
        )
        return decision

    def rows(self) -> list[dict[str, Any]]:
        return [e.row for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def dump(self, path: str | None = None) -> str | None:
        """把池落盘为 jsonl，返回路径；未指定且 config.output_path 为空则不落盘。"""
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
# TODO（待实测设计后实现，见 docs/design-decisions.md §4.3/§7）：
# 1. main.py 第⑧步：诊断后按 admit() 收割当前节点数据（仅当 review_passed 且 dev_score≥τ）。
# 2. 行级 bad-case-aware 上采样：命中 DiagnosticReport.failure_patterns 的行加权/保留。
# 3. 嵌入近邻去重（config.dedup == "embedding"）。
# 4. 去污染：并入前对 test benchmark 做 n-gram 重叠过滤。
# 5. （可选，仅当改主打 pool）边际贡献奖励：在 池∪节点 上重训以反映池质量。
# ---------------------------------------------------------------------------
