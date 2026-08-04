"""Deterministic row-level enforcement for external/hybrid/synthetic modes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceAuditResult:
    total: int = 0
    external_fraction: float = 0.0
    synthetic_fraction: float = 0.0
    unknown_fraction: float = 0.0
    source_id_valid_rate: float = 1.0
    seed_link_rate: float = 1.0
    record_id_valid_rate: float = 1.0
    compliance_score: float = 1.0
    issues: list[str] = field(default_factory=list)


def audit_mode_dataset(path: str | Path, recipe: dict[str, Any] | None) -> SourceAuditResult:
    if not recipe or recipe.get("mode") not in {"external", "hybrid", "synthetic"}:
        return SourceAuditResult()
    mode = str(recipe["mode"])
    declared = {
        str(src.get("source_id"))
        for src in recipe.get("sources", [])
        if isinstance(src, dict) and src.get("source_id")
    }
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    total = len(rows)
    if not total:
        return SourceAuditResult(total=0, compliance_score=0.0, issues=["数据集为空，无法验证 mode 来源契约"])

    external = [r for r in rows if r.get("_source_type") == "external"]
    synthetic = [r for r in rows if r.get("_source_type") == "synthetic"]
    unknown = total - len(external) - len(synthetic)
    ext_fraction = len(external) / total
    syn_fraction = len(synthetic) / total
    unknown_fraction = unknown / total

    valid_source_ids = sum(
        1
        for row in rows
        if (
            row.get("_source_type") == "external"
            and str(row.get("_source_id", "")) in declared
        )
        or (
            row.get("_source_type") == "synthetic"
            and bool(str(row.get("_source_id", "")).strip())
        )
    )
    source_id_rate = valid_source_ids / total
    record_ids = [str(r.get("_record_id", "")).strip() for r in rows]
    nonempty_unique = len({x for x in record_ids if x})
    record_id_rate = nonempty_unique / total

    external_ids = {str(r.get("_record_id", "")).strip() for r in external if r.get("_record_id")}
    linked = sum(1 for r in synthetic if str(r.get("_seed_id", "")).strip() in external_ids)
    seed_link_rate = linked / len(synthetic) if synthetic else (0.0 if mode == "hybrid" else 1.0)

    issues: list[str] = []
    ratio_score = 1.0
    if unknown:
        issues.append(f"{unknown}/{total} 行缺少合法 _source_type（只能是 external 或 synthetic）")
    if source_id_rate < 1.0:
        issues.append("存在缺失/非法 _source_id；external 行必须引用 recipe.sources 中声明的 source_id")
    if record_id_rate < 1.0:
        issues.append("每行必须有全局唯一且非空的 _record_id")

    if mode == "external":
        ratio_score = ext_fraction
        if len(external) != total:
            issues.append("external mode 要求最终数据 100% 为 _source_type=external")
    elif mode == "synthetic":
        ratio_score = syn_fraction
        if len(synthetic) != total:
            issues.append("synthetic mode 要求最终数据 100% 为 _source_type=synthetic")
    else:
        policy = recipe.get("mixing_policy", {})
        ext_min = float(policy.get("external_min_fraction", 0.0) or 0.0)
        syn_min = float(policy.get("synthetic_min_fraction", 0.0) or 0.0)
        ext_score = min(1.0, ext_fraction / ext_min) if ext_min > 0 else 0.0
        syn_score = min(1.0, syn_fraction / syn_min) if syn_min > 0 else 0.0
        ratio_score = ext_score * syn_score
        if ext_fraction < ext_min:
            issues.append(f"hybrid external 占比 {ext_fraction:.1%} 低于 recipe 下限 {ext_min:.1%}")
        if syn_fraction < syn_min:
            issues.append(f"hybrid synthetic 占比 {syn_fraction:.1%} 低于 recipe 下限 {syn_min:.1%}")
        if seed_link_rate < 1.0:
            issues.append(f"hybrid synthetic 行的有效 _seed_id 追溯率仅 {seed_link_rate:.1%}，要求 100%")

    score = (
        (1.0 - unknown_fraction)
        * source_id_rate
        * record_id_rate
        * ratio_score
        * (seed_link_rate if mode == "hybrid" else 1.0)
    )
    return SourceAuditResult(
        total=total,
        external_fraction=ext_fraction,
        synthetic_fraction=syn_fraction,
        unknown_fraction=unknown_fraction,
        source_id_valid_rate=source_id_rate,
        seed_link_rate=seed_link_rate,
        record_id_valid_rate=record_id_rate,
        compliance_score=max(0.0, min(1.0, score)),
        issues=issues,
    )
