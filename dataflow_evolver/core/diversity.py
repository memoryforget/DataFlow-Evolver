"""Deterministic diversity and recipe-coverage metrics for generated JSONL."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiversityMetrics:
    normalized_dup_rate: float = 0.0
    structural_dup_rate: float = 0.0
    coverage_score: float = 1.0
    largest_bucket_fraction: float = 0.0
    total: int = 0
    metadata_complete: bool = True


def normalize_content(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"[-+]?\d+(?:[.,]\d+)*", "<num>", text)
    text = re.sub(r"[^\w<>]+", "", text, flags=re.UNICODE)
    return text


def scan_diversity(
    path: str | Path,
    required_fields: list[str],
    recipe: dict[str, Any] | None = None,
) -> DiversityMetrics:
    recipe = recipe or {}
    enforce = recipe.get("mode") in {"hybrid", "synthetic"}
    content_counts: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    total = 0
    metadata_rows = 0
    p = Path(path)
    if not p.exists():
        return DiversityMetrics(total=0, coverage_score=0.0 if enforce else 1.0)

    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            total += 1
            parts = [normalize_content(row.get(name, "")) for name in required_fields]
            fingerprint = "|".join(part for part in parts if part)
            if fingerprint:
                content_counts[fingerprint] += 1
            graph = row.get("_operation_graph") or row.get("_archetype")
            bucket = row.get("_bucket")
            strategy = row.get("_generation_strategy")
            if graph:
                structure_counts[normalize_content(graph)] += 1
            if bucket:
                bucket_counts[str(bucket)] += 1
            if graph and bucket and strategy:
                metadata_rows += 1

    normalized_dup = _duplicate_rate(content_counts, total)
    structural_dup = _concentration_rate(structure_counts, total) if structure_counts else (1.0 if enforce else 0.0)
    largest = max(bucket_counts.values(), default=0) / total if total else 0.0
    expected = recipe.get("coverage_buckets", []) if enforce else []
    coverage = _coverage_score(bucket_counts, expected) if enforce else 1.0
    complete = (metadata_rows == total) if enforce else True
    if enforce and not complete:
        coverage *= metadata_rows / total if total else 0.0
    return DiversityMetrics(normalized_dup, structural_dup, coverage, largest, total, complete)


def _duplicate_rate(counts: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, (sum(counts.values()) - len(counts)) / total)


def _concentration_rate(counts: Counter[str], total: int) -> float:
    """0 when every row has a distinct structure; 1 when one structure owns every row."""
    if total <= 1:
        return 0.0
    return max(0.0, (max(counts.values(), default=0) - 1) / (total - 1))


def _coverage_score(observed: Counter[str], expected: list[dict[str, Any]]) -> float:
    quotas = {str(x.get("name")): int(x.get("quota", 0)) for x in expected if isinstance(x, dict)}
    quotas = {k: v for k, v in quotas.items() if k and v > 0}
    observed_total = sum(observed.values())
    quota_total = sum(quotas.values())
    if not quotas or observed_total <= 0 or quota_total <= 0:
        return 0.0
    # One minus total-variation distance between planned and observed distributions.
    keys = set(quotas) | set(observed)
    distance = 0.5 * sum(
        abs(observed.get(k, 0) / observed_total - quotas.get(k, 0) / quota_total) for k in keys
    )
    return max(0.0, min(1.0, 1.0 - distance))
