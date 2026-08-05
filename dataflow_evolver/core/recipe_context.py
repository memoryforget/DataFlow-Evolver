"""Structured evidence passed to RecipeAgent during tree evolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataflow_evolver.core.schemas import ReviewResult
from dataflow_evolver.core.tree_node import PipelineNode


@dataclass
class RecipeContext:
    parent: dict[str, Any] | None = None
    siblings: list[dict[str, Any]] = field(default_factory=list)
    global_memory: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {
            "top_recipes": [],
            "pareto_recipes": [],
            "failed_patterns": [],
            "observed_effects": [],
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "siblings": self.siblings,
            "global_memory": self.global_memory,
        }


def build_recipe_context(
    parent: PipelineNode | None,
    siblings: list[PipelineNode],
    all_nodes: list[PipelineNode],
    *,
    top_k: int = 5,
    compliance_threshold: float = 0.9,
) -> RecipeContext:
    """Summarize tree evidence without injecting complete sibling pipelines."""
    parent_block = (
        _parent_record(parent, compliance_threshold) if parent and parent.config else None
    )
    sibling_blocks = [
        _compact_outcome(node)
        for node in siblings
        if node.config is not None and node.review_result is not None
    ]
    excluded = {node.id for node in siblings}
    if parent is not None:
        excluded.add(parent.id)
    candidates = [
        node for node in all_nodes
        if node.id not in excluded and node.config is not None and node.review_result is not None
    ]
    credible = [
        node for node in candidates
        if _is_credible(node, compliance_threshold)
    ]
    top = sorted(credible, key=_outcome_score, reverse=True)[:top_k]
    pareto = _pareto_front(credible)[:top_k]
    failed = sorted(
        [node for node in candidates if not _is_credible(node, compliance_threshold)],
        key=lambda n: (n.dead, 1.0 - _compliance(n.review_result)),
        reverse=True,
    )[:top_k]
    effects = _observed_effects(credible, top_k)
    return RecipeContext(
        parent=parent_block,
        siblings=sibling_blocks,
        global_memory={
            "top_recipes": [_compact_outcome(node) for node in top],
            "pareto_recipes": [_compact_outcome(node) for node in pareto],
            "failed_patterns": [_failure_record(node) for node in failed],
            "observed_effects": effects,
        },
    )


def implementation_feedback(review: ReviewResult | None) -> dict[str, Any]:
    """Keep only implementation-facing evidence for PipelineAgent."""
    if review is None:
        return {}
    return {
        "schema_score": review.schema_score,
        "parse_valid_rate": review.parse_valid_rate,
        "null_rate": review.null_rate,
        "normalized_dup_rate": review.normalized_dup_rate,
        "structural_dup_rate": review.structural_dup_rate,
        "coverage_score": review.coverage_score,
        "source_id_valid_rate": review.source_id_valid_rate,
        "seed_link_rate": review.seed_link_rate,
        "source_compliance_score": review.source_compliance_score,
        "failure_signals": list(review.failure_signals),
        "issues": list(review.issues[:5]),
    }


def _parent_record(node: PipelineNode, compliance_threshold: float) -> dict[str, Any]:
    review = node.review_result
    train = node.train_result
    return {
        "recipe": node.config.recipe if node.config else {},
        "recipe_hash": node.config.recipe_hash if node.config else "",
        "compliance": _compliance_record(review),
        "quality": _quality_record(review),
        "downstream": {
            "metric": train.metric_name if train else None,
            "score": train.score if train and not train.failed else None,
            "credible": bool(review and _is_credible(node, compliance_threshold)),
            "diagnostic": _diagnostic_lines(node),
        },
    }


def _compact_outcome(node: PipelineNode) -> dict[str, Any]:
    review = node.review_result
    train = node.train_result
    recipe = node.config.recipe if node.config else {}
    parent_recipe = node.parent.config.recipe if node.parent and node.parent.config else {}
    return {
        "node_id": node.id,
        "recipe_hash": node.config.recipe_hash if node.config else "",
        "mode": recipe.get("mode"),
        "recipe_delta": _recipe_delta(parent_recipe, recipe),
        "compliance_score": _compliance(review),
        "coverage_score": review.coverage_score if review else None,
        "downstream_score": train.score if train and not train.failed else None,
        "tier1_score": review.tier1_score if review else None,
        "failure_summary": _failure_lines(node),
    }


def _compliance_record(review: ReviewResult | None) -> dict[str, Any]:
    if review is None:
        return {}
    return {
        "coverage_score": review.coverage_score,
        "external_fraction": review.external_fraction,
        "synthetic_fraction": review.synthetic_fraction,
        "seed_link_rate": review.seed_link_rate,
        "normalized_dup_rate": review.normalized_dup_rate,
        "structural_dup_rate": review.structural_dup_rate,
        "source_compliance_score": review.source_compliance_score,
    }


def _quality_record(review: ReviewResult | None) -> dict[str, Any]:
    if review is None:
        return {}
    return {
        "correctness": review.correctness_score,
        "difficulty": review.difficulty_score,
        "diversity": review.diversity_score,
        "relevance": review.relevance_score,
    }


def _compliance(review: ReviewResult | None) -> float:
    if review is None:
        return 0.0
    return min(review.coverage_score, review.source_compliance_score)


def _is_credible(node: PipelineNode, threshold: float) -> bool:
    review = node.review_result
    return bool(
        review and review.passed and not node.dead
        and review.coverage_score >= threshold
        and review.source_compliance_score >= threshold
    )


def _outcome_score(node: PipelineNode) -> tuple[int, float, float]:
    train = node.train_result
    review = node.review_result
    if train and not train.failed:
        return (1, train.score, review.tier1_score if review else 0.0)
    return (0, review.tier1_score if review else 0.0, 0.0)


def _pareto_front(nodes: list[PipelineNode]) -> list[PipelineNode]:
    def vector(node: PipelineNode) -> tuple[float, float, float]:
        review = node.review_result
        return (
            _outcome_score(node)[1],
            _compliance(review),
            review.diversity_score if review else 0.0,
        )

    front: list[PipelineNode] = []
    for node in nodes:
        own = vector(node)
        dominated = any(
            all(a >= b for a, b in zip(vector(other), own))
            and any(a > b for a, b in zip(vector(other), own))
            for other in nodes if other is not node
        )
        if not dominated:
            front.append(node)
    return sorted(front, key=_outcome_score, reverse=True)


def _observed_effects(nodes: list[PipelineNode], top_k: int) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for node in nodes:
        parent = node.parent
        if (
            not parent or not parent.config or not parent.review_result
            or not _is_credible(parent, 0.9)
        ):
            continue
        child_train, parent_train = node.train_result, parent.train_result
        if not child_train or child_train.failed or not parent_train or parent_train.failed:
            continue
        effects.append({
            "recipe_delta": _recipe_delta(parent.config.recipe, node.config.recipe),
            "downstream_delta": child_train.score - parent_train.score,
            "diversity_delta": node.review_result.diversity_score - parent.review_result.diversity_score,
            "compliance_delta": _compliance(node.review_result) - _compliance(parent.review_result),
        })
    return sorted(effects, key=lambda item: abs(item["downstream_delta"]), reverse=True)[:top_k]


def _recipe_delta(parent: dict[str, Any], child: dict[str, Any]) -> str:
    if not parent:
        return "initial recipe"
    changes: list[str] = []
    if parent.get("mode") != child.get("mode"):
        changes.append(f"mode {parent.get('mode')} -> {child.get('mode')}")
    for key in ("candidate_multiplier", "mixing_policy", "proposal_strategies", "verification"):
        if parent.get(key) != child.get(key):
            changes.append(f"changed {key}")
    parent_buckets = {b.get("name"): b.get("quota") for b in parent.get("coverage_buckets", []) if isinstance(b, dict)}
    child_buckets = {b.get("name"): b.get("quota") for b in child.get("coverage_buckets", []) if isinstance(b, dict)}
    if parent_buckets != child_buckets:
        changes.append("changed coverage buckets/quotas")
    parent_sources = {s.get("source_id") for s in parent.get("sources", []) if isinstance(s, dict)}
    child_sources = {s.get("source_id") for s in child.get("sources", []) if isinstance(s, dict)}
    if parent_sources != child_sources:
        changes.append("changed sources")
    return "; ".join(changes) if changes else "no material recipe change"


def _diagnostic_lines(node: PipelineNode) -> list[str]:
    diagnostic = node.diagnostic_report
    if diagnostic is None:
        return []
    lines = [diagnostic.summary] if diagnostic.summary else []
    lines.extend(diagnostic.failure_patterns[:3])
    lines.extend(diagnostic.actionable_suggestions[:3])
    return lines


def _failure_lines(node: PipelineNode) -> list[str]:
    review = node.review_result
    lines = list(review.issues[:3]) if review else []
    lines.extend(_diagnostic_lines(node)[:3])
    if node.dead:
        lines.insert(0, "node marked dead")
    return lines[:6]


def _failure_record(node: PipelineNode) -> dict[str, Any]:
    outcome = _compact_outcome(node)
    return {
        "recipe_hash": outcome["recipe_hash"],
        "mode": outcome["mode"],
        "recipe_delta": outcome["recipe_delta"],
        "compliance_score": outcome["compliance_score"],
        "failure_summary": outcome["failure_summary"],
    }
