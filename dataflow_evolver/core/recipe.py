"""Deterministic contract for planning data generation before writing a pipeline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RECIPE_VERSION = 1
VALID_MODES = {"external", "hybrid", "synthetic"}


@dataclass
class RecipeValidationResult:
    valid: bool
    recipe: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recipe_hash: str = ""


def allowed_modes(*, grounded: bool, external_data: bool) -> set[str]:
    if grounded:
        return {"external", "hybrid"}
    if external_data:
        return set(VALID_MODES)
    return {"synthetic"}


def validate_recipe(
    raw: Any,
    *,
    grounded: bool,
    external_data: bool,
    default_target_samples: int = 0,
) -> RecipeValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return RecipeValidationResult(False, errors=["recipe must be a JSON object"])

    recipe = dict(raw)
    version = recipe.get("version")
    if version != RECIPE_VERSION:
        errors.append(f"version must be {RECIPE_VERSION}")

    mode = str(recipe.get("mode", "")).strip().lower()
    modes = allowed_modes(grounded=grounded, external_data=external_data)
    if mode not in VALID_MODES:
        errors.append("mode must be one of: external, hybrid, synthetic")
    elif mode not in modes:
        errors.append(f"mode {mode!r} is unavailable; allowed modes: {sorted(modes)}")
    recipe["mode"] = mode

    target = recipe.get("target_samples", default_target_samples)
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        errors.append("target_samples must be a positive integer")
    recipe["target_samples"] = target

    multiplier = recipe.get("candidate_multiplier", 1.0)
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool) or multiplier < 1.0:
        errors.append("candidate_multiplier must be a number >= 1.0")
    elif mode in {"hybrid", "synthetic"} and multiplier <= 1.0:
        errors.append("hybrid/synthetic recipes must over-generate: candidate_multiplier > 1.0")

    buckets = recipe.get("coverage_buckets", [])
    if not isinstance(buckets, list):
        errors.append("coverage_buckets must be an array")
        buckets = []
    names: set[str] = set()
    quota_sum = 0
    for i, bucket in enumerate(buckets):
        if not isinstance(bucket, dict):
            errors.append(f"coverage_buckets[{i}] must be an object")
            continue
        name = str(bucket.get("name", "")).strip()
        quota = bucket.get("quota")
        if not name:
            errors.append(f"coverage_buckets[{i}].name is required")
        elif name in names:
            errors.append(f"duplicate coverage bucket name: {name}")
        names.add(name)
        if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
            errors.append(f"coverage_buckets[{i}].quota must be a positive integer")
        else:
            quota_sum += quota
        if not any(bucket.get(k) for k in ("skills", "operations", "contexts", "difficulty", "answer_type")):
            errors.append(f"coverage_buckets[{i}] must define at least one semantic dimension")

    strategies = recipe.get("proposal_strategies", {})
    if not isinstance(strategies, dict):
        errors.append("proposal_strategies must be an object of strategy -> positive weight")
        strategies = {}
    positive_strategies = [k for k, v in strategies.items() if isinstance(v, (int, float)) and v > 0]
    verification = recipe.get("verification", {})
    methods = verification.get("methods", []) if isinstance(verification, dict) else []
    selection = recipe.get("selection", {})

    if mode in {"hybrid", "synthetic"}:
        if not buckets:
            errors.append("hybrid/synthetic recipes require non-empty coverage_buckets")
        if isinstance(target, int) and target > 0 and quota_sum < target:
            errors.append("sum of coverage bucket quotas must cover target_samples")
        if not positive_strategies:
            errors.append("hybrid/synthetic recipes require at least one positive proposal strategy")
        if not isinstance(methods, list) or not methods:
            errors.append("hybrid/synthetic recipes require verification.methods")
        if not isinstance(selection, dict) or selection.get("normalized_dedup") is not True:
            errors.append("hybrid/synthetic recipes require selection.normalized_dedup=true")
        if not isinstance(selection, dict) or selection.get("coverage_aware") is not True:
            errors.append("hybrid/synthetic recipes require selection.coverage_aware=true")

    if mode == "synthetic":
        forbidden = {"variant", "variant_number", "index_only", "fixed_prompt_variants"}
        if forbidden.intersection(str(k).lower() for k in strategies):
            errors.append("synthetic recipes may not use fixed-prompt/index variants as a strategy")
        if len(positive_strategies) < 2:
            errors.append("synthetic recipes require at least two distinct proposal strategies")

    canonical = _canonicalize(recipe)
    digest = hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()
    return RecipeValidationResult(not errors, canonical, errors, warnings, digest)


def load_and_validate_recipe(path: str | Path, **kwargs: Any) -> RecipeValidationResult:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return RecipeValidationResult(False, errors=[f"recipe file not found: {path}"])
    except (OSError, json.JSONDecodeError) as exc:
        return RecipeValidationResult(False, errors=[f"invalid recipe JSON: {exc}"])
    return validate_recipe(raw, **kwargs)


def write_canonical_recipe(path: str | Path, result: RecipeValidationResult) -> None:
    if not result.valid:
        raise ValueError("cannot write an invalid recipe")
    Path(path).write_text(_canonical_json(result.recipe) + "\n", encoding="utf-8")


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
