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
        return RecipeValidationResult(False, errors=["recipe 必须是 JSON object"])

    recipe = dict(raw)
    version = recipe.get("version")
    if version != RECIPE_VERSION:
        errors.append(f"version 必须为 {RECIPE_VERSION}")

    mode = str(recipe.get("mode", "")).strip().lower()
    modes = allowed_modes(grounded=grounded, external_data=external_data)
    if mode not in VALID_MODES:
        errors.append("mode 必须是 external、hybrid、synthetic 之一")
    elif mode not in modes:
        errors.append(f"当前输入与工具条件不允许 mode={mode!r}；允许值：{sorted(modes)}")
    recipe["mode"] = mode

    target = recipe.get("target_samples", default_target_samples)
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        errors.append("target_samples 必须是正整数")
    recipe["target_samples"] = target

    multiplier = recipe.get("candidate_multiplier", 1.0)
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool) or multiplier < 1.0:
        errors.append("candidate_multiplier 必须是大于等于 1.0 的数值")
    elif mode in {"hybrid", "synthetic"} and multiplier <= 1.0:
        errors.append("hybrid/synthetic 必须过量生成：candidate_multiplier > 1.0")

    buckets = recipe.get("coverage_buckets", [])
    if not isinstance(buckets, list):
        errors.append("coverage_buckets 必须是 array")
        buckets = []
    names: set[str] = set()
    quota_sum = 0
    for i, bucket in enumerate(buckets):
        if not isinstance(bucket, dict):
            errors.append(f"coverage_buckets[{i}] 必须是 object")
            continue
        name = str(bucket.get("name", "")).strip()
        quota = bucket.get("quota")
        if not name:
            errors.append(f"coverage_buckets[{i}].name 不能为空")
        elif name in names:
            errors.append(f"coverage bucket 名称重复：{name}")
        names.add(name)
        if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
            errors.append(f"coverage_buckets[{i}].quota 必须是正整数")
        else:
            quota_sum += quota
        if not any(bucket.get(k) for k in ("skills", "operations", "contexts", "difficulty", "answer_type")):
            errors.append(f"coverage_buckets[{i}] 至少要定义一个语义维度")

    strategies = recipe.get("proposal_strategies", {})
    if not isinstance(strategies, dict):
        errors.append("proposal_strategies 必须是 strategy -> 正权重的 object")
        strategies = {}
    positive_strategies = [k for k, v in strategies.items() if isinstance(v, (int, float)) and v > 0]
    verification = recipe.get("verification", {})
    methods = verification.get("methods", []) if isinstance(verification, dict) else []
    selection = recipe.get("selection", {})

    if mode in {"hybrid", "synthetic"}:
        if not buckets:
            errors.append("hybrid/synthetic 必须提供非空 coverage_buckets")
        if isinstance(target, int) and target > 0 and quota_sum < target:
            errors.append("coverage bucket 的 quota 总和必须覆盖 target_samples")
        if not positive_strategies:
            errors.append("hybrid/synthetic 至少需要一种正权重 proposal strategy")
        if not isinstance(methods, list) or not methods:
            errors.append("hybrid/synthetic 必须提供 verification.methods")
        if not isinstance(selection, dict) or selection.get("normalized_dedup") is not True:
            errors.append("hybrid/synthetic 要求 selection.normalized_dedup=true")
        if not isinstance(selection, dict) or selection.get("coverage_aware") is not True:
            errors.append("hybrid/synthetic 要求 selection.coverage_aware=true")

    if mode == "synthetic":
        forbidden = {"variant", "variant_number", "index_only", "fixed_prompt_variants"}
        if forbidden.intersection(str(k).lower() for k in strategies):
            errors.append("synthetic 禁止把 fixed-prompt/index variant 作为生成策略")
        if len(positive_strategies) < 2:
            errors.append("synthetic 至少需要两种实质不同的 proposal strategy")

    canonical = _canonicalize(recipe)
    digest = hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()
    return RecipeValidationResult(not errors, canonical, errors, warnings, digest)


def load_and_validate_recipe(path: str | Path, **kwargs: Any) -> RecipeValidationResult:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return RecipeValidationResult(False, errors=[f"找不到 recipe 文件：{path}"])
    except (OSError, json.JSONDecodeError) as exc:
        return RecipeValidationResult(False, errors=[f"recipe JSON 无效：{exc}"])
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
