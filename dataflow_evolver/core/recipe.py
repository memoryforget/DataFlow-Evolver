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

    sources = recipe.get("sources")
    if not isinstance(sources, list):
        errors.append("sources 必须是 array；synthetic 使用空数组 []")
        sources = []
    source_ids: set[str] = set()
    for i, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"sources[{i}] 必须是 object")
            continue
        source_id = str(source.get("source_id", "")).strip()
        kind = str(source.get("kind", "")).strip()
        location = str(source.get("location", "")).strip()
        license_name = str(source.get("license", "")).strip()
        split = str(source.get("split", "")).strip().lower()
        if not source_id:
            errors.append(f"sources[{i}].source_id 不能为空")
        elif source_id in source_ids:
            errors.append(f"source_id 重复：{source_id}")
        source_ids.add(source_id)
        if kind not in {"input_corpus", "external_dataset"}:
            errors.append(f"sources[{i}].kind 必须是 input_corpus 或 external_dataset")
        if not location:
            errors.append(f"sources[{i}].location 不能为空，必须记录绝对路径或输入语料入口")
        if not license_name:
            errors.append(f"sources[{i}].license 不能为空")
        if kind == "external_dataset" and not split:
            errors.append(f"sources[{i}].split 不能为空")
        if split in {"test", "validation", "eval", "evaluation"}:
            errors.append(f"sources[{i}].split={split!r} 禁止用于训练数据")

    if mode in {"external", "hybrid"} and not sources:
        errors.append(f"{mode} mode 必须声明至少一个真实数据 source")
    if mode == "synthetic" and sources:
        errors.append("synthetic mode 的 sources 必须为空数组，禁止声明真实数据源")

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

    if not isinstance(methods, list) or not methods:
        errors.append("所有 mode 都必须提供 verification.methods")
    if not isinstance(selection, dict) or selection.get("normalized_dedup") is not True:
        errors.append("所有 mode 都要求 selection.normalized_dedup=true")

    if mode == "hybrid":
        policy = recipe.get("mixing_policy")
        if not isinstance(policy, dict):
            errors.append("hybrid mode 必须提供 mixing_policy")
            policy = {}
        ext_min = policy.get("external_min_fraction")
        syn_min = policy.get("synthetic_min_fraction")
        for name, value in (("external_min_fraction", ext_min), ("synthetic_min_fraction", syn_min)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 1:
                errors.append(f"mixing_policy.{name} 必须在 (0, 1] 范围内")
        if isinstance(ext_min, (int, float)) and isinstance(syn_min, (int, float)) and ext_min + syn_min > 1:
            errors.append("hybrid mixing_policy 的两个最低占比之和不能超过 1")
        if policy.get("require_seed_link") is not True:
            errors.append("hybrid 要求 mixing_policy.require_seed_link=true")
        if policy.get("preserve_external_rows") is not True:
            errors.append("MVP hybrid 要求 mixing_policy.preserve_external_rows=true，以便验证 seed 追溯")

    if mode in {"hybrid", "synthetic"}:
        if not buckets:
            errors.append("hybrid/synthetic 必须提供非空 coverage_buckets")
        if isinstance(target, int) and target > 0 and quota_sum < target:
            errors.append("coverage bucket 的 quota 总和必须覆盖 target_samples")
        if not positive_strategies:
            errors.append("hybrid/synthetic 至少需要一种正权重 proposal strategy")
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
