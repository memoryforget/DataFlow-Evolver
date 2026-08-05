import json
import sys
import types

import pytest

import dataflow_evolver.llm.agent_sdk as agent_sdk
from dataflow_evolver.agents.pipeline_agent import PipelineAgent
from dataflow_evolver.agents.recipe_agent import RecipeAgent
from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.recipe import validate_recipe
from dataflow_evolver.core.recipe_context import RecipeContext, build_recipe_context
from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult, TaskProfile, TrainResult
from dataflow_evolver.core.tree_node import PipelineNode
from dataflow_evolver.llm.agent_sdk import AgentRunResult, RECIPE_SYSTEM_PROMPT_APPEND
from dataflow_evolver.llm.prompts import build_recipe_prompt, build_recipe_repair_prompt
from dataflow_evolver.utils.config import Config


def _synthetic():
    return {
        "version": 1,
        "mode": "synthetic",
        "target_samples": 4,
        "candidate_multiplier": 2.0,
        "sources": [],
        "coverage_buckets": [
            {"name": "add", "operations": ["addition"], "quota": 2},
            {"name": "rate", "skills": ["rate"], "quota": 2},
        ],
        "proposal_strategies": {"program_first": 0.5, "constraint_composition": 0.5},
        "verification": {"methods": ["recompute"]},
        "selection": {"normalized_dedup": True, "coverage_aware": True},
    }


def _external_source(tmp_path, name="math", split="train"):
    links = tmp_path / "data_links"
    links.mkdir(exist_ok=True)
    data = links / f"{name}.jsonl"
    data.write_text(json.dumps({"question": "1+1", "answer": "2"}) + "\n", encoding="utf-8")
    data.with_suffix(".meta.json").write_text(
        json.dumps({
            "source": "huggingface",
            "dataset_id": f"example/{name}",
            "split": split,
            "license": "MIT",
            "rows": 1,
            "complete": True,
        }),
        encoding="utf-8",
    )
    return links, {
        "source_id": name,
        "kind": "external_dataset",
        "location": str(data),
        "license": "MIT",
        "split": split,
    }


def test_valid_synthetic_recipe_is_canonical_and_hashed():
    a = validate_recipe(_synthetic(), grounded=False, external_data=False)
    b = validate_recipe(json.loads(json.dumps(_synthetic())), grounded=False, external_data=False)
    assert a.valid
    assert a.recipe_hash == b.recipe_hash


def test_synthetic_rejects_index_variants_and_single_strategy():
    recipe = _synthetic()
    recipe["proposal_strategies"] = {"variant_number": 1.0}
    result = validate_recipe(recipe, grounded=False, external_data=False)
    assert not result.valid
    assert any("fixed-prompt" in error for error in result.errors)
    assert any("两种实质不同" in error for error in result.errors)


def test_mode_is_bounded_by_available_inputs_and_tools():
    recipe = _synthetic()
    recipe["mode"] = "external"
    result = validate_recipe(recipe, grounded=False, external_data=False)
    assert not result.valid
    assert any("不允许" in error for error in result.errors)


def test_required_mode_is_enforced_by_validator_and_prompt():
    recipe = _synthetic()
    result = validate_recipe(
        recipe, grounded=False, external_data=True, required_mode="hybrid"
    )
    assert not result.valid
    assert any("强制 mode='hybrid'" in error for error in result.errors)

    profile = TaskProfile("生成数学题", {"instruction": "str", "output": "str"}, [], "sft")
    prompt = build_recipe_prompt(
        profile,
        InputCorpus(entry_path="empty.jsonl"),
        "node_0",
        8,
        external_data=True,
        required_mode="hybrid",
    )
    assert "允许的 mode：['hybrid']" in prompt
    assert "实验强制 mode：hybrid" in prompt


def test_recipe_agent_rejects_required_mode_without_required_capability(tmp_path):
    with pytest.raises(ValueError, match="不兼容"):
        RecipeAgent(
            Config({"recipe_mode": "hybrid"}),
            str(tmp_path),
            InputCorpus(entry_path="empty.jsonl"),
            4,
            False,
        )


def test_grounded_recipe_cannot_discard_corpus_for_pure_synthesis():
    result = validate_recipe(_synthetic(), grounded=True, external_data=True)
    assert not result.valid
    assert any("不允许" in error for error in result.errors)


def test_pipeline_agent_validates_recipe_before_generating_code(tmp_path, monkeypatch):
    phases = []
    recipe = _synthetic()

    system_prompts = []

    tool_profiles = []

    class FakeRuntime:
        def run(self, prompt, **kwargs):
            phase = kwargs["phase"]
            phases.append(phase)
            system_prompts.append(kwargs.get("system_prompt_append"))
            tool_profiles.append(kwargs["tool_profile"])
            node = tmp_path / "node_0"
            node.mkdir(exist_ok=True)
            if phase == "plan_recipe":
                (node / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
            else:
                (node / "pipeline.py").write_text("print('ok')", encoding="utf-8")
                (node / "decision.json").write_text(
                    json.dumps({"ops": [], "field_flow": "Empty -> output", "reason": "test"}),
                    encoding="utf-8",
                )
            return AgentRunResult(success=True)

    runtime = FakeRuntime()
    corpus = InputCorpus(entry_path="empty.jsonl")
    cfg = Config({"recipe_max_revisions": 1})
    recipe_agent = RecipeAgent(cfg, str(tmp_path), corpus, 4, False, runtime=runtime)
    agent = PipelineAgent(
        cfg,
        str(tmp_path),
        corpus,
        target_samples=4,
        runtime=runtime,
    )
    profile = TaskProfile("math", {"instruction": "str", "output": "str"}, [], "sft")
    planned = recipe_agent.plan(profile, "node_0", RecipeContext())
    assert planned is not None
    validated, recipe_hash = planned
    config = agent.generate_initial(
        profile, "node_0", validated_recipe=validated, recipe_hash=recipe_hash
    )

    assert phases == ["plan_recipe", "generate_init"]
    assert system_prompts == [RECIPE_SYSTEM_PROMPT_APPEND, None]
    assert tool_profiles == ["recipe", "pipeline"]
    assert config is not None and config.recipe["mode"] == "synthetic"
    assert len(config.recipe_hash) == 64


def test_recipe_prompts_use_consistent_chinese_control_language():
    profile = TaskProfile("生成数学题", {"instruction": "str", "output": "str"}, [], "sft")
    prompt = build_recipe_prompt(
        profile, InputCorpus(entry_path="empty.jsonl"), "node_1", 100, external_data=False
    )
    repair = build_recipe_repair_prompt("node_1", ["target_samples 必须是正整数"])
    assert "本阶段只能写入" in prompt and "确定性校验器" in prompt
    assert "Plan the data-generation strategy" not in prompt
    assert "确定性校验器拒绝" in repair and "只修改" in repair


def test_recipe_prompt_receives_structured_search_evidence():
    profile = TaskProfile("生成数学题", {"instruction": "str", "output": "str"}, [], "sft")
    context = RecipeContext(
        parent={"recipe": _synthetic(), "compliance": {"coverage_score": 0.72}},
        siblings=[{"recipe_hash": "sibling", "downstream_score": 0.51}],
        global_memory={
            "top_recipes": [{"recipe_hash": "top"}], "pareto_recipes": [],
            "failed_patterns": [{"recipe_hash": "bad"}], "observed_effects": [],
        },
    )
    prompt = build_recipe_prompt(
        profile, InputCorpus(entry_path="empty.jsonl"), "node_2", 100, False, context=context
    )
    assert "RecipeContext" in prompt
    assert '"coverage_score": 0.72' in prompt
    assert '"recipe_hash": "sibling"' in prompt
    assert '"recipe_hash": "top"' in prompt and '"recipe_hash": "bad"' in prompt


def test_recipe_context_separates_parent_sibling_and_global_memory():
    root = PipelineNode(PipelineConfig([], "f", "x", "r", _synthetic(), "root"))
    root.id = 0
    root.review_result = ReviewResult(
        schema_score=1, relevance_score=.9, passed=True, correctness_score=.85,
        difficulty_score=.61, diversity_score=.55, coverage_score=.72,
        source_compliance_score=1.0, normalized_dup_rate=.18, structural_dup_rate=.44,
        external_fraction=.31, synthetic_fraction=.69, seed_link_rate=1.0,
    )
    root.train_result = TrainResult(score=.47, metric_name="gsm8k")
    sibling = PipelineNode(PipelineConfig([], "f", "x", "r", _synthetic(), "sib"), root)
    sibling.id = 1
    sibling.review_result = ReviewResult(1, 1, True, coverage_score=.95, source_compliance_score=.96)
    sibling.train_result = TrainResult(score=.51, metric_name="gsm8k")
    historical = PipelineNode(PipelineConfig([], "f", "x", "r", _synthetic(), "hist"))
    historical.id = 9
    historical.review_result = ReviewResult(
        1, 1, True, diversity_score=.8, coverage_score=.97, source_compliance_score=.98,
        tier1_score=.7,
    )
    historical.train_result = TrainResult(score=.55, metric_name="gsm8k")
    context = build_recipe_context(root, [sibling], [root, sibling, historical])
    assert context.parent["downstream"]["score"] == .47
    assert context.parent["compliance"]["structural_dup_rate"] == .44
    assert context.siblings[0]["recipe_hash"] == "sib"
    assert context.global_memory["top_recipes"][0]["recipe_hash"] == "hist"


def test_agent_sdk_mcp_profiles_are_phase_isolated(monkeypatch, tmp_path):
    class FakeOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", types.SimpleNamespace(ClaudeAgentOptions=FakeOptions))
    monkeypatch.setitem(
        sys.modules, "dataflow_evolver.llm.mcp_operators",
        types.SimpleNamespace(operator_mcp_server="operators"),
    )
    monkeypatch.setitem(
        sys.modules, "dataflow_evolver.llm.mcp_datasets",
        types.SimpleNamespace(dataset_mcp_server="datasets"),
    )
    monkeypatch.setitem(
        sys.modules, "dataflow_evolver.llm.mcp_discovery",
        types.SimpleNamespace(discovery_mcp_server="discovery"),
    )
    cfg = Config({"external_data": True, "discovery": True})
    recipe_opts = agent_sdk._build_options(cfg, str(tmp_path), tool_profile="recipe")
    pipeline_opts = agent_sdk._build_options(cfg, str(tmp_path), tool_profile="pipeline")
    assert set(recipe_opts.mcp_servers) == {"dataset-search", "data-discovery"}
    assert set(pipeline_opts.mcp_servers) == {"dataflow-operators"}


def test_recipe_agent_rejects_implementation_artifacts(tmp_path):
    class LeakyRuntime:
        def run(self, prompt, **kwargs):
            node = tmp_path / "node_0"
            node.mkdir(exist_ok=True)
            (node / "recipe.json").write_text(json.dumps(_synthetic()), encoding="utf-8")
            (node / "pipeline.py").write_text("print('too early')", encoding="utf-8")
            return AgentRunResult(success=True)

    agent = RecipeAgent(
        Config({"recipe_max_revisions": 0}), str(tmp_path),
        InputCorpus(entry_path="empty.jsonl"), 4, False, runtime=LeakyRuntime(),
    )
    profile = TaskProfile("math", {"instruction": "str", "output": "str"}, [], "sft")
    assert agent.plan(profile, "node_0", RecipeContext()) is None


def test_pipeline_agent_cannot_modify_frozen_recipe(tmp_path):
    node = tmp_path / "node_0"
    node.mkdir()
    recipe_path = node / "recipe.json"
    recipe_path.write_text(json.dumps(_synthetic()), encoding="utf-8")

    class MutatingRuntime:
        def run(self, prompt, **kwargs):
            mutated = _synthetic()
            mutated["candidate_multiplier"] = 3.0
            recipe_path.write_text(json.dumps(mutated), encoding="utf-8")
            (node / "pipeline.py").write_text("print('ok')", encoding="utf-8")
            (node / "decision.json").write_text("{}", encoding="utf-8")
            return AgentRunResult(success=True)

    agent = PipelineAgent(
        Config({}), str(tmp_path), InputCorpus(entry_path="empty.jsonl"), runtime=MutatingRuntime()
    )
    profile = TaskProfile("math", {"instruction": "str", "output": "str"}, [], "sft")
    assert agent.generate_initial(
        profile, "node_0", validated_recipe=_synthetic(), recipe_hash="frozen"
    ) is None


def test_recipe_resume_nudge_does_not_switch_to_pipeline(tmp_path, monkeypatch):
    prompts = []
    option_overrides = []
    results = [
        AgentRunResult(False, error="timeout", error_type="Timeout", session_id="session-1"),
        AgentRunResult(True),
    ]

    def fake_options(cfg, cwd, resume=None, system_prompt_append=None, tool_profile="pipeline"):
        option_overrides.append((resume, system_prompt_append, tool_profile))
        return object()

    async def fake_run(prompt, options, timeout):
        prompts.append(prompt)
        return results.pop(0)

    monkeypatch.setattr(agent_sdk, "_build_options", fake_options)
    monkeypatch.setattr(agent_sdk, "_run_with_timeout", fake_run)
    result = agent_sdk.run_pipeline_agent(
        "首次 recipe prompt",
        str(tmp_path),
        Config({"max_retries": 2, "retry_base_delay": 0, "resume_on_retry": True}),
        phase="plan_recipe",
        system_prompt_append=RECIPE_SYSTEM_PROMPT_APPEND,
        tool_profile="recipe",
    )
    assert result.success
    assert prompts[0] == "首次 recipe prompt"
    assert "只创建或修订" in prompts[1]
    assert "完成原任务：write a working pipeline.py" not in prompts[1]
    assert option_overrides == [
        (None, RECIPE_SYSTEM_PROMPT_APPEND, "recipe"),
        ("session-1", RECIPE_SYSTEM_PROMPT_APPEND, "recipe"),
    ]


def test_external_requires_auditable_real_source(tmp_path):
    recipe = _synthetic()
    recipe.update({"mode": "external", "candidate_multiplier": 1.0})
    recipe["proposal_strategies"] = {}
    recipe["coverage_buckets"] = []
    recipe["selection"] = {"normalized_dedup": True}
    missing = validate_recipe(recipe, grounded=False, external_data=True)
    assert not missing.valid and any("至少一个真实数据 source" in e for e in missing.errors)

    links, source = _external_source(tmp_path, "math-train")
    recipe["sources"] = [source]
    assert validate_recipe(
        recipe, grounded=False, external_data=True, data_links_dir=links
    ).valid


def test_hybrid_requires_mixing_policy_and_forbids_eval_split(tmp_path):
    recipe = _synthetic()
    recipe["mode"] = "hybrid"
    links, source = _external_source(tmp_path, "seed", split="train")
    source["split"] = "test"
    recipe["sources"] = [source]
    result = validate_recipe(
        recipe, grounded=False, external_data=True, data_links_dir=links
    )
    assert not result.valid
    assert any("split='test'" in e for e in result.errors)
    assert any("mixing_policy" in e for e in result.errors)

    recipe["sources"][0]["split"] = "train"
    recipe["mixing_policy"] = {
        "external_min_fraction": 0.25, "synthetic_min_fraction": 0.5,
        "require_seed_link": True, "preserve_external_rows": True,
    }
    assert validate_recipe(
        recipe, grounded=False, external_data=True, data_links_dir=links
    ).valid


def test_external_source_must_exist_inside_data_links_with_valid_sidecar(tmp_path):
    recipe = _synthetic()
    recipe.update({"mode": "external", "candidate_multiplier": 1.0})
    recipe["proposal_strategies"] = {}
    recipe["coverage_buckets"] = []
    recipe["selection"] = {"normalized_dedup": True}

    links = tmp_path / "data_links"
    links.mkdir()
    recipe["sources"] = [{
        "source_id": "missing", "kind": "external_dataset",
        "location": str(links / "missing.jsonl"), "license": "MIT", "split": "train",
    }]
    missing = validate_recipe(
        recipe, grounded=False, external_data=True, data_links_dir=links
    )
    assert not missing.valid and any("文件不存在" in e for e in missing.errors)

    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    outside.with_suffix(".meta.json").write_text(
        json.dumps({"dataset_id": "x/y", "split": "train", "rows": 1}), encoding="utf-8"
    )
    recipe["sources"][0]["location"] = str(outside)
    escaped = validate_recipe(
        recipe, grounded=False, external_data=True, data_links_dir=links
    )
    assert not escaped.valid and any("受管 data_links" in e for e in escaped.errors)

    data = links / "no_meta.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    recipe["sources"][0]["location"] = str(data)
    no_meta = validate_recipe(
        recipe, grounded=False, external_data=True, data_links_dir=links
    )
    assert not no_meta.valid and any("缺少下载溯源边车" in e for e in no_meta.errors)


def test_multiple_external_sources_are_validated_and_canonicalized(tmp_path):
    links, source_a = _external_source(tmp_path, "a")
    _, source_b = _external_source(tmp_path, "b")
    source_a["source_id"] = "  source-a  "
    source_b["split"] = " TRAIN "
    recipe = _synthetic()
    recipe["mode"] = "hybrid"
    recipe["sources"] = [source_a, source_b]
    recipe["mixing_policy"] = {
        "external_min_fraction": 0.25, "synthetic_min_fraction": 0.5,
        "require_seed_link": True, "preserve_external_rows": True,
    }
    result = validate_recipe(
        recipe, grounded=False, external_data=True, data_links_dir=links
    )
    assert result.valid
    assert [source["source_id"] for source in result.recipe["sources"]] == ["source-a", "b"]
    assert result.recipe["sources"][1]["split"] == "train"
