import json

import dataflow_evolver.llm.agent_sdk as agent_sdk
from dataflow_evolver.agents.pipeline_agent import PipelineAgent
from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.recipe import validate_recipe
from dataflow_evolver.core.schemas import TaskProfile
from dataflow_evolver.llm.agent_sdk import AgentRunResult, RECIPE_SYSTEM_PROMPT_APPEND
from dataflow_evolver.llm.prompts import build_recipe_prompt, build_recipe_repair_prompt
from dataflow_evolver.utils.config import Config


def _synthetic():
    return {
        "version": 1,
        "mode": "synthetic",
        "target_samples": 4,
        "candidate_multiplier": 2.0,
        "coverage_buckets": [
            {"name": "add", "operations": ["addition"], "quota": 2},
            {"name": "rate", "skills": ["rate"], "quota": 2},
        ],
        "proposal_strategies": {"program_first": 0.5, "constraint_composition": 0.5},
        "verification": {"methods": ["recompute"]},
        "selection": {"normalized_dedup": True, "coverage_aware": True},
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


def test_grounded_recipe_cannot_discard_corpus_for_pure_synthesis():
    result = validate_recipe(_synthetic(), grounded=True, external_data=True)
    assert not result.valid
    assert any("不允许" in error for error in result.errors)


def test_pipeline_agent_validates_recipe_before_generating_code(tmp_path, monkeypatch):
    phases = []
    recipe = _synthetic()

    system_prompts = []

    def fake_run(prompt, cwd, agent_cfg, tool_log_dir, phase, system_prompt_append=None):
        phases.append(phase)
        system_prompts.append(system_prompt_append)
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

    monkeypatch.setattr("dataflow_evolver.agents.pipeline_agent.run_pipeline_agent", fake_run)
    agent = PipelineAgent(
        Config({"recipe_max_revisions": 1}),
        str(tmp_path),
        InputCorpus(entry_path="empty.jsonl"),
        target_samples=4,
    )
    profile = TaskProfile("math", {"instruction": "str", "output": "str"}, [], "sft")
    config = agent.generate_initial(profile, "node_0")

    assert phases == ["plan_recipe", "generate_init"]
    assert system_prompts == [RECIPE_SYSTEM_PROMPT_APPEND, None]
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


def test_recipe_resume_nudge_does_not_switch_to_pipeline(tmp_path, monkeypatch):
    prompts = []
    option_overrides = []
    results = [
        AgentRunResult(False, error="timeout", error_type="Timeout", session_id="session-1"),
        AgentRunResult(True),
    ]

    def fake_options(cfg, cwd, resume=None, system_prompt_append=None):
        option_overrides.append((resume, system_prompt_append))
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
    )
    assert result.success
    assert prompts[0] == "首次 recipe prompt"
    assert "只创建或修订" in prompts[1]
    assert "完成原任务：write a working pipeline.py" not in prompts[1]
    assert option_overrides == [
        (None, RECIPE_SYSTEM_PROMPT_APPEND),
        ("session-1", RECIPE_SYSTEM_PROMPT_APPEND),
    ]
