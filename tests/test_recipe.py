import json

from dataflow_evolver.agents.pipeline_agent import PipelineAgent
from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.recipe import validate_recipe
from dataflow_evolver.core.schemas import TaskProfile
from dataflow_evolver.llm.agent_sdk import AgentRunResult
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
    assert any("two distinct" in error for error in result.errors)


def test_mode_is_bounded_by_available_inputs_and_tools():
    recipe = _synthetic()
    recipe["mode"] = "external"
    result = validate_recipe(recipe, grounded=False, external_data=False)
    assert not result.valid
    assert any("unavailable" in error for error in result.errors)


def test_grounded_recipe_cannot_discard_corpus_for_pure_synthesis():
    result = validate_recipe(_synthetic(), grounded=True, external_data=True)
    assert not result.valid
    assert any("unavailable" in error for error in result.errors)


def test_pipeline_agent_validates_recipe_before_generating_code(tmp_path, monkeypatch):
    phases = []
    recipe = _synthetic()

    def fake_run(prompt, cwd, agent_cfg, tool_log_dir, phase):
        phases.append(phase)
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
    assert config is not None and config.recipe["mode"] == "synthetic"
    assert len(config.recipe_hash) == 64
