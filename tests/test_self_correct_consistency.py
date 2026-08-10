import json

from dataflow_evolver.agents.review_agent import _sample_jsonl
from dataflow_evolver.llm.prompts import build_selfcorrect_prompt


def test_exec_repair_prompt_syncs_decision():
    prompt = build_selfcorrect_prompt(
        "/workspace/runs/run/iteration_004",
        "Traceback: boom",
        failure_stage="compile",
        skill_root=".agents/skills/dataflow-dev",
    )
    assert "/workspace/runs/run/iteration_004/decision.json" in prompt
    assert "ops` / `field_flow` / `reason" in prompt
    assert "失败阶段：`compile`" in prompt
    assert "不要自行执行" in prompt
    assert ".agents/skills/dataflow-dev/diagnostics/known_issues.md" in prompt
    assert ".claude/skills/dataflow-dev" not in prompt


def test_stale_decision_warns_after_execution_repair(tmp_path, monkeypatch, caplog):
    import dataflow_evolver.agents.pipeline_agent as module

    iteration = tmp_path / "iteration_001"
    iteration.mkdir()
    (iteration / "pipeline.py").write_text("print(1)", encoding="utf-8")
    decision = iteration / "decision.json"
    decision.write_text(json.dumps({"ops": [], "field_flow": "f"}), encoding="utf-8")

    agent = module.PipelineAgent.__new__(module.PipelineAgent)
    agent.workspace_dir = tmp_path
    agent.pipeline_filename = "pipeline.py"
    agent.agent_cfg = {}
    agent.runtime = None
    agent.logger = module.get_logger("test.pipeline")

    class Result:
        success = True
        retryable = False
        error = None

    class Runtime:
        def run(self, *args, **kwargs):
            return Result()

    agent.runtime = Runtime()
    with caplog.at_level("WARNING"):
        assert agent._run_and_assemble("prompt", iteration, warn_stale_decision=True)
    assert "未更新" in caplog.text


def test_fixed_review_seed_is_reproducible(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps({"q": str(i)}) for i in range(100)), encoding="utf-8"
    )
    assert _sample_jsonl(str(path), 10, seed=42) == _sample_jsonl(str(path), 10, seed=42)


def test_unset_review_seed_can_vary(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps({"q": str(i)}) for i in range(500)), encoding="utf-8"
    )
    assert len({_sample_jsonl(str(path), 20, seed=None) for _ in range(6)}) > 1
