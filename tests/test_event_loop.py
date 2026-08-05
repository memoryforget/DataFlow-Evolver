"""主事件环集成冒烟测试：用桩替换 Agent SDK 与 LLM，但跑真实的 select/expand/执行/训练/回传。

不依赖 ANTHROPIC_API_KEY、不依赖已安装的 DataFlow——验证编排逻辑本身正确接线。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.core.schemas import (
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TaskProfile,
)
from dataflow_evolver.main import Orchestrator
from dataflow_evolver.utils.config import Config

FAKE_PIPELINE = (
    "import json, pathlib\n"
    "out = pathlib.Path(__file__).parent / 'gen.jsonl'\n"
    "rows = [{'instruction': f'q{i}', 'answer': f'a{i}'} for i in range(5)]\n"
    "out.write_text('\\n'.join(json.dumps(r) for r in rows) + '\\n', encoding='utf-8')\n"
)


def _make_config(tmp_path: Path) -> Config:
    agent_root = tmp_path / "_agent_root"
    agent_root.mkdir(parents=True)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "seed.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
    ref_script = Path(__file__).resolve().parents[1] / "scripts" / "reference_train_eval.py"
    # dev 题集来源：一半题能被 FAKE_PIPELINE 产出的 q0..q4 答上，一半答不上 → score=0.5
    dev_source = tmp_path / "dev_source.jsonl"
    dev_source.write_text(
        "".join(
            json.dumps({"instruction": q, "answer": q.replace("q", "a")}) + "\n"
            for q in ("q0", "q1", "q7", "q8")
        ),
        encoding="utf-8",
    )
    raw = {
        "project": {
            "workspace_dir": str(tmp_path / "workspace"),
            "agent_root": str(agent_root),
            "input_dir": str(input_dir),
        },
        "llm": {"backend": "api", "api": {
            "api_url": "https://example.invalid/v1/chat/completions",
            "model_name": "stub", "api_key": "stub-key", "max_workers": 2,
        }},
        "agent_sdk": {"pipeline_filename": "pipeline.py", "permission_mode": "bypassPermissions",
                      "allowed_tools": ["Read", "Write"], "setting_sources": ["project"]},
        "search": {"exploration_constant": 1.414, "max_iterations": 3,
                   "max_children_per_node": 2, "self_correction_retries": 1},
        "review": {"schema_threshold": 0.6, "relevance_threshold": 0.6, "sample_size": 5},
        "trainer": {
            "dev_eval_cmd": f'{sys.executable} "{ref_script}"',
            "timeout_sec": 60,
            "train_timeout_sec": 60,
            # 框架物化 dev 题集并经 --dev-set 下传；参考脚本拿不到会直接非零退出
            "dev_set": {"source": str(dev_source), "size": 0, "seed": 42},
        },
        "diagnostic": {"max_bad_cases": 10},
    }
    return Config(raw)


class _StubServing:
    def generate(self, prompts, **kwargs):
        return ["{}" for _ in prompts]

    def generate_one(self, prompt, **kwargs):
        return "{}"


def test_event_loop_runs_and_picks_best(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    orch = Orchestrator(cfg)

    # 桩：TaskProfiler 直接给出固定 profile
    profile = TaskProfile("生成问答", {"instruction": "str", "answer": "str"}, ["答案非空"], "qa-sft")
    monkeypatch.setattr(orch.profiler, "run", lambda *a, **k: profile)

    # 桩：ReviewAgent 总是通过
    monkeypatch.setattr(
        orch.reviewer, "review",
        lambda dataset_path, prof, **kwargs: ReviewResult(0.9, 0.9, True, []),
    )
    # 桩：DiagnosticAgent 返回空报告
    monkeypatch.setattr(
        orch.diagnostician, "analyze",
        lambda bad, downstream="": DiagnosticReport(summary="ok"),
    )

    # 桩：PipelineAgent 写出一个能跑的 pipeline.py 并返回 config（Init node 与变异节点同一桩）
    def fake_generate(profile, node_dir_name, **kwargs):
        node_dir = orch.agent_root / node_dir_name
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "pipeline.py").write_text(FAKE_PIPELINE, encoding="utf-8")
        return PipelineConfig(
            operators=[{"name": "PromptedGenerator", "params": {}}],
            field_flow="text -> qa",
            code=FAKE_PIPELINE,
            rationale="stub",
        )

    monkeypatch.setattr(orch.pipeline_agent, "generate", fake_generate)
    monkeypatch.setattr(orch.pipeline_agent, "generate_initial", fake_generate)
    monkeypatch.setattr(orch.recipe_agent, "plan", lambda *a, **k: ({}, "stub-recipe"))

    best = orch.run("生成问答数据集")
    assert best is not None
    assert best.train_result is not None
    assert best.train_result.failed is False
    # 参考脚本在框架物化的 dev 题集上评测：4 题里 q0/q1 被 FAKE_PIPELINE 产出的数据覆盖，
    # q7/q8 没有 → exact_match=0.5。分数来自题集而非训练数据自身的某个属性。
    assert best.train_result.metric_name == "exact_match"
    assert abs(best.train_result.score - 0.5) < 1e-9
    assert {bc.input for bc in best.train_result.bad_cases} == {"q7", "q8"}
    # dev 题集确实被物化并下传（而不是脚本自己挑的题）
    assert orch.dev_set is not None
    assert Path(orch.dev_set.path).exists()
    # 跑了 3 次迭代，应有若干已评估节点
    assert len(orch.tree.all_nodes()) >= 1


def test_resolve_pipeline_llm_fallback_and_override():
    """pipeline_llm 段按用途区分多 serving，逐字段回退 api。"""
    from dataflow_evolver.main import _resolve_pipeline_llm

    # 缺省 pipeline_llm → 返回 {"default": ...}，全部回退 api
    llm = Config({"api": {"api_url": "u", "model_name": "m", "api_key": "k", "max_workers": 7}})
    assert _resolve_pipeline_llm(llm) == {
        "default": {"api_url": "u", "model_name": "m", "api_key": "k", "max_workers": 7},
    }

    # 多用途格式：generation 只写 model_name → 覆盖模型，其余继承 api
    llm2 = Config({
        "api": {"api_url": "u", "model_name": "strong", "api_key": "k", "max_workers": 7},
        "pipeline_llm": {"generation": {"model_name": "cheap"}, "reasoning": {"model_name": "deep"}},
    })
    r2 = _resolve_pipeline_llm(llm2)
    assert r2["generation"]["model_name"] == "cheap"
    assert r2["generation"]["api_url"] == "u" and r2["generation"]["api_key"] == "k"
    assert r2["reasoning"]["model_name"] == "deep"
    assert r2["reasoning"]["max_workers"] == 7
