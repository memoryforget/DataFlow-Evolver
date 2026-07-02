"""主事件环集成冒烟测试：用桩替换 Agent SDK 与 LLM，但跑真实的 select/expand/执行/训练/回传。

不依赖 ANTHROPIC_API_KEY、不依赖已安装的 DataFlow——验证编排逻辑本身正确接线。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolve.core.schemas import (
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TaskProfile,
)
from dataflow_evolve.main import Orchestrator
from dataflow_evolve.utils.config import Config

FAKE_PIPELINE = (
    "import json, pathlib\n"
    "out = pathlib.Path(__file__).parent / 'gen.jsonl'\n"
    "rows = [{'instruction': f'q{i}', 'answer': f'a{i}'} for i in range(5)]\n"
    "out.write_text('\\n'.join(json.dumps(r) for r in rows) + '\\n', encoding='utf-8')\n"
)


def _make_config(tmp_path: Path) -> Config:
    agent_root = tmp_path / "_agent_root"
    agent_root.mkdir(parents=True)
    seed = tmp_path / "seed.jsonl"
    seed.write_text('{"text": "hello"}\n', encoding="utf-8")
    ref_script = Path(__file__).resolve().parents[1] / "scripts" / "reference_train_eval.py"
    raw = {
        "project": {
            "workspace_dir": str(tmp_path / "workspace"),
            "agent_root": str(agent_root),
            "seed_data": str(seed),
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
        "trainer": {"dev_eval_cmd": f'{sys.executable} "{ref_script}"', "timeout_sec": 60},
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
    profile = TaskProfile("生成问答", {"instruction": "str", "answer": "str"}, "qa", ["答案非空"], "qa-sft")
    monkeypatch.setattr(orch.profiler, "run", lambda *a, **k: profile)

    # 桩：ReviewAgent 总是通过
    monkeypatch.setattr(
        orch.reviewer, "review",
        lambda dataset_path, prof: ReviewResult(0.9, 0.9, True, []),
    )
    # 桩：DiagnosticAgent 返回空报告
    monkeypatch.setattr(
        orch.diagnostician, "analyze",
        lambda bad, downstream="": DiagnosticReport(summary="ok"),
    )

    # 桩：PipelineAgent.generate 写出一个能跑的 pipeline.py 并返回 config
    def fake_generate(profile, node_dir_name, diagnostic=None, memory_summary=None, review_issues=None, parent_code=None):
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

    best = orch.run("生成问答数据集")
    assert best is not None
    assert best.train_result is not None
    assert best.train_result.failed is False
    # 参考脚本：5 条非空 answer → score=1.0
    assert abs(best.train_result.score - 1.0) < 1e-9
    # 跑了 3 次迭代，应有若干已评估节点
    assert len(orch.tree.all_nodes()) >= 1


def test_resolve_pipeline_llm_fallback_and_override():
    """pipeline_llm 段逐字段回退 api：缺省全继承；只写 model_name 则只覆盖模型。"""
    from dataflow_evolve.main import _resolve_pipeline_llm

    # 缺省 pipeline_llm → 全部回退 api（行为与旧版一致）
    llm = Config({"api": {"api_url": "u", "model_name": "m", "api_key": "k", "max_workers": 7}})
    assert _resolve_pipeline_llm(llm) == {
        "api_url": "u", "model_name": "m", "api_key": "k", "max_workers": 7,
    }

    # 只写 model_name → 覆盖模型，其余继承 api（算子用便宜模型、辅助 Agent 用强模型的核心诉求）
    llm2 = Config({
        "api": {"api_url": "u", "model_name": "strong", "api_key": "k", "max_workers": 7},
        "pipeline_llm": {"model_name": "cheap"},
    })
    r2 = _resolve_pipeline_llm(llm2)
    assert r2["model_name"] == "cheap"
    assert r2["api_url"] == "u" and r2["api_key"] == "k" and r2["max_workers"] == 7
