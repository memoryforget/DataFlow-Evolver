"""节点内自修正循环的一致性：decision.json 同步、修复历史累积、抽样种子可复现。

三个隐患：
1) 自修正 prompt 从未要求更新 decision.json，导致 PipelineConfig 里 code 是新的、
   operators/field_flow/rationale 是旧的，并传染到 global memory 与子节点继承的 rationale；
2) self_correct_review 只拿到最新一轮 issues，LLM 会推翻上一轮为修 X 加的逻辑，形成震荡；
3) sample_seed 不设时，连同一份未改动文件的两次重评抽样都不同，passed 与 tier1 带纯 RNG 噪声。
"""
from __future__ import annotations

import json
from pathlib import Path

from dataflow_evolver.agents.review_agent import _sample_jsonl
from dataflow_evolver.llm.prompts import build_review_selfcorrect_prompt, build_selfcorrect_prompt
from dataflow_evolver.main import _fix_round_summary

from _ptb_stubs import make_review_result


# ---- 隐患三：decision.json 同步提示 ---------------------------------------
def test_exec_selfcorrect_prompt_asks_to_sync_decision():
    prompt = build_selfcorrect_prompt("node_3", "Traceback: boom")
    assert "node_3/decision.json" in prompt
    assert "增删了算子" in prompt


def test_exec_selfcorrect_prompt_cannot_bypass_compile_or_use_wrong_recipe():
    prompt = build_selfcorrect_prompt(
        "node_3",
        "FileNotFoundError: dataflow_cache_step_step9.jsonl does not exist",
    )
    assert "Issue #012" in prompt
    assert "dataflow/pipeline/Pipeline.py" in prompt
    assert "不得把 `DF_COMPILE_ONLY` 判断移到" in prompt
    assert 'Path(__file__).with_name("recipe.json")' in prompt


def test_review_selfcorrect_prompt_asks_to_sync_decision():
    prompt = build_review_selfcorrect_prompt("node_3", ["qa 为空"])
    assert "node_3/decision.json" in prompt
    assert "`ops` / `field_flow` / `reason`" in prompt


def test_stale_decision_warning_only_when_not_rewritten(tmp_path, monkeypatch, caplog):
    """自修正后 decision.json 未被改动才告警；被改写则不告警。"""
    import dataflow_evolver.agents.pipeline_agent as pa

    node_dir = tmp_path / "node_1"
    node_dir.mkdir()
    (node_dir / "pipeline.py").write_text("print(1)", encoding="utf-8")
    decision = node_dir / "decision.json"
    decision.write_text(
        json.dumps({"ops": [{"name": "Gen", "purpose": "生成"}], "field_flow": "f", "reason": "r"}),
        encoding="utf-8",
    )

    agent = pa.PipelineAgent.__new__(pa.PipelineAgent)
    agent.agent_root = tmp_path
    agent.pipeline_filename = "pipeline.py"
    agent.agent_cfg = {}
    agent.logger = pa.get_logger("test.pipeline_agent")

    class _Ok:
        success = True
        retryable = False
        error = None

    class _Runtime:
        def __init__(self, fn):
            self.fn = fn

        def run(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    # 情形 A：Agent 只改了 pipeline.py
    agent.runtime = _Runtime(lambda *a, **k: _Ok())
    with caplog.at_level("WARNING"):
        cfg = agent._run_and_assemble("p", "node_1", warn_stale_decision=True)
    assert cfg is not None
    assert "未更新" in caplog.text

    # 情形 B：Agent 同步改写了 decision.json
    caplog.clear()
    def _rewrite(*args, **kwargs):
        decision.write_text(
            json.dumps({"ops": [{"name": "Gen2", "purpose": "改后"}], "field_flow": "f2"}),
            encoding="utf-8",
        )
        return _Ok()

    agent.runtime = _Runtime(_rewrite)
    with caplog.at_level("WARNING"):
        cfg = agent._run_and_assemble("p", "node_1", warn_stale_decision=True)
    assert "未更新" not in caplog.text
    assert cfg is not None and cfg.operators[0]["name"] == "Gen2"


def test_first_generation_does_not_warn(tmp_path, monkeypatch, caplog):
    """首次生成不做新鲜度检查（decision.json 本来就不存在，不该刷告警）。"""
    import dataflow_evolver.agents.pipeline_agent as pa

    node_dir = tmp_path / "node_1"
    node_dir.mkdir()
    (node_dir / "pipeline.py").write_text("print(1)", encoding="utf-8")

    agent = pa.PipelineAgent.__new__(pa.PipelineAgent)
    agent.agent_root = tmp_path
    agent.pipeline_filename = "pipeline.py"
    agent.agent_cfg = {}
    agent.logger = pa.get_logger("test.pipeline_agent")

    class _Ok:
        success = True
        retryable = False
        error = None

    class _Runtime:
        def run(self, *args, **kwargs):
            return _Ok()

    agent.runtime = _Runtime()
    with caplog.at_level("WARNING"):
        agent._run_and_assemble("p", "node_1")
    assert "未更新" not in caplog.text


# ---- 隐患二：修复历史累积 --------------------------------------------------
def test_review_selfcorrect_prompt_injects_history():
    prompt = build_review_selfcorrect_prompt(
        "node_3",
        ["多样性仍偏低"],
        history=["第 1 轮针对：qa 字段为空 → 修复后 tier1 0.100→0.420（schema 0.30→0.80），评审仍未通过"],
    )
    assert "已尝试过的修复" in prompt
    assert "第 1 轮针对：qa 字段为空" in prompt


def test_review_selfcorrect_prompt_without_history():
    prompt = build_review_selfcorrect_prompt("node_3", ["qa 为空"])
    assert "已尝试过的修复" not in prompt


def test_fix_round_summary_separates_certain_from_reference():
    """硬指标是全量扫描的确定量，LLM 五维来自不同抽样——两类必须分开表述。"""
    before = make_review_result(tier1=0.10, dup=0.30)
    before.issues = ["qa 字段为空", "答案与题目不符"]
    before.schema_score = 0.30
    after = make_review_result(tier1=0.42, passed=False, dup=0.05)
    after.schema_score = 0.80

    summary = _fix_round_summary(1, before, after)
    assert "第 1 轮针对：qa 字段为空；答案与题目不符" in summary
    certain, reference = [ln for ln in summary.splitlines() if "变化（" in ln]
    assert "重复率 0.30→0.05" in certain and "全量扫描" in certain
    assert "schema 0.30→0.80" in reference and "Tier-1 复合分 0.10→0.42" in reference
    assert "不能据此断定" in reference
    assert "仍未通过" in summary


def test_fix_round_summary_suppresses_llm_dim_noise():
    """0.02 的 LLM 维度抖动远在 judge 噪声量级内，不能渲染成"改动见效"。"""
    before = make_review_result(tier1=0.40)
    before.issues = []
    before.diversity_score, before.correctness_score = 0.50, 0.60
    after = make_review_result(tier1=0.42, passed=False)
    after.diversity_score, after.correctness_score = 0.58, 0.66   # 均 < 0.10

    summary = _fix_round_summary(2, before, after)
    assert "多样性" not in summary and "正确性" not in summary
    assert "Tier-1" not in summary
    assert "未观察到超出噪声量级的变化" in summary


def test_fix_round_summary_reports_hard_metric_small_delta():
    """硬指标是确定量，0.05 的变化就该报——阈值不能和 LLM 维度共用。"""
    before = make_review_result(tier1=0.40, null=0.10)
    before.issues = []
    after = make_review_result(tier1=0.40, passed=True, null=0.02)

    summary = _fix_round_summary(3, before, after)
    assert "空值率 0.10→0.02" in summary
    assert "重评结论：通过" in summary


def test_fix_round_summary_caps_targets():
    """"针对什么"只取前几条，避免历史块随 issues 数量膨胀。"""
    before = make_review_result(tier1=0.10)
    before.issues = ["问题甲", "问题乙", "问题丙", "问题丁"]
    after = make_review_result(tier1=0.20, passed=False)

    summary = _fix_round_summary(1, before, after, max_targets=2)
    assert "问题甲；问题乙" in summary
    assert "问题丙" not in summary


def test_history_accumulates_across_rounds(tmp_path, monkeypatch):
    """编排器应把逐轮摘要累积传给下一轮，而非只给最新一轮。"""
    from _ptb_stubs import StubExecutor, StubReviewer, bare_orchestrator

    o = bare_orchestrator(tmp_path, review_self_correction_retries=2)
    o.executor = StubExecutor(tmp_path / "ds")
    # 首评不过 → 两轮自修正后仍不过
    o.reviewer = StubReviewer([
        make_review_result(tier1=0.10, passed=False),
        make_review_result(tier1=0.30, passed=False),
        make_review_result(tier1=0.40, passed=False),
    ])

    seen: list[list[str]] = []

    def fake_self_correct_review(node_dir, issues, history=None, **kwargs):
        seen.append(list(history or []))
        from dataflow_evolver.core.schemas import PipelineConfig
        return PipelineConfig(operators=[], field_flow="f", code="x", rationale="r")

    monkeypatch.setattr(o.pipeline_agent, "self_correct_review", fake_self_correct_review)

    parent = o.tree.root
    child = o.tree.reserve_child(parent)
    child.config = None
    monkeypatch.setattr(
        o.pipeline_agent, "generate",
        lambda **kwargs: __import__(
            "dataflow_evolver.core.schemas", fromlist=["PipelineConfig"]
        ).PipelineConfig(operators=[], field_flow="f", code="x", rationale="r"),
    )
    o._run_step(child, parent, None)

    assert len(seen) == 2
    assert seen[0] == []                        # 第 1 轮还没有历史
    assert len(seen[1]) == 1                    # 第 2 轮带上第 1 轮的成效
    assert "第 1 轮针对" in seen[1][0]


# ---- 隐患一：抽样种子 ------------------------------------------------------
def _dataset(tmp_path: Path, n: int = 50) -> str:
    path = tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(json.dumps({"q": f"问题{i}"}, ensure_ascii=False) for i in range(n)),
        encoding="utf-8",
    )
    return str(path)


def test_fixed_seed_makes_resampling_reproducible(tmp_path):
    """同一份未改动文件 + 固定种子 → 抽样一致，消掉纯 RNG 噪声。"""
    dataset = _dataset(tmp_path)
    assert _sample_jsonl(dataset, 10, seed=42) == _sample_jsonl(dataset, 10, seed=42)


def test_unset_seed_can_differ_across_calls(tmp_path):
    """种子不设时，同一份文件多次抽样会不同——这正是当前默认配置的噪声来源。"""
    dataset = _dataset(tmp_path, n=500)
    samples = {_sample_jsonl(dataset, 20, seed=None) for _ in range(8)}
    assert len(samples) > 1


def test_ptb_config_pins_sample_seed():
    import yaml

    cfg = yaml.safe_load(
        Path(__file__).resolve().parents[1].joinpath("configs/posttrainbench_gsm8k.yaml")
        .read_text(encoding="utf-8")
    )
    assert cfg["review"]["sample_seed"] == 42


def test_fix_history_persisted_to_node_dir(tmp_path, monkeypatch):
    """prompt 里实际喂回的那段文字受阈值影响，无法从 review.jsonl 还原，故单独落盘。"""
    from dataflow_evolver.core.schemas import PipelineConfig

    from _ptb_stubs import StubExecutor, StubReviewer, bare_orchestrator

    o = bare_orchestrator(tmp_path, review_self_correction_retries=1)
    o.executor = StubExecutor(tmp_path / "ds")
    first = make_review_result(tier1=0.10, passed=False, dup=0.4)
    first.issues = ["qa 字段为空"]
    o.reviewer = StubReviewer([first, make_review_result(tier1=0.30, passed=False, dup=0.1)])

    cfg = PipelineConfig(operators=[], field_flow="f", code="x", rationale="r")
    monkeypatch.setattr(o.pipeline_agent, "generate", lambda **kwargs: cfg)
    monkeypatch.setattr(
        o.pipeline_agent, "self_correct_review", lambda node_dir, issues, **kwargs: cfg
    )

    child = o.tree.reserve_child(o.tree.root)
    o._run_step(child, o.tree.root, None)

    path = tmp_path / child.node_dir_name / "fix_history.jsonl"
    (rec,) = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert rec["round"] == 1
    assert rec["targets"] == ["qa 字段为空"]
    assert "重复率 0.40→0.10" in rec["rendered"]
    assert "ts" in rec
