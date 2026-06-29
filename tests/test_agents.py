import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolve.agents.diagnostic_agent import DiagnosticAgent
from dataflow_evolve.agents.review_agent import ReviewAgent
from dataflow_evolve.agents.task_profiler import TaskProfilerAgent
from dataflow_evolve.core.schemas import BadCase
from dataflow_evolve.llm.serving import LLMServingABC


class StubServing(LLMServingABC):
    """返回预设文本的假后端，免去真实 API 调用。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._i = 0

    def generate(self, prompts, **kwargs):
        out = []
        for _ in prompts:
            out.append(self._responses[min(self._i, len(self._responses) - 1)])
            self._i += 1
        return out


def test_profiler_parses_profile():
    stub = StubServing([
        '{"task_description":"生成数学问答","target_schema":{"instruction":"str","answer":"str"},'
        '"domain":"math","quality_criteria":["答案正确"],"downstream_task":"math-QA-SFT"}'
    ])
    agent = TaskProfilerAgent(stub)
    profile = agent.run("做一个数学问答数据集")
    assert profile is not None
    assert profile.domain == "math"
    assert profile.target_schema == {"instruction": "str", "answer": "str"}
    assert profile.downstream_task == "math-QA-SFT"


def test_review_threshold_pass_and_fail(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"instruction":"a","answer":"b"}\n', encoding="utf-8")
    from dataflow_evolve.core.schemas import TaskProfile

    profile = TaskProfile("t", {"instruction": "str", "answer": "str"}, "math", [], "sft")

    high = StubServing(['{"schema_score":0.9,"relevance_score":0.8,"issues":[]}'])
    r = ReviewAgent(high, schema_threshold=0.6, relevance_threshold=0.6).review(str(ds), profile)
    assert r.passed is True

    low = StubServing(['{"schema_score":0.3,"relevance_score":0.9,"issues":["缺字段"]}'])
    r2 = ReviewAgent(low, schema_threshold=0.6, relevance_threshold=0.6).review(str(ds), profile)
    assert r2.passed is False
    assert r2.issues == ["缺字段"]


def test_review_empty_dataset_fails(tmp_path):
    from dataflow_evolve.core.schemas import TaskProfile

    profile = TaskProfile("t", {}, "d", [], "sft")
    r = ReviewAgent(StubServing([""]) ).review(str(tmp_path / "missing.jsonl"), profile)
    assert r.passed is False


def test_diagnostic_empty_bad_cases_short_circuits():
    agent = DiagnosticAgent(StubServing(["{}"]))
    rep = agent.analyze([], downstream_task="sft")
    assert rep.actionable_suggestions == []
    assert "表现良好" in rep.summary


def test_diagnostic_parses_report():
    stub = StubServing([
        '{"summary":"几何题失败","failure_patterns":["分数几何"],'
        '"actionable_suggestions":["增加几何分数题"]}'
    ])
    agent = DiagnosticAgent(stub)
    bad = [BadCase("q", "wrong", "right", "calc")]
    rep = agent.analyze(bad, downstream_task="math-QA-SFT")
    assert rep.summary == "几何题失败"
    assert rep.actionable_suggestions == ["增加几何分数题"]


def test_pipeline_prompt_injects_review_issues():
    """Phase-1 评审打回的 issues 应注入下一轮生成 prompt；无 issues 则不出现该段。"""
    from dataflow_evolve.core.schemas import TaskProfile
    from dataflow_evolve.llm.prompts import build_pipeline_prompt

    profile = TaskProfile("QA", {"q": "str", "a": "str"}, "general", [], "sft")
    base = build_pipeline_prompt(profile, "seed.jsonl", "node_1")
    assert "评审打回" not in base

    with_issues = build_pipeline_prompt(
        profile, "seed.jsonl", "node_1", review_issues=["qa 字段为空", "与任务不相关"]
    )
    assert "评审打回" in with_issues
    assert "qa 字段为空" in with_issues and "与任务不相关" in with_issues


def test_review_selfcorrect_prompt_points_to_intermediate_cache():
    """评审自修正 prompt 应告知 Agent：cache 下有每个算子的中间产物，去 Read 定位坏在哪一步。"""
    from dataflow_evolve.llm.prompts import build_review_selfcorrect_prompt

    p = build_review_selfcorrect_prompt("node_1", ["answer 字段为空"])
    assert "answer 字段为空" in p
    assert "每个算子" in p and "cache" in p and "forward()" in p

