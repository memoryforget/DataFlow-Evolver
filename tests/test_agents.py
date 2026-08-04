import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.agents.diagnostic_agent import DiagnosticAgent
from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.agents.task_profiler import TaskProfilerAgent
from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.schemas import BadCase
from dataflow_evolver.llm.serving import LLMServingABC

SCRATCH = InputCorpus(entry_path="empty.jsonl")   # 无输入语料 = 从零合成


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
        '"quality_criteria":["答案正确"],"downstream_task":"math-QA-SFT"}'
    ])
    agent = TaskProfilerAgent(stub)
    profile = agent.run("做一个数学问答数据集")
    assert profile is not None
    assert profile.target_schema == {"instruction": "str", "answer": "str"}
    assert profile.downstream_task == "math-QA-SFT"


def test_review_threshold_pass_and_fail(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"instruction":"a","answer":"b"}\n', encoding="utf-8")
    from dataflow_evolver.core.schemas import TaskProfile

    profile = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")

    high = StubServing(['{"schema_score":0.9,"relevance_score":0.8,"issues":[]}'])
    r = ReviewAgent(high, schema_threshold=0.6, relevance_threshold=0.6).review(str(ds), profile)
    assert r.passed is True

    low = StubServing(['{"schema_score":0.3,"relevance_score":0.9,"issues":["缺字段"]}'])
    r2 = ReviewAgent(low, schema_threshold=0.6, relevance_threshold=0.6).review(str(ds), profile)
    assert r2.passed is False
    assert r2.issues == ["缺字段"]


def test_review_empty_dataset_fails(tmp_path):
    from dataflow_evolver.core.schemas import TaskProfile

    profile = TaskProfile("t", {}, [], "sft")
    r = ReviewAgent(StubServing([""]) ).review(str(tmp_path / "missing.jsonl"), profile)
    assert r.passed is False


class _RaisingServing(LLMServingABC):
    """每次调用都抛异常的后端，用于验证 run_json 的兜底（review H1）。"""

    def generate(self, prompts, **kwargs):
        raise RuntimeError("网络炸了")


def test_review_survives_serving_exception(tmp_path):
    """后端抛异常不应冒泡到主循环：Review 须降级为未通过而非崩溃。"""
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"instruction":"a","answer":"b"}\n', encoding="utf-8")
    from dataflow_evolver.core.schemas import TaskProfile

    profile = TaskProfile("t", {"instruction": "str"}, [], "sft")
    r = ReviewAgent(_RaisingServing()).review(str(ds), profile)
    assert r.passed is False


def test_diagnostic_survives_serving_exception():
    """诊断后端抛异常时返回空报告，不阻塞主循环。"""
    rep = DiagnosticAgent(_RaisingServing()).analyze(
        [BadCase("q", "wrong", "right", "calc")], downstream_task="sft"
    )
    assert rep.actionable_suggestions == []


def test_diagnostic_empty_bad_cases_short_circuits():
    agent = DiagnosticAgent(StubServing(["{}"]))
    rep = agent.analyze([], downstream_task="sft")
    assert rep.actionable_suggestions == []
    assert rep.summary == ""


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
    """评审 issues 经 parent_review 注入；parent_review 缺失则不出现该段。

    曾另有一条只收 issues 列表的 elif 分支，与 parent_review 表达同一语义、格式却不同。
    生产代码从不走它，测试却在测它——绿灯不代表真实路径可用，故删掉分支、用例改走这里。
    """
    from dataflow_evolver.core.schemas import ReviewResult, TaskProfile
    from dataflow_evolver.llm.prompts import build_pipeline_prompt

    profile = TaskProfile("QA", {"q": "str", "a": "str"}, [], "sft")
    base = build_pipeline_prompt(profile, SCRATCH, "node_1")
    assert "父节点数据质量" not in base

    review = ReviewResult(
        schema_score=0.7, relevance_score=0.7, passed=True,
        issues=["qa 字段为空", "与任务不相关"],
    )
    with_issues = build_pipeline_prompt(
        profile, SCRATCH, "node_1", parent_review=review
    )
    assert "父节点数据质量归因" in with_issues
    assert "qa 字段为空" in with_issues and "与任务不相关" in with_issues


def test_pipeline_prompt_injects_target_samples():
    """目标产出规模：>0 时注入规模指引（含 num_samples 提示）；默认 0 不注入。"""
    from dataflow_evolver.core.schemas import TaskProfile
    from dataflow_evolver.llm.prompts import build_pipeline_prompt

    profile = TaskProfile("QA", {"instruction": "str", "output": "str"}, [], "sft")
    base = build_pipeline_prompt(profile, SCRATCH, "node_1")
    assert "目标产出规模" not in base                      # 默认 0 → 不注入

    sized = build_pipeline_prompt(profile, SCRATCH, "node_1", target_samples=1500)
    assert "目标产出规模" in sized and "1500" in sized
    assert "num_samples" in sized                          # 指导从零合成类算子


def test_review_selfcorrect_prompt_points_to_intermediate_cache():
    """评审自修正 prompt 应告知 Agent：cache 下有每个算子的中间产物，去 Read 定位坏在哪一步。"""
    from dataflow_evolver.llm.prompts import build_review_selfcorrect_prompt

    p = build_review_selfcorrect_prompt("node_1", ["answer 字段为空"])
    assert "answer 字段为空" in p
    assert "每个算子" in p and "cache" in p and "forward()" in p

