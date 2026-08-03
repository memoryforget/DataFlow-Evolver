"""下游 schema 契约单测：配置说了算、评审在训练前拦下字段不符。

守的是一条因果链：训练脚本按固定字段名取值 → 取不到就整份丢光 → 报"非零退出"。
所以字段名不能由 LLM 现编，也不能等到训练阶段才发现对不上。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.agents.task_profiler import TaskProfilerAgent
from dataflow_evolver.core.schemas import ReviewResult, TaskProfile
from dataflow_evolver.llm.prompts import build_profiler_prompt

CONTRACT = {"instruction": "str", "output": "str"}


class _Serving:
    """回放一份预置的 profiler JSON。"""

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate_one(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return json.dumps(self.payload, ensure_ascii=False)


def _profiler(schema_from_llm):
    serving = _Serving({
        "task_description": "数学 CoT 合成",
        "target_schema": schema_from_llm,
        "quality_criteria": ["推理自洽"],
        "downstream_task": "math-QA-SFT",
    })
    return TaskProfilerAgent(serving), serving


def _write(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return str(path)


# ---- 契约由配置决定 --------------------------------------------------------
def test_configured_schema_overrides_what_the_model_invents():
    """problem/solution 对数学任务是完全合理的臆造——但训练脚本一条都读不到。"""
    agent, _ = _profiler({"problem": "str", "solution": "str"})
    profile = agent.run("造数学题", fixed_schema=CONTRACT)
    assert profile.target_schema == CONTRACT


def test_the_model_is_told_the_schema_not_just_overridden_afterwards():
    """只在事后覆盖的话，模型的 quality_criteria 会围绕它自己臆造的字段展开。"""
    agent, serving = _profiler({"problem": "str"})
    agent.run("造数学题", fixed_schema=CONTRACT)
    assert "instruction" in serving.prompts[0]


def test_without_a_contract_the_model_still_decides():
    """留空 = 旧行为，不是所有下游都有可枚举的固定字段。"""
    agent, _ = _profiler({"problem": "str", "solution": "str"})
    profile = agent.run("造数学题")
    assert profile.target_schema == {"problem": "str", "solution": "str"}


def test_profiler_prompt_stays_clean_when_no_contract():
    assert "schema 已固定" not in build_profiler_prompt("造题")
    assert "schema 已固定" in build_profiler_prompt("造题", fixed_schema=CONTRACT)


# ---- 评审硬门槛 ------------------------------------------------------------
def _review_with(tmp_path, rows, threshold=0.5):
    agent = ReviewAgent(serving=None, null_threshold=threshold)
    profile = TaskProfile("t", dict(CONTRACT), [], "sft")
    result = ReviewResult(schema_score=1.0, relevance_score=1.0, passed=True)
    result.llm_composite = 0.9
    path = _write(tmp_path / "ds.jsonl", rows)
    return agent.apply_hard_metrics(result, path, profile)


def test_wrong_field_names_are_stopped_before_any_gpu_is_spent(tmp_path):
    rows = [{"problem": f"q{i}", "solution": f"s{i}"} for i in range(20)]
    result = _review_with(tmp_path, rows)

    assert result.passed is False
    assert result.tier1_score == pytest.approx(0.0)


def test_the_rejection_says_which_fields_are_expected(tmp_path):
    """自修正循环拿这条 issue 去改代码，说不清要什么字段就修不好。"""
    rows = [{"problem": f"q{i}", "solution": f"s{i}"} for i in range(20)]
    (issue,) = _review_with(tmp_path, rows).issues

    assert "instruction" in issue and "output" in issue


def test_matching_fields_pass_the_gate(tmp_path):
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(20)]
    result = _review_with(tmp_path, rows)

    assert result.passed is True
    assert result.null_rate == pytest.approx(0.0)


def test_a_few_missing_rows_are_discounted_not_rejected(tmp_path):
    """个别行缺字段是质量问题，按比例扣分；只有整体对不上才是契约问题。"""
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(18)]
    rows += [{"instruction": "q18", "output": ""}, {"instruction": "q19", "output": ""}]
    result = _review_with(tmp_path, rows)

    assert result.passed is True
    assert 0.0 < result.null_rate < 0.5
    assert result.tier1_score < 0.9


# ---- 配置读取 --------------------------------------------------------------
def test_the_schema_survives_the_trip_through_real_config():
    """`Config` 不能直接 `dict()`（没有 keys()，会退化成整数索引）。

    单测里各处都直接塞普通 dict，这层永远不会被触发——只有拿真配置走一遍才看得见。
    """
    from dataflow_evolver.main import _plain_dict
    from dataflow_evolver.utils.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert _plain_dict(cfg.trainer.get("dataset_schema")) == CONTRACT


def test_plain_dict_takes_both_shapes_and_absence():
    from dataflow_evolver.main import _plain_dict

    assert _plain_dict(None) == {}
    assert _plain_dict({"a": "str"}) == {"a": "str"}


def test_no_contract_means_no_gate(tmp_path):
    """没配 target_schema 时不该凭空拦人——此时框架也不知道下游要什么。"""
    agent = ReviewAgent(serving=None)
    profile = TaskProfile("t", {}, [], "sft")
    result = ReviewResult(schema_score=1.0, relevance_score=1.0, passed=True)
    result.llm_composite = 0.9
    path = _write(tmp_path / "ds.jsonl", [{"whatever": "x"} for _ in range(10)])

    assert agent.apply_hard_metrics(result, path, profile).passed is True
