"""终选 Tier-2 单测：top-K 排序、放大重产 env、安全阀跳过用次优、按 dev 选最优、终评触发。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.core.schemas import PipelineConfig, TaskProfile, TrainResult

from _ptb_stubs import (
    StubExecutor,
    StubReviewer,
    StubTrainer,
    bare_orchestrator,
    make_review_result,
)

PROFILE = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")


def _add_candidate(o, tier1, dup=0.0):
    child = o.tree.reserve_child(o.tree.root)
    child.config = PipelineConfig(operators=[], field_flow="f", code="code", rationale="r")
    child.review_result = make_review_result(tier1=tier1, dup=dup)
    child.visits = 1
    return child


def _patch_final_eval(o, captured):
    def _fake(final_model):
        captured.append(final_model)
        return 0.55
    o._run_final_eval = _fake


def test_top_k_selects_best_dev_and_uses_4000(tmp_path):
    o = bare_orchestrator(tmp_path, tier2_top_k=2, time_budget_sec=0.0)
    o._run_start = time.time()
    o.final_result = None
    a = _add_candidate(o, tier1=0.9)
    b = _add_candidate(o, tier1=0.7)
    _add_candidate(o, tier1=0.5)

    o.executor = StubExecutor(tmp_path / "ds")
    # 重产后复评全部合格
    o.reviewer = StubReviewer([make_review_result(tier1=0.8)] * 5)
    # 训练：A(0.9) 先跑 → 0.60；B(0.7) 次跑 → 0.80
    o.trainer = StubTrainer([TrainResult(0.60, "gsm8k"), TrainResult(0.80, "gsm8k")])
    captured = []
    _patch_final_eval(o, captured)

    o._final_selection(PROFILE)

    assert o.final_result is not None
    assert o.final_result["best_node"] == b.node_dir_name      # dev 0.80 胜出
    assert o.final_result["dev_score"] == pytest.approx(0.80)
    assert o.final_result["test_score"] == pytest.approx(0.55)
    assert len(captured) == 1                                   # 终评跑了一次
    # 放大重产用 target_samples=4000
    envs = [env for _, env in o.executor.calls]
    assert {"DF_TARGET_SAMPLES": "4000"} in envs
    assert o.trainer.calls == 2                                 # 只跑 top-2


def test_safety_valve_skips_collapsed_and_uses_next(tmp_path):
    o = bare_orchestrator(tmp_path, tier2_top_k=2, time_budget_sec=0.0)
    o._run_start = time.time()
    o.final_result = None
    a = _add_candidate(o, tier1=0.9)
    b = _add_candidate(o, tier1=0.7)
    c = _add_candidate(o, tier1=0.5)

    o.executor = StubExecutor(tmp_path / "ds")
    # A 重产后质量塌陷（dup 高）；B、C 合格
    o.reviewer = StubReviewer([
        make_review_result(tier1=0.1, dup=0.9),   # A → collapsed
        make_review_result(tier1=0.7),            # B → ok
        make_review_result(tier1=0.5),            # C → ok
    ])
    o.trainer = StubTrainer([TrainResult(0.80, "gsm8k"), TrainResult(0.60, "gsm8k")])
    captured = []
    _patch_final_eval(o, captured)

    o._final_selection(PROFILE)

    assert o.reviewer.calls == 3            # A/B/C 都做了重产复评
    assert o.trainer.calls == 2             # A 被跳过，B/C 训练
    assert o.final_result["best_node"] == b.node_dir_name   # B dev 0.80


def test_all_collapse_yields_no_final(tmp_path):
    o = bare_orchestrator(tmp_path, tier2_top_k=2, time_budget_sec=0.0)
    o._run_start = time.time()
    o.final_result = None
    _add_candidate(o, tier1=0.9)
    _add_candidate(o, tier1=0.7)

    o.executor = StubExecutor(tmp_path / "ds")
    o.reviewer = StubReviewer([make_review_result(tier1=0.1, dup=0.99)] * 5)
    o.trainer = StubTrainer([TrainResult(0.80, "gsm8k")])
    _patch_final_eval(o, [])

    o._final_selection(PROFILE)
    assert o.final_result is None
    assert o.trainer.calls == 0


def test_budget_reduces_k_to_one(tmp_path):
    # 预算只够 1 次 Tier-2：k 应缩到 1
    o = bare_orchestrator(
        tmp_path, tier2_top_k=3, time_budget_sec=250.0,
        tier2_per_run_sec=100.0, final_eval_reserve_sec=100.0,
    )
    o._run_start = time.time()   # 剩余 ~ 250-0-100=150 → affordable=1
    o.final_result = None
    a = _add_candidate(o, tier1=0.9)
    _add_candidate(o, tier1=0.7)
    _add_candidate(o, tier1=0.5)

    o.executor = StubExecutor(tmp_path / "ds")
    o.reviewer = StubReviewer([make_review_result(tier1=0.8)] * 5)
    o.trainer = StubTrainer([TrainResult(0.70, "gsm8k")] * 5)
    _patch_final_eval(o, [])

    o._final_selection(PROFILE)
    assert o.trainer.calls == 1                         # 缩到 K=1
    assert o.final_result["best_node"] == a.node_dir_name


def test_tier2_review_written_back_to_node(tmp_path):
    """放大后的重评必须回写节点：它与 train.score 出自同一份数据，是唯一可比的样本对。

    历史实现只把 re_review 用于塌陷安全阀，算完即弃，导致 tier1 与下游真分的相关性无从检验。
    """
    o = bare_orchestrator(tmp_path, tier2_top_k=1, time_budget_sec=0.0)
    o._run_start = time.time()
    o.final_result = None
    node = _add_candidate(o, tier1=0.9)

    o.executor = StubExecutor(tmp_path / "ds")
    o.reviewer = StubReviewer([make_review_result(tier1=0.62)])
    o.trainer = StubTrainer([TrainResult(0.71, "gsm8k")])
    _patch_final_eval(o, [])

    o._final_selection(PROFILE)

    assert node.tier2_review_result is not None
    assert node.tier2_review_result.tier1_score == pytest.approx(0.62)
    assert node.review_result.tier1_score == pytest.approx(0.9)   # 搜索期探针分不被覆盖
    assert node.train_result.score == pytest.approx(0.71)          # 与之配对的下游真分


def test_tier2_review_written_back_even_when_collapsed(tmp_path):
    """质量塌陷而被跳过的候选也要留下重评结果，否则塌陷案例的证据同样丢失。"""
    o = bare_orchestrator(tmp_path, tier2_top_k=1, time_budget_sec=0.0)
    o._run_start = time.time()
    o.final_result = None
    node = _add_candidate(o, tier1=0.9)

    o.executor = StubExecutor(tmp_path / "ds")
    o.reviewer = StubReviewer([make_review_result(tier1=0.1, dup=0.9)])
    o.trainer = StubTrainer([TrainResult(0.71, "gsm8k")])
    _patch_final_eval(o, [])

    o._final_selection(PROFILE)

    assert node.tier2_review_result is not None
    assert node.tier2_review_result.dup_rate == pytest.approx(0.9)
    assert o.trainer.calls == 0                                    # 塌陷候选不训练


def test_review_evidence_path_lands_in_node_dir(tmp_path):
    o = bare_orchestrator(tmp_path)
    assert o._review_evidence_path("node_7") == str(tmp_path / "node_7" / "review.jsonl")


class _StubProfiler:
    def run(self, task_description, sample_text=None, fixed_schema=None):
        return PROFILE


def test_checkpoint_written_after_final_selection(tmp_path):
    """终选产出的 tier2 重评与下游真分必须落进 checkpoint。

    循环内最后一次落盘早于终选，若不在终选后补写一次，这两个字段在真实运行的 checkpoint 里
    永远是 null——而它们正是检验「tier1 能否预测下游真分」的那一对样本。
    """
    ck = tmp_path / "ck.json"
    o = bare_orchestrator(tmp_path, tier2_top_k=1, time_budget_sec=0.0)
    o.max_iterations = 0                    # 跳过搜索循环，只验证终选后的落盘
    o.profiler = _StubProfiler()
    o.checkpoint_path = str(ck)
    node = _add_candidate(o, tier1=0.9)

    o.executor = StubExecutor(tmp_path / "ds")
    o.reviewer = StubReviewer([make_review_result(tier1=0.62)])
    o.trainer = StubTrainer([TrainResult(0.71, "gsm8k")])
    _patch_final_eval(o, [])

    o.run("任务")

    state = json.loads(ck.read_text(encoding="utf-8"))
    nd = next(n for n in state["nodes"] if n["id"] == node.id)
    assert nd["tier2_review_result"]["tier1_score"] == pytest.approx(0.62)
    assert nd["train_result"]["score"] == pytest.approx(0.71)
    assert nd["review_result"]["tier1_score"] == pytest.approx(0.9)   # 探针分仍在
