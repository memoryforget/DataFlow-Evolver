"""终选 Tier-2 单测：top-K 排序、放大重产 env、安全阀跳过用次优、按 dev 选最优、终评触发。"""
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
