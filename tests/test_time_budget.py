"""时间预算终止单测：_final_reserve_sec / _search_should_stop 行为。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _ptb_stubs import bare_orchestrator

from dataflow_evolver.core import evaluation


def test_reserve_only_in_review_mode(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="review", tier2_top_k=3,
                          tier2_per_run_sec=100.0, final_eval_reserve_sec=100.0)
    assert o._final_reserve_sec() == 400.0        # 3*100 + 100

    o2 = bare_orchestrator(tmp_path, reward_source="train")
    assert o2._final_reserve_sec() == 0.0


def test_should_stop_respects_reserve(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="review", time_budget_sec=1000.0,
                          tier2_top_k=3, tier2_per_run_sec=100.0, final_eval_reserve_sec=100.0)
    # reserve=400 → 搜索截止在 elapsed>=600
    o._run_start = time.time() - 500
    assert o._search_should_stop() is False
    o._run_start = time.time() - 700
    assert o._search_should_stop() is True


def test_should_never_stop_when_budget_disabled(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="review", time_budget_sec=0.0)
    o._run_start = time.time() - 100000
    assert o._search_should_stop() is False


def test_samples_env_helper():
    assert evaluation.samples_env(200) == {"DF_TARGET_SAMPLES": "200"}
    assert evaluation.samples_env(0) is None
    assert evaluation.samples_env(-5) is None
