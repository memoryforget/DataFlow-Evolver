"""编排器 reward 来源开关单测：review 模式用 tier1 且搜索期不训练；train 模式沿用真分。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.core.baseline import REWARD_ABSOLUTE, RewardPolicy
from dataflow_evolver.core.schemas import TaskProfile, TrainResult
from dataflow_evolver.core.tree_manager import TreeManager

from _ptb_stubs import (
    StubTrainer,
    StubReviewer,
    bare_orchestrator,
    make_review_result,
)

PROFILE = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")


def _run_one_step(o):
    parent = o.tree.root
    child = o.tree.reserve_child(parent)
    o._run_step(child, parent, PROFILE)
    return child


def test_review_mode_reward_is_tier1_and_no_training(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="review")
    o.reviewer = StubReviewer([make_review_result(tier1=0.42)])
    o.trainer = StubTrainer([TrainResult(score=0.99, metric_name="gsm8k")])
    child = _run_one_step(o)

    assert child.visits == 1
    assert child.mean_reward == pytest.approx(0.42)   # reward = tier1，不是训练分
    assert o.trainer.calls == 0                        # 搜索期不训练


def test_review_mode_injects_probe_samples_env(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="review", probe_samples=200)
    o.reviewer = StubReviewer([make_review_result(tier1=0.3)])
    _run_one_step(o)
    # executor 第一次执行应带 DF_TARGET_SAMPLES=probe_samples
    node_dir, env = o.executor.calls[0]
    assert env == {"DF_TARGET_SAMPLES": "200"}


def test_review_mode_failed_gate_penalizes_and_no_training(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="review")
    o.reviewer = StubReviewer([make_review_result(passed=False, tier1=0.0)])
    child = _run_one_step(o)
    assert child.dead is True
    assert o.trainer.calls == 0


def test_train_mode_reward_is_train_score_and_trains(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="train")
    o.reward_policy = RewardPolicy(reward_mode=REWARD_ABSOLUTE)
    o.reviewer = StubReviewer([make_review_result(passed=True, tier1=0.1)])
    o.trainer = StubTrainer([TrainResult(score=0.77, metric_name="math")])
    child = _run_one_step(o)

    assert o.trainer.calls == 1                         # train 模式确实训练
    assert child.mean_reward == pytest.approx(0.77)     # reward = 训练真分
    assert o.diagnostician.calls == 1                   # train 模式跑诊断


def test_train_mode_does_not_inject_probe_env(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="train")
    o.reviewer = StubReviewer([make_review_result(passed=True)])
    o.trainer = StubTrainer([TrainResult(score=0.5, metric_name="math")])
    _run_one_step(o)
    node_dir, env = o.executor.calls[0]
    assert env is None                                   # train 模式不注入 DF_TARGET_SAMPLES
