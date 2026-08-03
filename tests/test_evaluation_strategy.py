"""评分策略：口径与其连带纪律必须成套，且拼错的口径名要在启动时炸掉。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.core import evaluation
from dataflow_evolver.core.schemas import TaskProfile, TrainResult
from dataflow_evolver.core.tree_node import PipelineNode

from _ptb_stubs import StubDiagnostician, StubTrainer, make_review_result

PROFILE = TaskProfile("t", {"a": "str"}, [], "math-sft")


def _ctx(tmp_path, **kw):
    base = dict(
        trainer=StubTrainer([TrainResult(score=0.8, metric_name="gsm8k")]),
        diagnostician=StubDiagnostician(),
        agent_root=Path(tmp_path),
        probe_samples=200,
        target_samples=4000,
        tier2_top_k=3,
        tier2_per_run_sec=100.0,
        final_eval_reserve_sec=50.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ---- 建策略 ----------------------------------------------------------------
def test_build_known_names():
    assert isinstance(evaluation.build("train"), evaluation.TrainScore)
    assert isinstance(evaluation.build(" Review "), evaluation.ReviewScore)


def test_build_rejects_unknown_name():
    """拼错口径名若静默回退默认值，一次几小时的实验会安静地跑成另一种实验。"""
    with pytest.raises(ValueError, match="reveiw"):
        evaluation.build("reveiw")


# ---- train 口径：分数一步到位，无需收尾 ------------------------------------
def test_train_scores_by_real_training_and_diagnoses(tmp_path):
    ctx = _ctx(tmp_path)
    node = PipelineNode()
    score = evaluation.TrainScore().score(ctx, node, PROFILE, "/tmp/ds.jsonl")

    assert score == pytest.approx(0.8)
    assert node.train_result is not None
    assert node.diagnostic_report is not None      # 有真实 bad case 才谈得上诊断
    assert ctx.trainer.calls == 1


def test_train_failure_returns_none_without_diagnosing(tmp_path):
    failed = TrainResult(score=0.0, metric_name="gsm8k", failed=True, error_msg="OOM")
    ctx = _ctx(tmp_path, trainer=StubTrainer([failed]))
    node = PipelineNode()
    assert evaluation.TrainScore().score(ctx, node, PROFILE, "/tmp/ds.jsonl") is None
    assert node.diagnostic_report is None
    assert ctx.diagnostician.calls == 0


def test_train_needs_no_scale_up_and_no_reserve(tmp_path):
    s = evaluation.TrainScore()
    assert s.needs_scale_up_selection is False
    assert s.reserve_sec(_ctx(tmp_path)) == 0.0
    assert s.search_env(_ctx(tmp_path)) is None            # 搜索期就出全量数据
    assert s.default_prompt_samples(_ctx(tmp_path)) == 4000


# ---- review 口径：代理分 + 收尾兑现 ----------------------------------------
def test_review_scores_by_tier1_without_training(tmp_path):
    ctx = _ctx(tmp_path)
    node = PipelineNode()
    node.review_result = make_review_result(tier1=0.42)

    assert evaluation.ReviewScore().score(ctx, node, PROFILE, "/tmp/ds.jsonl") == pytest.approx(0.42)
    assert ctx.trainer.calls == 0
    assert ctx.diagnostician.calls == 0


def test_review_runs_probe_scale_and_reserves_for_finalization(tmp_path):
    s = evaluation.ReviewScore()
    ctx = _ctx(tmp_path)
    assert s.search_env(ctx) == {"DF_TARGET_SAMPLES": "200"}   # 搜索期只产探针规模
    assert s.default_prompt_samples(ctx) == 200
    assert s.needs_scale_up_selection is True                  # tier1 只是代理分，须兑现
    assert s.reserve_sec(ctx) == 350.0                         # 3*100 + 50


def test_review_falls_back_to_target_when_no_probe_configured(tmp_path):
    ctx = _ctx(tmp_path, probe_samples=0)
    s = evaluation.ReviewScore()
    assert s.search_env(ctx) is None
    assert s.default_prompt_samples(ctx) == 4000
