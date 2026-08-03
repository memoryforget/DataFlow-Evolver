"""训练集切齐单测：超量被切、切法各节点一致、不足只记录、两条训练路径都受约束。

底线是一句话：训练配方冻结时，喂进去多少条数据就等于给了多少算力预算。这些用例守的是
"节点之间算力预算相同"，而不是文件长什么样。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.core import train_set
from dataflow_evolver.core.schemas import PipelineConfig, TaskProfile, TrainResult
from dataflow_evolver.engine.trainer_wrapper import CappedTrainer

from _ptb_stubs import (
    StubExecutor,
    StubReviewer,
    StubTrainer,
    bare_orchestrator,
    make_review_result,
)

PROFILE = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")


def _write(path, n, suffix=".jsonl"):
    rows = [{"instruction": f"q{i}", "answer": f"a{i}"} for i in range(n)]
    p = Path(path).with_suffix(suffix)
    if suffix == ".json":
        p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    else:
        p.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    return p


def _rows(path):
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _manifest(out_dir):
    return json.loads((Path(out_dir) / train_set.MANIFEST_NAME).read_text(encoding="utf-8"))


# ---- 切齐本身 --------------------------------------------------------------
def test_overproduction_is_cut_down_to_the_cap(tmp_path):
    src = _write(tmp_path / "ds", 500)
    ts = train_set.materialize(src, tmp_path / "out", cap=100)

    assert ts.used == 100 and ts.produced == 500
    assert len(_rows(ts.path)) == 100
    assert Path(ts.path) != src                    # 原始产出保持不动，供评审复查


def test_every_node_gets_the_same_cut(tmp_path):
    """同样的超量产出必须切成同一批——否则"同规模"换成了"随机不同的规模内容"。"""
    a = train_set.materialize(_write(tmp_path / "a", 500), tmp_path / "oa", cap=100)
    b = train_set.materialize(_write(tmp_path / "b", 500), tmp_path / "ob", cap=100)
    assert _rows(a.path) == _rows(b.path)


def test_data_within_the_cap_reaches_the_trainer_untouched(tmp_path):
    src = _write(tmp_path / "ds", 80)
    ts = train_set.materialize(src, tmp_path / "out", cap=100)

    assert ts.path == str(src)                     # 不做无谓的复制
    assert ts.used == 80 and not ts.truncated


def test_a_shortfall_cannot_be_cut_away_so_it_is_recorded(tmp_path):
    """产出不足是反方向的同一个问题（欠训），切不出来，但不能安静地混进分数里。"""
    train_set.materialize(_write(tmp_path / "ds", 40), tmp_path / "out", cap=100)
    m = _manifest(tmp_path / "out")
    assert m["short"] is True and m["produced"] == 40 and m["cap"] == 100


def test_manifest_lands_beside_the_training_output(tmp_path):
    src = _write(tmp_path / "ds", 500)
    train_set.materialize(src, tmp_path / "out", cap=100, seed=7)
    m = _manifest(tmp_path / "out")

    assert m["produced"] == 500 and m["used"] == 100 and m["truncated"] is True
    assert m["seed"] == 7 and m["source"] == str(src)


def test_no_cap_means_data_volume_is_free_to_vary(tmp_path):
    """关掉切齐 = 把"数据量"也放进搜索空间，是消融时的显式选择。"""
    src = _write(tmp_path / "ds", 500)
    ts = train_set.materialize(src, tmp_path / "out", cap=0)
    assert ts.path == str(src) and ts.used == 500


def test_a_json_array_output_stays_a_json_array(tmp_path):
    """训练脚本按扩展名决定怎么读；换了容器格式它会直接读不出数据。"""
    src = _write(tmp_path / "ds", 500, suffix=".json")
    ts = train_set.materialize(src, tmp_path / "out", cap=100)

    assert Path(ts.path).suffix == ".json"
    assert isinstance(json.loads(Path(ts.path).read_text(encoding="utf-8")), list)
    assert len(_rows(ts.path)) == 100


def test_an_unreadable_dataset_still_reaches_the_trainer(tmp_path):
    """空/坏数据交给训练脚本去报错——那里的信息比这里具体。"""
    src = tmp_path / "broken.jsonl"
    src.write_text("not json at all\n", encoding="utf-8")
    ts = train_set.materialize(src, tmp_path / "out", cap=100)
    assert ts.path == str(src)


# ---- 包装层：两条训练路径都受约束 ------------------------------------------
def test_the_trainer_only_ever_sees_the_capped_file(tmp_path):
    src = _write(tmp_path / "ds", 500)
    stub = StubTrainer([TrainResult(0.5, "gsm8k")])
    CappedTrainer(stub, cap=100).run(str(src), str(tmp_path / "out"))

    assert len(_rows(stub.datasets[0])) == 100


def test_search_scoring_trains_on_the_capped_data(tmp_path):
    o = bare_orchestrator(tmp_path, reward_source="train", target_samples=10)
    o.executor = StubExecutor(tmp_path / "ds", rows=50)
    o.reviewer = StubReviewer([make_review_result(passed=True)])
    stub = StubTrainer([TrainResult(0.7, "math")])
    o.trainer = CappedTrainer(stub, cap=o.target_samples)

    child = o.tree.reserve_child(o.tree.root)
    o._run_step(child, o.tree.root, PROFILE)

    assert len(_rows(stub.datasets[0])) == 10


def test_final_selection_trains_on_the_capped_data(tmp_path):
    o = bare_orchestrator(tmp_path, tier2_top_k=1, time_budget_sec=0.0, target_samples=10)
    o._run_start = time.time()
    o.final_result = None
    child = o.tree.reserve_child(o.tree.root)
    child.config = PipelineConfig(operators=[], field_flow="f", code="c", rationale="r")
    child.review_result = make_review_result(tier1=0.9)
    child.visits = 1

    o.executor = StubExecutor(tmp_path / "ds", rows=50)
    o.reviewer = StubReviewer([make_review_result(tier1=0.8)] * 3)
    stub = StubTrainer([TrainResult(0.8, "gsm8k")])
    o.trainer = CappedTrainer(stub, cap=o.target_samples)
    o._run_final_eval = lambda m: 0.55

    o._final_selection(PROFILE)

    assert len(_rows(stub.datasets[0])) == 10
