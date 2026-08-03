"""dev 集物化单测：口径固定、幂等复用、换口径报错、与旧抽样逻辑等价。"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from dataflow_evolver.core import checkpoint, dev_set


def _write_source(path, n=100):
    path.write_text(
        "".join(
            json.dumps({"problem": f"q{i}", "answer": str(i)}, ensure_ascii=False) + "\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )
    return path


def _read(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def test_materialize_writes_fixed_subset(tmp_path):
    src = _write_source(tmp_path / "src.jsonl", n=100)
    ds = dev_set.materialize(str(src), tmp_path / "eval", size=10, seed=42)

    rows = _read(tmp_path / "eval" / dev_set.DEV_NAME)
    assert len(rows) == 10
    assert ds.count == 10
    assert ds.size == 10
    assert ds.seed == 42
    # manifest 与实际内容一致，供事后审计
    manifest = json.loads((tmp_path / "eval" / dev_set.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["count"] == 10
    assert manifest["source_sha"] == ds.source_sha


def test_same_seed_same_subset(tmp_path):
    """同口径两次物化到不同目录必须给出同一批题——否则"所有节点同题集"无从谈起。"""
    src = _write_source(tmp_path / "src.jsonl", n=100)
    a = dev_set.materialize(str(src), tmp_path / "a", size=10, seed=7)
    b = dev_set.materialize(str(src), tmp_path / "b", size=10, seed=7)
    assert _read(a.path) == _read(b.path)


def test_different_seed_different_subset(tmp_path):
    src = _write_source(tmp_path / "src.jsonl", n=100)
    a = dev_set.materialize(str(src), tmp_path / "a", size=10, seed=1)
    b = dev_set.materialize(str(src), tmp_path / "b", size=10, seed=2)
    assert _read(a.path) != _read(b.path)


def test_reuse_is_idempotent(tmp_path):
    """续跑必须拿到同一批题，且不重写文件。"""
    src = _write_source(tmp_path / "src.jsonl", n=100)
    first = dev_set.materialize(str(src), tmp_path / "eval", size=10, seed=42)
    mtime = (tmp_path / "eval" / dev_set.DEV_NAME).stat().st_mtime_ns

    second = dev_set.materialize(str(src), tmp_path / "eval", size=10, seed=42)
    assert second == first
    assert (tmp_path / "eval" / dev_set.DEV_NAME).stat().st_mtime_ns == mtime


@pytest.mark.parametrize("changed", [{"size": 20}, {"seed": 43}])
def test_recipe_change_raises(tmp_path, changed):
    """换口径不能静默改写：已跑节点的分数出自旧题集，混在一起比就是错的。"""
    src = _write_source(tmp_path / "src.jsonl", n=100)
    dev_set.materialize(str(src), tmp_path / "eval", size=10, seed=42)
    kwargs = {"size": 10, "seed": 42, **changed}
    with pytest.raises(dev_set.DevSetChanged):
        dev_set.materialize(str(src), tmp_path / "eval", **kwargs)


def test_source_content_change_raises(tmp_path):
    """源文件被换掉（路径不变）同样是换口径。"""
    src = _write_source(tmp_path / "src.jsonl", n=100)
    dev_set.materialize(str(src), tmp_path / "eval", size=10, seed=42)
    _write_source(src, n=120)
    with pytest.raises(dev_set.DevSetChanged):
        dev_set.materialize(str(src), tmp_path / "eval", size=10, seed=42)


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dev_set.materialize(str(tmp_path / "nope.jsonl"), tmp_path / "eval", size=10)


def test_size_zero_keeps_everything(tmp_path):
    src = _write_source(tmp_path / "src.jsonl", n=30)
    ds = dev_set.materialize(str(src), tmp_path / "eval", size=0)
    assert ds.count == 30


def test_size_larger_than_source_keeps_everything(tmp_path):
    src = _write_source(tmp_path / "src.jsonl", n=5)
    ds = dev_set.materialize(str(src), tmp_path / "eval", size=500)
    assert ds.count == 5


def test_subset_matches_legacy_sampling(tmp_path):
    """与评测脚本里原先的 `random.Random(seed).sample(rows, k)` 选出同一批题。

    钉住这条等价性，物化后的 dev 分数才与历史运行可比；否则这次改动会悄悄让所有旧分数作废。
    """
    src = _write_source(tmp_path / "src.jsonl", n=7500)
    ds = dev_set.materialize(str(src), tmp_path / "eval", size=500, seed=42)

    rows = _read(src)
    legacy = random.Random(42).sample(rows, 500)
    assert {r["problem"] for r in _read(ds.path)} == {r["problem"] for r in legacy}


def test_build_returns_none_without_source(tmp_path):
    """未配 source = 不物化（不下传 --dev-set）。

    注意这不等于「评测脚本自己挑题」：本仓库的评测脚本拿不到题集会直接拒绝跑
    （见 test_math_sft_eval.py）。物化与否是框架的事，怎么应对是脚本契约的事。
    """
    assert dev_set.build({"size": 10}, tmp_path) is None
    assert dev_set.build(None, tmp_path) is None


def test_materialized_dev_set_is_scannable_for_leakage(tmp_path):
    """物化的意义之一：dev 题集能被去污染扫到。

    dev 是搜索直接优化的目标，把 dev 题蒸馏进合成数据是最省力的刷分路径。题集只存在于评测
    子进程内存里时，这个洞根本无从检测——落盘之后才谈得上扫描。
    """
    from dataflow_evolver.core import decontamination

    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"problem": "Let f be a continuous function on the closed unit "
                               "interval such that its integral vanishes identically",
                    "answer": "0"}) + "\n",
        encoding="utf-8",
    )
    ds = dev_set.materialize(str(src), tmp_path / "eval", size=0)

    index = decontamination.build(reference_files=[], n=8)
    assert not index.enabled                       # 只配 test 参考集时 dev 泄露完全看不见
    index.add_file(ds.path)
    assert index.enabled

    leaked = tmp_path / "synth.jsonl"
    leaked.write_text(
        json.dumps({"instruction": "Let f be a continuous function on the closed unit "
                                   "interval such that its integral vanishes identically",
                    "output": "the answer is 0"}) + "\n",
        encoding="utf-8",
    )
    assert index.scan(leaked).rate == 1.0


# ---- checkpoint 口径核对 ----------------------------------------------------
def test_verify_dev_set_passes_when_identical():
    recipe = {"source_sha": "abc", "seed": 42, "size": 500}
    checkpoint.verify_dev_set({"dev_set": recipe}, dict(recipe))


def test_verify_dev_set_raises_on_mismatch():
    """workspace/eval 被清掉时 materialize 会心安理得重建，只有 checkpoint 能发现口径变了。"""
    with pytest.raises(ValueError, match="dev 集口径"):
        checkpoint.verify_dev_set(
            {"dev_set": {"source_sha": "abc", "seed": 42, "size": 500}},
            {"source_sha": "abc", "seed": 42, "size": 200},
        )


def test_verify_dev_set_tolerates_legacy_checkpoint():
    """旧 checkpoint 没记录口径，只告警不阻断。"""
    checkpoint.verify_dev_set({}, {"source_sha": "abc", "seed": 42, "size": 500})
    checkpoint.verify_dev_set({}, None)
