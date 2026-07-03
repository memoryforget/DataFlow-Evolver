"""M7 接线测试：强基线作 root（warm start）+ Relative Gain 奖励 + penalty 随模式切换 + 池门槛。

覆盖：
- RewardPolicy：绝对分 / Relative Gain 奖励、penalty 随模式切换、relative 缺 baseline 报错。
- adapt_baseline_source：api_url / model_name / first_entry_file_name / cache_path 替换 + key 注入。
- TreeManager.set_baseline_root：root 持有基线 config、visits=1、reward=0、记录 baseline_dev。
- DataPool 门槛：冷启动分位数(中位数)门槛、暖启动严格 > baseline、去重、占比裁剪 water-filling、空池。
- Orchestrator 暖/冷两条路径（桩替换执行/训练/SDK），验证奖励与 penalty 实际取值。

不依赖 ANTHROPIC_API_KEY、不真正调用 API。
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolve.core.baseline import (
    REWARD_ABSOLUTE,
    REWARD_RELATIVE,
    RewardPolicy,
    adapt_baseline_source,
)
from dataflow_evolve.core.data_pool import DataPool, PoolConfig
from dataflow_evolve.core.schemas import (
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TaskProfile,
    TrainResult,
)
from dataflow_evolve.core.tree_manager import TreeManager
from dataflow_evolve.engine.execution_wrapper import ExecutionResult
from dataflow_evolve.main import Orchestrator
from dataflow_evolve.utils.config import Config


# ---------------------------------------------------------------------------
# RewardPolicy
# ---------------------------------------------------------------------------
def test_reward_absolute_is_node_score():
    rp = RewardPolicy(reward_mode=REWARD_ABSOLUTE)
    assert rp.reward(0.7) == 0.7
    assert rp.penalty == 0.0
    assert rp.is_relative is False


def test_reward_relative_is_gain_over_baseline():
    rp = RewardPolicy(reward_mode=REWARD_RELATIVE, baseline_dev_score=0.6)
    assert abs(rp.reward(0.8) - 0.2) < 1e-9     # 打败基线 → 正
    assert abs(rp.reward(0.6) - 0.0) < 1e-9     # 持平 → 0
    assert abs(rp.reward(0.4) + 0.2) < 1e-9     # 逊于基线 → 负
    assert rp.is_relative is True


def test_penalty_switches_with_mode_keeps_failure_worst():
    """relative 模式下 penalty=−baseline_dev（映射到绝对 0），保证失败恒为最差。"""
    rp = RewardPolicy(reward_mode=REWARD_RELATIVE, baseline_dev_score=0.6)
    assert abs(rp.penalty + 0.6) < 1e-9
    # 一个「能跑但逊于基线」的节点（dev=0.4 → reward=−0.2）仍优于失败 penalty（−0.6）
    assert rp.reward(0.4) > rp.penalty
    # 一个崩溃节点取 penalty（−0.6），映射回绝对分恰为 0
    assert abs((rp.penalty + rp.baseline_dev_score) - 0.0) < 1e-9


def test_relative_mode_requires_baseline_score():
    with pytest.raises(ValueError):
        RewardPolicy(reward_mode=REWARD_RELATIVE)
    with pytest.raises(ValueError):
        RewardPolicy(reward_mode="bogus")


# ---------------------------------------------------------------------------
# adapt_baseline_source
# ---------------------------------------------------------------------------
BASELINE_SNIPPET = '''
class P:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/empty.jsonl",
            cache_path="./cache",
        )
        self.llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )
'''


def test_adapt_baseline_source_rewrites_constants_and_injects_key():
    out = adapt_baseline_source(
        BASELINE_SNIPPET,
        api_url="http://proxy/v1/chat/completions",
        model_name="gpt-5.5",
        key_env="DF_PIPELINE_API_KEY",
        empty_jsonl_path="C:/tmp/node_3/empty.jsonl",
        cache_path="C:/tmp/node_3/cache",
    )
    assert "https://api.openai.com" not in out
    assert 'api_url="http://proxy/v1/chat/completions"' in out
    assert 'model_name="gpt-5.5"' in out
    assert 'gpt-4o' not in out
    assert 'key_name_of_api_key="DF_PIPELINE_API_KEY"' in out
    # 路径被改写为本地空种子与节点缓存
    assert 'first_entry_file_name="C:/tmp/node_3/empty.jsonl"' in out
    assert 'cache_path="C:/tmp/node_3/cache"' in out
    assert "example_data" not in out


def test_adapt_baseline_source_replaces_existing_key_name():
    src = (
        'APILLMServing_request(api_url="x", key_name_of_api_key="DF_API_KEY", model_name="gpt-4o")\n'
        'storage = FileStorage(first_entry_file_name="old.jsonl", cache_path="./cache")'
    )
    out = adapt_baseline_source(
        src, api_url="u", model_name="m", key_env="DF_PIPELINE_API_KEY",
        empty_jsonl_path="e", cache_path="c",
    )
    assert 'key_name_of_api_key="DF_PIPELINE_API_KEY"' in out
    assert "DF_API_KEY" not in out


# ---------------------------------------------------------------------------
# TreeManager.set_baseline_root
# ---------------------------------------------------------------------------
def _cfg(name: str = "op") -> PipelineConfig:
    return PipelineConfig(operators=[{"name": name, "params": {}}],
                          field_flow="a -> b", code="# code", rationale="r")


def test_set_baseline_root_holds_config_visits1_reward0():
    tm = TreeManager()
    assert tm.root.config is None        # 起始为虚拟空根
    tm.set_baseline_root(_cfg("baseline"), baseline_dev_score=0.55)
    assert tm.root.config is not None
    assert tm.root.visits == 1
    assert abs(tm.root.total_reward) < 1e-9            # 基线相对自身 Relative Gain = 0
    assert abs(tm.root.baseline_dev_score - 0.55) < 1e-9
    assert len(tm.global_memory) == 1                  # 基线进入历史记忆


# ---------------------------------------------------------------------------
# DataPool 门槛（运行期 stage → finalize 统一筛/去重/裁剪）
# ---------------------------------------------------------------------------
def _rows(node_tag, n):
    return [{"instruction": f"{node_tag}-{i}", "output": "x"} for i in range(n)]


def test_pool_cold_quantile_gate_admits_upper_half():
    """冷启动：门槛 = 本轮 dev 分中位数（>=）。"""
    pool = DataPool(PoolConfig(enabled=True, quantile=0.5,
                               max_pool_fraction_per_pipeline=0))  # 关掉占比上限单测门槛
    pool.stage(1, _rows("a", 1), dev_score=0.3, review_passed=True)
    pool.stage(2, _rows("b", 1), dev_score=0.5, review_passed=True)
    pool.stage(3, _rows("c", 1), dev_score=0.7, review_passed=True)
    pool.finalize()
    sources = set(e.source_node_id for e in pool._entries)
    # 中位数=0.5 → 收 node2(0.5,=) 与 node3(0.7)，剔除 node1(0.3)
    assert sources == {2, 3}
    assert pool.last_stats.threshold == 0.5


def test_pool_warm_gate_strict_greater_than_baseline():
    """暖启动：门槛 = baseline+margin，严格 >（等号的 baseline 自身也被排除）。"""
    pool = DataPool(PoolConfig(enabled=True, max_pool_fraction_per_pipeline=0))
    pool.set_warm_gate(baseline_dev=0.6, margin=0.0)
    pool.stage(1, _rows("a", 1), dev_score=0.65, review_passed=True)  # > 0.6 → 收
    pool.stage(2, _rows("b", 1), dev_score=0.60, review_passed=True)  # == 0.6 → 拒（严格）
    pool.stage(3, _rows("c", 1), dev_score=0.55, review_passed=True)  # < 0.6 → 拒
    pool.finalize()
    assert set(e.source_node_id for e in pool._entries) == {1}


def test_pool_skips_failed_and_unreviewed_nodes():
    pool = DataPool(PoolConfig(enabled=True, max_pool_fraction_per_pipeline=0))
    assert pool.stage(1, _rows("a", 1), dev_score=None, review_passed=True) is False
    assert pool.stage(2, _rows("b", 1), dev_score=0.9, review_passed=False) is False
    assert pool.stage(3, _rows("c", 1), dev_score=0.9, review_passed=True) is True


def test_pool_dedup_across_sources_first_come():
    pool = DataPool(PoolConfig(enabled=True, quantile=0.0, max_pool_fraction_per_pipeline=0))
    dup = [{"instruction": "same", "output": "x"}]
    pool.stage(1, list(dup), dev_score=0.9, review_passed=True)
    pool.stage(2, list(dup), dev_score=0.9, review_passed=True)  # 同内容 → 去重
    pool.finalize()
    assert len(pool) == 1
    assert pool.last_stats.rejected_dedup == 1


def test_pool_fraction_cap_inactive_below_threshold_sources():
    """来源数 < ⌈1/0.3⌉=4 时占比上限不生效（单一来源必占 100%，裁剪无意义）。"""
    pool = DataPool(PoolConfig(enabled=True, quantile=0.0, max_pool_fraction_per_pipeline=0.3))
    pool.stage(1, _rows("a", 100), dev_score=0.9, review_passed=True)
    pool.finalize()
    assert len(pool) == 100          # 不被裁
    assert pool.last_stats.rejected_cap == 0


def test_pool_fraction_cap_water_fills_when_feasible():
    """≥4 个来源时，单一高产来源被裁到 ≤30% 池总量。"""
    pool = DataPool(PoolConfig(enabled=True, quantile=0.0, max_pool_fraction_per_pipeline=0.3))
    pool.stage(1, _rows("a", 100), dev_score=0.9, review_passed=True)  # 高产
    for nid in (2, 3, 4):
        pool.stage(nid, _rows(chr(96 + nid), 10), dev_score=0.9, review_passed=True)
    pool.finalize()
    total = len(pool)
    by_src = {}
    for e in pool._entries:
        by_src[e.source_node_id] = by_src.get(e.source_node_id, 0) + 1

    # 关键断言：pool 不应为空（防止水填充螺旋至零）
    assert total > 0, "Pool should not be empty after water-filling"
    assert total >= 30, f"Pool total {total} too small for input of 130 rows"

    # 每个来源占比 ≤ 30%（含浮点容差）
    for nid, c in by_src.items():
        assert c <= math.ceil(0.3 * total) + 1, f"src {nid} 占 {c}/{total} 超 30%"

    # 高产来源确被裁，小来源应保留全部
    assert by_src[1] < 100, "Dominant source should be capped"
    for nid in (2, 3, 4):
        assert by_src[nid] == 10, f"Small source {nid} should keep all 10 rows, got {by_src[nid]}"
    assert pool.last_stats.rejected_cap > 0


def test_pool_empty_when_nothing_passes():
    pool = DataPool(PoolConfig(enabled=True))
    pool.set_warm_gate(baseline_dev=1.0, margin=0.0)
    pool.stage(1, _rows("a", 5), dev_score=1.0, review_passed=True)  # 等号，严格 > 拒
    pool.finalize()
    assert len(pool) == 0           # 无人打败基线 → 池空（不报错）


def test_pool_disabled_stage_is_noop():
    pool = DataPool(PoolConfig(enabled=False))
    assert pool.stage(1, _rows("a", 5), dev_score=0.9, review_passed=True) is False
    pool.finalize()
    assert len(pool) == 0


# ---------------------------------------------------------------------------
# Orchestrator 暖/冷路径
# ---------------------------------------------------------------------------
FAKE_PIPELINE = (
    "import json, pathlib\n"
    "out = pathlib.Path(__file__).parent / 'gen.jsonl'\n"
    "rows = [{'instruction': f'q{i}', 'answer': f'a{i}'} for i in range(5)]\n"
    "out.write_text('\\n'.join(json.dumps(r) for r in rows) + '\\n', encoding='utf-8')\n"
)

REAL_BASELINE = (
    Path(__file__).resolve().parents[1]
    / "workspace/_agent_root/dataflow/statics/pipelines/api_pipelines/text_sft_synthesis_pipeline.py"
)


def _make_config(tmp_path: Path, *, baseline_enabled: bool, data_pool: bool = False,
                 checkpoint: str | None = None, max_iterations: int = 2) -> Config:
    agent_root = tmp_path / "_agent_root"
    agent_root.mkdir(parents=True, exist_ok=True)
    seed = tmp_path / "seed.jsonl"
    seed.write_text('{"text": "hello"}\n', encoding="utf-8")
    ref_script = Path(__file__).resolve().parents[1] / "scripts" / "reference_train_eval.py"
    raw = {
        "project": {
            "workspace_dir": str(tmp_path / "workspace"),
            "agent_root": str(agent_root),
            "seed_data": str(seed),
        },
        "llm": {"backend": "api", "api": {
            "api_url": "https://example.invalid/v1/chat/completions",
            "model_name": "stub", "api_key": "stub-key", "max_workers": 2,
        }},
        "agent_sdk": {"pipeline_filename": "pipeline.py", "permission_mode": "bypassPermissions",
                      "allowed_tools": ["Read", "Write"], "setting_sources": ["project"]},
        "baseline": {
            "enabled": baseline_enabled,
            "margin": 0.0,
            "seed": 42,
            "baseline_pipeline": str(REAL_BASELINE),
        },
        "search": {"exploration_constant": 1.414, "max_iterations": max_iterations,
                   "max_children_per_node": 2, "self_correction_retries": 1,
                   "checkpoint": checkpoint or ""},
        "review": {"schema_threshold": 0.6, "relevance_threshold": 0.6, "sample_size": 5},
        "trainer": {"dev_eval_cmd": f'{sys.executable} "{ref_script}"', "timeout_sec": 60, "train_timeout_sec": 60},
        "diagnostic": {"max_bad_cases": 10},
    }
    if data_pool:
        raw["data_pool"] = {"enabled": True, "score_threshold": 0.0,
                            "dedup": "content_hash",
                            "output_path": str(tmp_path / "pool.jsonl")}
    return Config(raw)


def _common_stubs(orch, monkeypatch, train_scores):
    """桩：profiler / reviewer / diagnostician / pipeline_agent / executor / trainer。

    train_scores：trainer.run 依次返回的 dev 分（首个为基线分，其后为各节点分）。
    返回记录回传奖励的列表 backprop_log（(node_id, reward)）。
    """
    profile = TaskProfile("生成问答", {"instruction": "str", "answer": "str"}, "qa", ["答案非空"], "qa-sft")
    monkeypatch.setattr(orch.profiler, "run", lambda *a, **k: profile)
    monkeypatch.setattr(orch.reviewer, "review",
                        lambda dataset_path, prof: ReviewResult(0.9, 0.9, True, []))
    monkeypatch.setattr(orch.diagnostician, "analyze",
                        lambda bad, downstream="": DiagnosticReport(summary="ok"))

    def fake_generate(profile, node_dir_name, diagnostic=None, memory_summary=None,
                      review_issues=None, parent_code=None):
        node_dir = orch.agent_root / node_dir_name
        node_dir.mkdir(parents=True, exist_ok=True)
        return PipelineConfig(operators=[{"name": "Gen", "params": {}}],
                              field_flow="text -> qa", code=FAKE_PIPELINE, rationale="stub")
    monkeypatch.setattr(orch.pipeline_agent, "generate", fake_generate)

    counter = {"n": 0}

    def fake_exec_run(config, node_dir):
        node_path = orch.agent_root / node_dir
        node_path.mkdir(parents=True, exist_ok=True)
        ds = node_path / "gen.jsonl"
        counter["n"] += 1
        ds.write_text(f'{{"instruction":"q{counter["n"]}","answer":"a"}}\n', encoding="utf-8")
        return ExecutionResult(True, dataset_path=str(ds))
    monkeypatch.setattr(orch.executor, "run", fake_exec_run)

    scores = list(train_scores)

    def fake_train_run(dataset_path, output_dir):
        s = scores.pop(0) if scores else 0.0
        return TrainResult(score=s, metric_name="acc", bad_cases=[], raw_log_path="", failed=False)
    monkeypatch.setattr(orch.trainer, "run", fake_train_run)

    backprop_log = []
    orig_bp = orch.tree.backpropagate

    def spy_bp(node, reward):
        backprop_log.append((node.id, reward))
        orig_bp(node, reward)
    monkeypatch.setattr(orch.tree, "backpropagate", spy_bp)
    return backprop_log


def test_warm_start_baseline_root_and_relative_reward(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, baseline_enabled=True)
    orch = Orchestrator(cfg)
    # 基线 dev=0.5；两个节点 dev=0.8、0.3 → Relative Gain = +0.3、−0.2
    backprop_log = _common_stubs(orch, monkeypatch, train_scores=[0.5, 0.8, 0.3])

    best = orch.run("生成问答数据集")

    # 奖励策略切换到 Relative Gain，基线分记录正确
    assert orch.reward_policy.is_relative is True
    assert abs(orch.reward_policy.baseline_dev_score - 0.5) < 1e-9
    # 基线作 root：root 持有 config 且记录基线分
    assert orch.tree.root.config is not None
    assert abs(orch.tree.root.baseline_dev_score - 0.5) < 1e-9
    # 节点回传的是相对增益（基线 root 自身不在 backprop_log 中）
    node_rewards = sorted(r for _, r in backprop_log)
    assert any(abs(r - 0.3) < 1e-9 for r in node_rewards)
    assert any(abs(r + 0.2) < 1e-9 for r in node_rewards)
    assert best is not None


def test_warm_start_failed_node_gets_relative_penalty(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, baseline_enabled=True)
    orch = Orchestrator(cfg)
    backprop_log = _common_stubs(orch, monkeypatch, train_scores=[0.5])
    # 第一次 trainer 调用（基线）成功得 0.5；其后所有节点训练系统级失败 → penalty = −0.5
    calls = {"n": 0}

    def trainer_run(dataset_path, output_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            return TrainResult(0.5, "acc", [], "", False, None)
        return TrainResult(0.0, "acc", [], "", True, "boom")
    monkeypatch.setattr(orch.trainer, "run", trainer_run)

    orch.run("生成问答数据集")
    assert orch.reward_policy.is_relative
    # 失败节点的回传奖励应为 −0.5（不是 0）
    assert any(abs(r + 0.5) < 1e-9 for _, r in backprop_log)
    assert all(abs(r - 0.0) > 1e-12 for _, r in backprop_log)  # 没有任何 0（否则失败节点会与崩溃节点同分）


def test_cold_start_keeps_absolute_reward_and_virtual_root(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, baseline_enabled=False)
    orch = Orchestrator(cfg)
    backprop_log = _common_stubs(orch, monkeypatch, train_scores=[0.8, 0.3])

    best = orch.run("生成问答数据集")

    # 冷启动：绝对分奖励、penalty=0、虚拟空根
    assert orch.reward_policy.reward_mode == REWARD_ABSOLUTE
    assert orch.reward_policy.penalty == 0.0
    assert orch.tree.root.config is None
    # 回传的是绝对分本身（非相对）
    node_rewards = sorted(r for _, r in backprop_log)
    assert any(abs(r - 0.8) < 1e-9 for r in node_rewards)
    assert any(abs(r - 0.3) < 1e-9 for r in node_rewards)
    assert best is not None


def test_warm_start_pool_admits_above_baseline(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, baseline_enabled=True, data_pool=True)
    orch = Orchestrator(cfg)
    assert orch.pool is not None
    # 基线 0.5；节点 0.8（准入）、0.3（拒，逊于基线）
    _common_stubs(orch, monkeypatch, train_scores=[0.5, 0.8, 0.3])

    orch.run("生成问答数据集")
    # 暖启动池门槛 = baseline_dev + margin = 0.5，严格 >：0.8 收、0.3 拒
    assert orch.pool.gate_mode == "relative_baseline"
    assert abs(orch.pool.last_stats.threshold - 0.5) < 1e-9
    # 仅 0.8 的节点数据被收割（1 行）
    assert len(orch.pool) == 1


def test_checkpoint_resume_continues_without_redoing(tmp_path, monkeypatch):
    """续跑：run1 完成 1 轮落 checkpoint；run2 用同一 checkpoint 续到第 2 轮，不重做第 1 轮。"""
    ckpt = str(tmp_path / "run.ckpt.json")

    # run1：max_iterations=1 → 评估 1 个节点，落 checkpoint
    cfg1 = _make_config(tmp_path / "a", baseline_enabled=False, checkpoint=ckpt, max_iterations=1)
    orch1 = Orchestrator(cfg1)
    _common_stubs(orch1, monkeypatch, train_scores=[0.7])
    orch1.run("任务")
    nodes1 = orch1.tree.all_nodes()
    assert len(nodes1) == 1
    node1_id = nodes1[0].id
    assert Path(ckpt).exists()

    # run2：新 Orchestrator + 同 checkpoint + max_iterations=2 → 续跑 1 轮
    cfg2 = _make_config(tmp_path / "b", baseline_enabled=False, checkpoint=ckpt, max_iterations=2)
    orch2 = Orchestrator(cfg2)
    _common_stubs(orch2, monkeypatch, train_scores=[0.4])
    # 续跑应从 checkpoint 恢复 profile —— profiler 不应再被调用
    def boom(*a, **k):
        raise AssertionError("续跑不应调用 profiler")
    monkeypatch.setattr(orch2.profiler, "run", boom)

    orch2.run("任务")
    ids = sorted(n.id for n in orch2.tree.all_nodes())
    assert node1_id in ids          # run1 的节点被恢复（未重做）
    assert len(ids) == 2            # 仅新增 1 个节点
