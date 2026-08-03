import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper
from dataflow_evolver.engine.trainer_wrapper import DownstreamTrainerWrapper

ROOT = Path(__file__).resolve().parents[1]
REF_SCRIPT = ROOT / "scripts" / "reference_train_eval.py"


def _dev_set(tmp_path: Path, rows: list[str]) -> Path:
    dev = tmp_path / "dev.jsonl"
    dev.write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    return dev


def test_trainer_parses_reference_results(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        '{"instruction":"q1","answer":"a1"}\n'
        '{"instruction":"q2","answer":"a2"}\n',
        encoding="utf-8",
    )
    # dev 题集里 q1 被训练数据覆盖，q9 没有 → 1/2
    dev = _dev_set(tmp_path, [
        '{"instruction":"q1","answer":"a1"}',
        '{"instruction":"q9","answer":"a9"}',
    ])

    cmd = f'{sys.executable} "{REF_SCRIPT}"'
    wrapper = DownstreamTrainerWrapper(
        cmd, timeout_sec=60, extra_args=["--dev-set", str(dev)]
    )
    result = wrapper.run(str(ds), str(tmp_path / "out"))

    assert result.failed is False
    assert result.metric_name == "exact_match"
    assert abs(result.score - 0.5) < 1e-9
    assert len(result.bad_cases) == 1
    assert result.bad_cases[0].error_type == "no_answer"


def test_reference_script_refuses_to_run_without_a_question_set(tmp_path):
    """拿不到题集必须非零退出，而不是回退到某份"默认题库"。

    参考实现同时是契约文档：这里若允许兜底，读它的人就会照抄一份能静默评错题集的脚本，
    而框架的 dev 口径（core/dev_set.py 物化 + manifest）也就只剩君子协定。
    """
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"instruction":"q1","answer":"a1"}\n', encoding="utf-8")

    wrapper = DownstreamTrainerWrapper(f'{sys.executable} "{REF_SCRIPT}"', timeout_sec=60)
    result = wrapper.run(str(ds), str(tmp_path / "out"))

    assert result.failed is True
    assert not (tmp_path / "out" / "results.json").exists()


def test_trainer_forwards_extra_args(tmp_path, monkeypatch):
    """--dev-set 之类由编排器决定的参数必须原样传到脚本，且排在 --dataset/--output 之后。"""
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"answer":"x"}\n', encoding="utf-8")
    wrapper = DownstreamTrainerWrapper(
        "nonexistent_cmd_xyz", timeout_sec=10, extra_args=["--dev-set", "/tmp/dev.jsonl"]
    )
    wrapper.run(str(ds), str(tmp_path / "out"))

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cmd"][-2:] == ["--dev-set", "/tmp/dev.jsonl"]
    assert manifest["cmd"].index("--dataset") < manifest["cmd"].index("--dev-set")


def test_trainer_writes_run_manifest(tmp_path, monkeypatch):
    """训练配方散在命令行+环境变量里，跑到一半被改动会让节点分数不可比——必须留痕。"""
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"instruction":"q1","answer":"a1"}\n', encoding="utf-8")
    monkeypatch.setenv("SFT_NUM_EPOCHS", "15")
    monkeypatch.setenv("UNRELATED_VAR", "noise")
    dev = _dev_set(tmp_path, ['{"instruction":"q1","answer":"a1"}'])

    cmd = f'{sys.executable} "{REF_SCRIPT}"'
    DownstreamTrainerWrapper(
        cmd, timeout_sec=60, extra_args=["--dev-set", str(dev)]
    ).run(str(ds), str(tmp_path / "out"))

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["env"]["SFT_NUM_EPOCHS"] == "15"
    assert "UNRELATED_VAR" not in manifest["env"]        # 只留会改变分数的量
    assert manifest["dataset"] == str(ds.resolve())
    # 同一条命令下脚本被改动也不再等价，故记内容摘要而非只记路径
    assert str(REF_SCRIPT) in manifest["scripts"]


def test_trainer_missing_script_degrades(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"answer":"x"}\n', encoding="utf-8")
    wrapper = DownstreamTrainerWrapper("nonexistent_cmd_xyz --flag", timeout_sec=10)
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    assert result.failed is True
    assert result.error_msg is not None


def test_trainer_times_out_and_degrades(tmp_path):
    """训练超时 → 及时杀掉进程树并降级 failed，不抛出、不干等满。

    卡住的训练不能拖垮主循环：MCTS 还有别的节点要跑，而 TrainResult(failed=True)
    足以让这个节点按系统级失败处理。
    """
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"instruction":"q","answer":"a"}\n', encoding="utf-8")
    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(30)\n", encoding="utf-8")

    wrapper = DownstreamTrainerWrapper(f'{sys.executable} "{slow}"', timeout_sec=2)
    t0 = time.time()
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    elapsed = time.time() - t0

    assert result.failed is True
    assert result.error_msg is not None
    assert elapsed < 20  # 超时即返回，而不是等满 30s


def test_execution_missing_pipeline_file(tmp_path):
    (tmp_path / "node_1").mkdir()
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolver.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is False
    assert "不存在" in (result.traceback or "")


def test_execution_runs_simple_pipeline(tmp_path):
    node = tmp_path / "node_1"
    node.mkdir()
    # 一个最小"流水线"：写出一个 jsonl 到 node 目录
    (node / "pipeline.py").write_text(
        "import json, pathlib\n"
        "out = pathlib.Path(__file__).parent / 'result.jsonl'\n"
        "out.write_text(json.dumps({'instruction':'a','answer':'b'}) + '\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolver.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is True
    assert result.dataset_path is not None
    assert result.dataset_path.endswith("result.jsonl")


def test_execution_captures_crash(tmp_path):
    node = tmp_path / "node_1"
    node.mkdir()
    (node / "pipeline.py").write_text("raise ValueError('boom')\n", encoding="utf-8")
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolver.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is False
    assert "ValueError" in (result.traceback or "") or "boom" in (result.traceback or "")


def test_execution_prefers_nonempty_when_last_step_empty(tmp_path):
    node = tmp_path / "node_1"
    (node / "cache").mkdir(parents=True)
    # 模拟 DataFlow 多 step 落盘：step1 有数据，step2（更新）为空
    (node / "pipeline.py").write_text(
        "import pathlib, time\n"
        "c = pathlib.Path(__file__).parent / 'cache'\n"
        "(c / 's_step1.jsonl').write_text('{\\\"q\\\":\\\"a\\\"}\\n', encoding='utf-8')\n"
        "time.sleep(0.05)\n"
        "(c / 's_step2.jsonl').write_text('\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolver.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is True
    assert result.dataset_path.endswith("s_step1.jsonl")


def test_execution_picks_max_step_over_mtime(tmp_path):
    """两步都非空时，应按 step 号取最大（最终算子），而非按 mtime（review M2）。

    故意让 step1 比 step2 写得更晚（mtime 更新），证明选择不依赖 mtime。
    """
    node = tmp_path / "node_1"
    (node / "cache").mkdir(parents=True)
    (node / "pipeline.py").write_text(
        "import pathlib, time\n"
        "c = pathlib.Path(__file__).parent / 'cache'\n"
        "(c / 's_step2.jsonl').write_text('{\\\"q\\\":\\\"final\\\"}\\n', encoding='utf-8')\n"
        "time.sleep(0.05)\n"
        "(c / 's_step1.jsonl').write_text('{\\\"q\\\":\\\"mid\\\"}\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolver.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is True
    assert result.dataset_path.endswith("s_step2.jsonl")

