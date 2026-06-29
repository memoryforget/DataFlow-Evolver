import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolve.engine.execution_wrapper import SubprocessExecutionWrapper
from dataflow_evolve.engine.trainer_wrapper import DownstreamTrainerWrapper

ROOT = Path(__file__).resolve().parents[1]
REF_SCRIPT = ROOT / "scripts" / "reference_train_eval.py"


def test_trainer_parses_reference_results(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        '{"instruction":"q1","answer":"a1"}\n'
        '{"instruction":"q2","answer":""}\n',
        encoding="utf-8",
    )
    cmd = f'{sys.executable} "{REF_SCRIPT}"'
    wrapper = DownstreamTrainerWrapper(cmd, timeout_sec=60)
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    assert result.failed is False
    assert result.metric_name == "non_empty_ratio"
    assert abs(result.score - 0.5) < 1e-9
    assert len(result.bad_cases) == 1
    assert result.bad_cases[0].error_type == "empty_answer"


def test_trainer_missing_script_degrades(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"answer":"x"}\n', encoding="utf-8")
    wrapper = DownstreamTrainerWrapper("nonexistent_cmd_xyz --flag", timeout_sec=10)
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    assert result.failed is True
    assert result.error_msg is not None


def test_execution_missing_pipeline_file(tmp_path):
    (tmp_path / "node_1").mkdir()
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolve.core.schemas import PipelineConfig

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
    from dataflow_evolve.core.schemas import PipelineConfig

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
    from dataflow_evolve.core.schemas import PipelineConfig

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
    from dataflow_evolve.core.schemas import PipelineConfig

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
    from dataflow_evolve.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is True
    assert result.dataset_path.endswith("s_step2.jsonl")
