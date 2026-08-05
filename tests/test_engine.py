from pathlib import Path

from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper


CFG = PipelineConfig([], "", "", "")


def _pipeline(node: Path, code: str) -> None:
    node.mkdir(parents=True, exist_ok=True)
    (node / "pipeline.py").write_text(code, encoding="utf-8")


def test_missing_pipeline_file(tmp_path):
    (tmp_path / "node").mkdir()
    result = SubprocessExecutionWrapper(str(tmp_path), timeout_sec=30).run(CFG, "node")
    assert not result.success and "不存在" in (result.traceback or "")


def test_runs_pipeline_and_finds_jsonl(tmp_path):
    _pipeline(
        tmp_path / "node",
        "import pathlib\n"
        "p=pathlib.Path(__file__).parent/'result.jsonl'\n"
        "p.write_text('{\"instruction\":\"q\",\"output\":\"a\"}\\n', encoding='utf-8')\n",
    )
    result = SubprocessExecutionWrapper(str(tmp_path), timeout_sec=30).run(CFG, "node")
    assert result.success and result.dataset_path.endswith("result.jsonl")


def test_captures_pipeline_crash(tmp_path):
    _pipeline(tmp_path / "node", "raise ValueError('boom')\n")
    result = SubprocessExecutionWrapper(str(tmp_path), timeout_sec=30).run(CFG, "node")
    assert not result.success and "ValueError" in (result.traceback or "")


def test_prefers_latest_nonempty_step_number(tmp_path):
    _pipeline(
        tmp_path / "node",
        "import pathlib,time\n"
        "c=pathlib.Path(__file__).parent/'cache'; c.mkdir()\n"
        "(c/'x_step2.jsonl').write_text('{\"q\":\"final\"}\\n')\n"
        "time.sleep(0.05)\n"
        "(c/'x_step1.jsonl').write_text('{\"q\":\"middle\"}\\n')\n",
    )
    result = SubprocessExecutionWrapper(str(tmp_path), timeout_sec=30).run(CFG, "node")
    assert result.success and result.dataset_path.endswith("x_step2.jsonl")


def test_falls_back_when_last_step_is_empty(tmp_path):
    _pipeline(
        tmp_path / "node",
        "import pathlib\n"
        "c=pathlib.Path(__file__).parent/'cache'; c.mkdir()\n"
        "(c/'x_step1.jsonl').write_text('{\"q\":\"ok\"}\\n')\n"
        "(c/'x_step2.jsonl').write_text('\\n')\n",
    )
    result = SubprocessExecutionWrapper(str(tmp_path), timeout_sec=30).run(CFG, "node")
    assert result.success and result.dataset_path.endswith("x_step1.jsonl")
