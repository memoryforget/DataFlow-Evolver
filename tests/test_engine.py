from pathlib import Path
import textwrap

from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper


CFG = PipelineConfig([], "", "", "")
STEP_CFG = PipelineConfig(
    [{"name": "FirstStep"}, {"name": "SecondStep"}], "", "", ""
)


def _pipeline(iteration_dir: Path, code: str) -> None:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    wrapped = (
        "import os\n"
        "class Pipeline:\n"
        "    def compile(self):\n"
        "        return None\n"
        "    def forward(self):\n"
        f"{textwrap.indent(code, '        ')}\n"
        "pipeline = Pipeline()\n"
        "pipeline.compile()\n"
        "if os.environ.get('DF_COMPILE_ONLY'):\n"
        "    raise SystemExit(0)\n"
        "pipeline.forward()\n"
    )
    (iteration_dir / "pipeline.py").write_text(wrapped, encoding="utf-8")


def _executor(tmp_path) -> SubprocessExecutionWrapper:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    raw_entry = workspace / "raw.jsonl"
    raw_entry.write_text('{"instruction":"raw"}\n', encoding="utf-8")
    return SubprocessExecutionWrapper(
        str(workspace), timeout_sec=30, raw_entry_path=str(raw_entry)
    )


def test_missing_pipeline_file(tmp_path):
    iteration = tmp_path / "iteration_001"
    iteration.mkdir()
    result = _executor(tmp_path).run(CFG, iteration)
    assert not result.success
    assert result.failure_stage == "artifact_validation"


def test_runs_pipeline_and_finds_jsonl(tmp_path):
    iteration = tmp_path / "iteration_001"
    _pipeline(iteration, (
        "import pathlib\np=pathlib.Path(__file__).parent/'result.jsonl'\n"
        "p.write_text('{\"instruction\":\"q\",\"output\":\"a\"}\\n', encoding='utf-8')\n"
    ))
    result = _executor(tmp_path).run(CFG, iteration)
    assert result.success and result.dataset_path.endswith("result.jsonl")


def test_captures_pipeline_crash(tmp_path):
    iteration = tmp_path / "iteration_001"
    _pipeline(iteration, "raise ValueError('boom')\n")
    result = _executor(tmp_path).run(CFG, iteration)
    assert not result.success and "ValueError" in (result.traceback or "")
    assert result.failure_stage == "execution"


def test_prefers_highest_step_number_not_latest_mtime(tmp_path):
    iteration = tmp_path / "iteration_001"
    _pipeline(iteration, (
        "import pathlib,time\nc=pathlib.Path(__file__).parent/'cache'; c.mkdir()\n"
        "(c/'x_step2.jsonl').write_text('{\"q\":\"final\"}\\n')\n"
        "time.sleep(0.05)\n(c/'x_step1.jsonl').write_text('{\"q\":\"middle\"}\\n')\n"
    ))
    result = _executor(tmp_path).run(STEP_CFG, iteration)
    assert result.success and result.dataset_path.endswith("x_step2.jsonl")


def test_empty_highest_step_is_failure(tmp_path):
    iteration = tmp_path / "iteration_001"
    _pipeline(iteration, (
        "import pathlib\nc=pathlib.Path(__file__).parent/'cache'; c.mkdir()\n"
        "(c/'x_step1.jsonl').write_text('{\"q\":\"ok\"}\\n')\n"
        "(c/'x_step2.jsonl').write_text('\\n')\n"
    ))
    result = _executor(tmp_path).run(STEP_CFG, iteration)
    assert not result.success
    assert '"rows": 0' in result.diagnostics
