"""Execution environment and final-artifact selection contracts."""
import json
import os
import textwrap
from pathlib import Path

from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper


def _cfg() -> PipelineConfig:
    return PipelineConfig(operators=[], rationale="t", field_flow="", code="")


def _contract(body: str, *, compile_body: str = "return None") -> str:
    return (
        "import os\n"
        "class Pipeline:\n"
        "    def compile(self):\n"
        f"{textwrap.indent(compile_body, '        ')}\n"
        "    def forward(self):\n"
        f"{textwrap.indent(body, '        ')}\n"
        "pipeline = Pipeline()\n"
        "pipeline.compile()\n"
        "if os.environ.get('DF_COMPILE_ONLY'):\n"
        "    raise SystemExit(0)\n"
        "pipeline.forward()\n"
    )


def _make_iteration(path: Path, body: str, *, raw: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pipeline.py").write_text(body if raw else _contract(body), encoding="utf-8")
    return path


def test_pipeline_can_import_package_from_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    pkg = workspace / "vendored_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 'vendored-ok'\n", encoding="utf-8")
    iteration = _make_iteration(workspace / "runs/run/iteration_001", (
        "import json, pathlib\nfrom vendored_pkg import VALUE\n"
        "p = pathlib.Path(__file__).parent / 'out_step1.jsonl'\n"
        "p.write_text(json.dumps({'v': VALUE}) + '\\n')\n"
    ))
    result = SubprocessExecutionWrapper(str(workspace), timeout_sec=120).run(_cfg(), iteration)
    assert result.success, result.traceback
    assert json.loads(Path(result.dataset_path).read_text())["v"] == "vendored-ok"


def test_pythonpath_preserves_inherited_entries(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("PYTHONPATH", "/inherited/one")
    iteration = _make_iteration(tmp_path / "iteration_001", (
        "import json, os, pathlib\n"
        "p = pathlib.Path(__file__).parent / 'out_step1.jsonl'\n"
        "p.write_text(json.dumps({'pp': os.environ['PYTHONPATH']}) + '\\n')\n"
    ))
    result = SubprocessExecutionWrapper(str(workspace), timeout_sec=120).run(_cfg(), iteration)
    parts = json.loads(Path(result.dataset_path).read_text())["pp"].split(os.pathsep)
    assert parts[0] == str(workspace.resolve())
    assert "/inherited/one" in parts


def test_empty_final_step_does_not_fall_back_to_intermediate_output(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    iteration = _make_iteration(tmp_path / "iteration_001", (
        "import json, pathlib\nroot = pathlib.Path(__file__).parent\n"
        "(root / 'cache_step1.jsonl').write_text(json.dumps({'x': 1}) + '\\n')\n"
        "(root / 'cache_step2.jsonl').write_text('')\n"
    ))
    result = SubprocessExecutionWrapper(str(workspace), timeout_sec=120).run(_cfg(), iteration)
    assert not result.success
    diagnostics = json.loads(result.diagnostics)
    assert [row["rows"] for row in diagnostics] == [1, 0]
    assert diagnostics[0]["fields"] == ["x"]


def test_session_jsonl_is_never_selected_as_candidate_dataset(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    iteration = _make_iteration(tmp_path / "iteration_001", (
        "import pathlib\nroot = pathlib.Path(__file__).parent\n"
        "events = root / 'sessions/pipeline/pipeline_llm_calls.jsonl'\n"
        "events.parent.mkdir(parents=True, exist_ok=True)\n"
        "events.write_text('{\"usage\": 1}\\n')\n"
    ))
    result = SubprocessExecutionWrapper(str(workspace), timeout_sec=120).run(_cfg(), iteration)
    assert not result.success
    assert result.diagnostics == "no fresh JSONL artifacts"


def test_artifact_validation_rejects_missing_compile_contract(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    iteration = _make_iteration(
        tmp_path / "iteration_001",
        "raise RuntimeError('must never run')\n",
        raw=True,
    )
    result = SubprocessExecutionWrapper(str(workspace)).run(_cfg(), iteration)
    assert not result.success
    assert result.failure_stage == "artifact_validation"
    assert "missing pipeline.compile()" in (result.traceback or "")


def test_compile_failure_never_enters_forward(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    iteration = tmp_path / "iteration_001"
    iteration.mkdir()
    marker = iteration / "forward_was_called"
    (iteration / "pipeline.py").write_text(
        _contract(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('bad')\n",
            compile_body="raise KeyError('missing input field')",
        ),
        encoding="utf-8",
    )
    result = SubprocessExecutionWrapper(str(workspace)).run(_cfg(), iteration)
    assert not result.success
    assert result.failure_stage == "compile"
    assert "KeyError" in (result.traceback or "")
    assert not marker.exists()


def test_inherited_compile_only_flag_cannot_skip_formal_execution(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("DF_COMPILE_ONLY", "1")
    iteration = _make_iteration(
        tmp_path / "iteration_001",
        "import pathlib\n"
        "p = pathlib.Path(__file__).parent / 'result_step1.jsonl'\n"
        "p.write_text('{\"instruction\":\"q\",\"output\":\"a\"}\\n')\n",
    )
    result = SubprocessExecutionWrapper(str(workspace)).run(_cfg(), iteration)
    assert result.success, result.traceback
