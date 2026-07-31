"""SubprocessExecutionWrapper 的执行环境契约。

重点回归：node_dir/pipeline.py 必须能直接 `import dataflow`（vendored 包在 agent_root 下）。
`python <node_dir>/pipeline.py` 的 sys.path[0] 是脚本目录而非 cwd，故必须靠 PYTHONPATH 兜住。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.core.schemas import PipelineConfig  # noqa: E402
from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper  # noqa: E402


def _cfg() -> PipelineConfig:
    return PipelineConfig(operators=[], rationale="t", field_flow="", code="")


def _make_node(agent_root: Path, name: str, body: str) -> None:
    node = agent_root / name
    node.mkdir(parents=True, exist_ok=True)
    (node / "pipeline.py").write_text(body, encoding="utf-8")


def test_pipeline_can_import_package_vendored_at_agent_root(tmp_path):
    """agent_root 下的包对 node_dir 里的脚本可直接 import（不需脚本自己补 sys.path）。"""
    agent_root = tmp_path / "_agent_root"
    pkg = agent_root / "vendored_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 'vendored-ok'\n", encoding="utf-8")

    _make_node(agent_root, "node_1", (
        "import json, pathlib\n"
        "from vendored_pkg import VALUE\n"
        "p = pathlib.Path(__file__).parent / 'out_step1.jsonl'\n"
        "p.write_text(json.dumps({'v': VALUE}) + '\\n')\n"
    ))

    ex = SubprocessExecutionWrapper(agent_root=str(agent_root), timeout_sec=120)
    result = ex.run(_cfg(), "node_1")

    assert result.success, result.traceback
    assert json.loads(Path(result.dataset_path).read_text(encoding="utf-8"))["v"] == "vendored-ok"


def test_pythonpath_preserves_inherited_entries_and_puts_agent_root_first(tmp_path, monkeypatch):
    """agent_root 置于 PYTHONPATH 首位，同时不丢外部继承的条目。"""
    agent_root = tmp_path / "_agent_root"
    agent_root.mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", "/inherited/one")

    _make_node(agent_root, "node_1", (
        "import json, os, pathlib\n"
        "p = pathlib.Path(__file__).parent / 'out_step1.jsonl'\n"
        "p.write_text(json.dumps({'pp': os.environ['PYTHONPATH']}) + '\\n')\n"
    ))

    ex = SubprocessExecutionWrapper(agent_root=str(agent_root), timeout_sec=120)
    result = ex.run(_cfg(), "node_1")

    assert result.success, result.traceback
    parts = json.loads(Path(result.dataset_path).read_text(encoding="utf-8"))["pp"].split(os.pathsep)
    assert parts[0] == str(agent_root.resolve())
    assert "/inherited/one" in parts


def test_env_overrides_win_over_defaults(tmp_path):
    """env_overrides（如 DF_TARGET_SAMPLES）仍然生效，不被 PYTHONPATH 处理干扰。"""
    agent_root = tmp_path / "_agent_root"
    agent_root.mkdir(parents=True)
    _make_node(agent_root, "node_1", (
        "import json, os, pathlib\n"
        "p = pathlib.Path(__file__).parent / 'out_step1.jsonl'\n"
        "p.write_text(json.dumps({'n': os.environ.get('DF_TARGET_SAMPLES')}) + '\\n')\n"
    ))

    ex = SubprocessExecutionWrapper(agent_root=str(agent_root), timeout_sec=120)
    result = ex.run(_cfg(), "node_1", env_overrides={"DF_TARGET_SAMPLES": "4000"})

    assert result.success, result.traceback
    assert json.loads(Path(result.dataset_path).read_text(encoding="utf-8"))["n"] == "4000"
