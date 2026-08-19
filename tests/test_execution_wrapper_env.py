from __future__ import annotations

import inspect

from dataflow_evolver.engine.execution_wrapper import (
    PROJECT_ROOT,
    SubprocessExecutionWrapper,
)


def test_execution_uses_project_adapter_not_vendored_dataflow():
    source = inspect.getsource(SubprocessExecutionWrapper.run)
    assert "str(PROJECT_ROOT)" in source
    assert "Path(item).resolve() == self.workspace_dir" in source
    assert "cwd=str(iteration_dir)" in source
    assert PROJECT_ROOT.name == "DataFlow-Evolver"
