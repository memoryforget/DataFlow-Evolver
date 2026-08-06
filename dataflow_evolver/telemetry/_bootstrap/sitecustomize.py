"""Automatically installed only in generated pipeline subprocesses."""

import sys
from pathlib import Path

try:
    # The bootstrap directory itself is added to PYTHONPATH. Ensure the directory that
    # contains the dataflow_evolver package is importable even without an editable install.
    package_parent = str(Path(__file__).resolve().parents[3])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    from dataflow_evolver.telemetry.pipeline_usage import install_requests_usage_tracking

    install_requests_usage_tracking()
except Exception:
    # Telemetry is best-effort and must never prevent a pipeline from starting.
    pass
