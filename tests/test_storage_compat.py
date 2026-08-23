from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from dataflow_evolver.compat.storage import dataframe_records_with_python_nulls
from dataflow_evolver.telemetry.pipeline_usage import BOOTSTRAP_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataframe_records_restore_python_none_without_changing_values():
    dataframe = pd.DataFrame(
        [
            {"teacher_output": None, "output": "original", "score": 1.0},
            {"teacher_output": "rewritten", "output": "original", "score": None},
        ]
    )

    records = dataframe_records_with_python_nulls(dataframe)

    assert records == [
        {"teacher_output": None, "output": "original", "score": 1.0},
        {"teacher_output": "rewritten", "output": "original", "score": None},
    ]
    assert math.isnan(dataframe.iloc[1]["score"])


def test_pipeline_bootstrap_normalizes_file_storage_dict_reads(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "teacher_output": None,
                        "final_output": "original solution",
                    }
                ),
                json.dumps(
                    {
                        "teacher_output": "teacher solution",
                        "final_output": "original solution",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    code = "\n".join(
        [
            "import json",
            "from dataflow.utils.storage import FileStorage",
            f"path = {str(input_path)!r}",
            "storage = FileStorage(first_entry_file_name=path)",
            "rows = storage.step().read('dict')",
            "chosen = [row.get('teacher_output') or row['final_output'] for row in rows]",
            "print(json.dumps({'teacher_output': rows[0]['teacher_output'], 'chosen': chosen}))",
        ]
    )
    env = dict(os.environ)
    inherited = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(BOOTSTRAP_DIR), *inherited]
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    assert payload["teacher_output"] is None
    assert payload["chosen"] == ["original solution", "teacher solution"]
