from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from dataflow_evolver.llm.session_state import (
    prepare_agent_session_store,
    record_codex_turn_usage,
)
from dataflow_evolver.utils.config import Config


def _result(session_id: str = "session-1"):
    return SimpleNamespace(session_id=session_id, success=True, error_type=None)


def _store(tmp_path, backend: str, name: str = "iteration_001"):
    iteration_dir = tmp_path / "workspace/runs/run" / name
    store = prepare_agent_session_store(
        Config({}), backend=backend, cwd=str(tmp_path / "agent"),
        tool_log_dir=str(iteration_dir),
    )
    assert store is not None
    return iteration_dir, store


def test_claude_session_has_one_flat_config_root_inside_iteration(tmp_path):
    iteration_dir, store = _store(tmp_path, "claude")
    assert store.root == (iteration_dir / "sessions/claude").resolve()
    env = store.environment()
    assert env["HOME"] == str(store.root / "home")
    assert env["CLAUDE_CONFIG_DIR"] == str(store.root)
    assert ".claude/.claude" not in env["CLAUDE_CONFIG_DIR"]


def test_opencode_session_uses_iteration_local_xdg_data(tmp_path):
    iteration_dir, store = _store(tmp_path, "opencode")
    env = store.environment()
    assert store.root == (iteration_dir / "sessions/opencode").resolve()
    assert env["XDG_DATA_HOME"] == str(store.root / "data")


def test_claude_usage_deduplicates_repeated_content_blocks(tmp_path):
    iteration_dir, store = _store(tmp_path, "claude")
    session_file = store.root / "projects/project/session-1.jsonl"
    session_file.parent.mkdir(parents=True)
    usage = {"input_tokens": 10, "cache_creation_input_tokens": 20,
             "cache_read_input_tokens": 30, "output_tokens": 40}
    row = {"type": "assistant", "timestamp": "2026-01-01T00:00:00Z",
           "message": {"id": "msg-1", "model": "claude", "usage": usage}}
    session_file.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    store.finalize(phase="generate", result=_result())
    report = json.loads((iteration_dir / "agent_token_usage.json").read_text())
    assert report["usage_records"] == 1
    assert report["tokens"]["total_including_cache_read"] == 100


def test_opencode_usage_reads_iteration_local_database(tmp_path):
    iteration_dir, store = _store(tmp_path, "opencode", "iteration_002")
    db_path = store.root / "data/opencode/opencode.db"
    db_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE session (id TEXT, title TEXT, model TEXT, cost REAL, "
        "tokens_input INTEGER, tokens_output INTEGER, tokens_reasoning INTEGER, "
        "tokens_cache_read INTEGER, tokens_cache_write INTEGER, time_created INTEGER)"
    )
    connection.execute("INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       ("ses-1", "pipeline", "model", 0.25, 100, 20, 5, 40, 3, 1))
    connection.commit()
    connection.close()
    store.finalize(phase="generate", result=_result("ses-1"))
    report = json.loads((iteration_dir / "agent_token_usage.json").read_text())
    assert report["status"] == "ok"
    assert report["tokens"]["total_including_cache"] == 168


def test_codex_state_and_turn_usage_are_iteration_local(tmp_path):
    iteration_dir, store = _store(tmp_path, "codex", "iteration_003")
    env = store.environment()
    assert env["CODEX_HOME"] == str(store.root)
    assert store.root == (iteration_dir / "sessions/codex").resolve()

    record_codex_turn_usage(
        store.root,
        phase="generate",
        session_id="thread-1",
        usage={
            "input_tokens": 100,
            "cached_input_tokens": 25,
            "output_tokens": 30,
            "reasoning_tokens": 10,
            "total_tokens": 130,
        },
    )
    store.finalize(phase="generate", result=_result("thread-1"))
    report = json.loads((iteration_dir / "agent_token_usage.json").read_text())
    assert report["backend"] == "codex"
    assert report["usage_records"] == 1
    assert report["session_ids"] == ["thread-1"]
    assert report["tokens"]["total_tokens"] == 130
    assert report["tokens"]["billable_tokens_estimate"] == 130
