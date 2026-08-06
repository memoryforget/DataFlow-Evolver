from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from dataflow_evolver.llm.session_state import prepare_agent_session_store
from dataflow_evolver.utils.config import Config


def _result(session_id: str = "session-1"):
    return SimpleNamespace(
        session_id=session_id,
        success=True,
        error_type=None,
    )


def test_default_session_store_is_centralized_next_to_agent_root(tmp_path):
    agent_root = tmp_path / "workspace" / "_agent_root"
    node_dir = agent_root / "run_a" / "node_0"
    store = prepare_agent_session_store(
        Config({}),
        backend="claude",
        cwd=str(agent_root),
        tool_log_dir=str(node_dir),
    )

    assert store is not None
    assert store.root == (
        tmp_path / "workspace/agent_sessions/run_a/node_0/claude"
    ).resolve()
    env = store.environment()
    assert env["HOME"] == str(store.root)
    assert env["CLAUDE_CONFIG_DIR"] == str(store.root / ".claude")
    assert "XDG_DATA_HOME" not in env


def test_configured_root_retains_node_hierarchy(tmp_path):
    agent_root = tmp_path / "workspace" / "_agent_root"
    node_dir = agent_root / "run_b" / "node_2"
    configured_root = tmp_path / "custom-sessions"
    store = prepare_agent_session_store(
        Config({"session_root": str(configured_root)}),
        backend="opencode",
        cwd=str(agent_root),
        tool_log_dir=str(node_dir),
    )

    assert store is not None
    assert store.root == (configured_root / "run_b/node_2/opencode").resolve()
    env = store.environment()
    assert env["XDG_DATA_HOME"] == str(store.root / ".local/share")


def test_claude_usage_deduplicates_repeated_content_blocks(tmp_path):
    agent_root = tmp_path / "workspace" / "_agent_root"
    node_dir = agent_root / "run" / "node_0"
    store = prepare_agent_session_store(
        Config({}),
        backend="claude",
        cwd=str(agent_root),
        tool_log_dir=str(node_dir),
    )
    assert store is not None
    session_file = store.root / ".claude/projects/project/session-1.jsonl"
    session_file.parent.mkdir(parents=True)
    usage = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 30,
        "output_tokens": 40,
    }
    rows = [
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"id": "msg-1", "model": "claude", "usage": usage},
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"id": "msg-1", "model": "claude", "usage": usage},
        },
    ]
    session_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    store.finalize(phase="generate", result=_result())
    report = json.loads((node_dir / "agent_token_usage.json").read_text())
    assert report["usage_records"] == 1
    assert report["tokens"]["total_including_cache_read"] == 100
    assert report["tokens"]["billable_tokens_estimate_no_cache_read"] == 70
    assert report["session_ids"] == ["session-1"]
    combined = json.loads((node_dir / "token_usage.json").read_text())
    assert combined["agent"]["tokens"]["total_including_cache_read"] == 100
    assert combined["pipeline_llm"] is None


def test_opencode_usage_reads_isolated_database(tmp_path):
    agent_root = tmp_path / "workspace" / "_agent_root"
    node_dir = agent_root / "run" / "node_1"
    store = prepare_agent_session_store(
        Config({}),
        backend="opencode",
        cwd=str(agent_root),
        tool_log_dir=str(node_dir),
    )
    assert store is not None
    db_path = store.root / ".local/share/opencode/opencode.db"
    db_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE session (
            id TEXT, title TEXT, model TEXT, cost REAL,
            tokens_input INTEGER, tokens_output INTEGER,
            tokens_reasoning INTEGER, tokens_cache_read INTEGER,
            tokens_cache_write INTEGER, time_created INTEGER
        )
        """
    )
    connection.execute(
        "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ses-1", "pipeline", "model", 0.25, 100, 20, 5, 40, 3, 1),
    )
    connection.commit()
    connection.close()

    store.finalize(phase="generate", result=_result("ses-1"))
    report = json.loads((node_dir / "agent_token_usage.json").read_text())
    assert report["status"] == "ok"
    assert report["tokens"]["total_including_cache"] == 168
    assert report["tokens"]["billable_tokens_estimate_no_cache_read"] == 128
    assert report["cost"] == 0.25
    manifest = [
        json.loads(line)
        for line in (node_dir / "agent_sessions.jsonl").read_text().splitlines()
    ]
    assert manifest[0]["backend"] == "opencode"
    assert manifest[0]["session_id"] == "ses-1"
