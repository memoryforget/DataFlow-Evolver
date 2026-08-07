import json
from pathlib import Path

import pytest

from dataflow_evolver.core import input_corpus
from dataflow_evolver.core.schemas import TaskSpec
from dataflow_evolver.llm.prompts import build_pipeline_prompt

TASK = TaskSpec("task", {"instruction": "str", "output": "str"}, [])


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("value", ["", None])
def test_raw_data_directory_is_required(value):
    with pytest.raises(ValueError, match="固定 raw data"):
        input_corpus.discover(value)


def test_missing_or_empty_directory_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        input_corpus.discover(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="目录为空"):
        input_corpus.discover(empty)


def test_structured_entry_is_required(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "notes.md").write_text("notes", encoding="utf-8")
    with pytest.raises(ValueError, match="结构化文件"):
        input_corpus.discover(root)


def test_entry_priority_and_explicit_override(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "a.csv").write_text("q,a\n1,2\n", encoding="utf-8")
    jsonl = _write_jsonl(root / "z.jsonl", [{"q": "1"}])
    assert input_corpus.discover(root).entry_path == str(jsonl)
    assert input_corpus.discover(root, entry_name="a.csv").entry_path.endswith("a.csv")


def test_missing_explicit_entry_fails(tmp_path):
    _write_jsonl(tmp_path / "raw" / "a.jsonl", [{"q": "1"}])
    with pytest.raises(FileNotFoundError, match="b.jsonl"):
        input_corpus.discover(tmp_path / "raw", entry_name="b.jsonl")


def test_nested_and_hidden_files(tmp_path):
    nested = _write_jsonl(tmp_path / "raw" / "sub" / "data.jsonl", [{"q": "1"}])
    _write_jsonl(tmp_path / "raw" / ".hidden.jsonl", [{"q": "hidden"}])
    corpus = input_corpus.discover(tmp_path / "raw")
    assert corpus.entry_path == str(nested)
    assert [item.rel for item in corpus.files] == [str(Path("sub/data.jsonl"))]


def test_jsonl_preview_reports_rows_fields_and_truncates(tmp_path):
    _write_jsonl(
        tmp_path / "raw" / "data.jsonl",
        [{"q": "1", "doc": "x" * 5000}, {"q": "2", "hint": "sparse"}],
    )
    entry = input_corpus.discover(tmp_path / "raw").entry_file
    assert entry.rows == 2
    assert entry.fields == ("q", "doc", "hint")
    assert "共 5000 字符" in entry.preview_lines[0]


def test_malformed_jsonl_still_yields_best_effort_preview(tmp_path):
    path = tmp_path / "raw" / "data.jsonl"
    path.parent.mkdir()
    path.write_text('{"q":"ok"}\nnot-json\n', encoding="utf-8")
    entry = input_corpus.discover(path.parent).entry_file
    assert entry.rows == 2 and entry.fields == ("q",)


def test_csv_and_json_array_preview(tmp_path):
    csv_root = tmp_path / "csv"
    csv_root.mkdir()
    (csv_root / "data.csv").write_text("q,a\n1,2\n2,4\n", encoding="utf-8")
    csv_entry = input_corpus.discover(csv_root).entry_file
    assert csv_entry.rows == 2 and csv_entry.fields == ("q", "a")

    json_root = tmp_path / "json"
    json_root.mkdir()
    (json_root / "data.json").write_text(json.dumps([{"q": "1"}, {"q": "2"}]), encoding="utf-8")
    json_entry = input_corpus.discover(json_root).entry_file
    assert json_entry.rows == 2 and json_entry.fields == ("q",)


def test_row_count_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(input_corpus, "ROW_COUNT_CAP", 5)
    _write_jsonl(tmp_path / "raw" / "data.jsonl", [{"q": str(i)} for i in range(20)])
    entry = input_corpus.discover(tmp_path / "raw").entry_file
    assert entry.rows == 5 and entry.rows_capped


def test_pipeline_prompt_uses_fixed_corpus_and_forbids_external_data(tmp_path):
    source = _write_jsonl(tmp_path / "raw" / "data.jsonl", [{"text": "photosynthesis"}])
    corpus = input_corpus.discover(source.parent)
    prompt = build_pipeline_prompt(TASK, corpus, str(tmp_path / "iteration_001"))
    assert str(source) in prompt
    assert "fixed raw" in prompt or "固定语料" in prompt
    assert "不允许联网拉取外部数据" in prompt
    assert "photosynthesis" in prompt
