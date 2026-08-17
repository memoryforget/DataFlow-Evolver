import json
from pathlib import Path

import pytest

from dataflow_evolver.core import input_corpus
from dataflow_evolver.core.schemas import TaskSpec
from dataflow_evolver.llm.prompts import build_pipeline_prompt


PROFILE = TaskSpec("task", {"instruction": "str", "output": "str"}, [])


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("value", ["", None])
def test_input_path_is_required(value):
    with pytest.raises(ValueError, match="project.input_path"):
        input_corpus.discover(value)


def test_input_path_must_be_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        input_corpus.discover(tmp_path / "missing.jsonl")
    directory = tmp_path / "raw"
    directory.mkdir()
    with pytest.raises(FileNotFoundError, match="不是文件"):
        input_corpus.discover(directory)


def test_input_path_must_be_structured(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("notes", encoding="utf-8")
    with pytest.raises(ValueError, match="结构化文件"):
        input_corpus.discover(source)


def test_input_path_does_not_scan_sibling_files(tmp_path):
    source = _write_jsonl(tmp_path / "raw" / "entry.jsonl", [{"q": "1"}])
    _write_jsonl(tmp_path / "raw" / "other.jsonl", [{"q": "other"}])
    corpus = input_corpus.discover(source)
    assert corpus.entry_path == str(source.resolve())
    assert [item.path for item in corpus.files] == [source.resolve()]


def test_jsonl_preview_reports_rows_fields_and_truncates(tmp_path):
    source = _write_jsonl(
        tmp_path / "data.jsonl",
        [{"q": "1", "doc": "x" * 5000}, {"q": "2", "hint": "sparse"}],
    )
    entry = input_corpus.discover(source).entry_file
    assert entry.rows == 2
    assert entry.fields == ("q", "doc", "hint")
    assert "共 5000 字符" in entry.preview_lines[0]


def test_malformed_jsonl_still_yields_best_effort_preview(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"q":"ok"}\nnot-json\n', encoding="utf-8")
    entry = input_corpus.discover(path).entry_file
    assert entry.rows == 2 and entry.fields == ("q",)


def test_csv_and_json_array_preview(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("q,a\n1,2\n2,4\n", encoding="utf-8")
    csv_entry = input_corpus.discover(csv_path).entry_file
    assert csv_entry.rows == 2 and csv_entry.fields == ("q", "a")

    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps([{"q": "1"}, {"q": "2"}]), encoding="utf-8")
    json_entry = input_corpus.discover(json_path).entry_file
    assert json_entry.rows == 2 and json_entry.fields == ("q",)


def test_row_count_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(input_corpus, "ROW_COUNT_CAP", 5)
    source = _write_jsonl(tmp_path / "data.jsonl", [{"q": str(i)} for i in range(20)])
    entry = input_corpus.discover(source).entry_file
    assert entry.rows == 5 and entry.rows_capped


def test_pipeline_prompt_uses_fixed_input_path_and_forbids_external_data(tmp_path):
    source = _write_jsonl(tmp_path / "data.jsonl", [{"text": "photosynthesis"}])
    corpus = input_corpus.discover(source)
    prompt = build_pipeline_prompt(PROFILE, corpus, "run/iteration_001")
    assert str(source.resolve()) in prompt
    assert "fixed raw" in prompt or "固定语料" in prompt
    assert "不允许联网拉取外部数据" in prompt
    assert "photosynthesis" in prompt


def test_pipeline_prompt_allows_only_validated_prefix_cache_reuse(tmp_path):
    source = _write_jsonl(tmp_path / "data.jsonl", [{"text": "algebra"}])
    corpus = input_corpus.discover(source)
    parent_code = 'CACHE_PATH = "/workspace/runs/run/iteration_001/cache"'

    prompt = build_pipeline_prompt(
        PROFILE,
        corpus,
        "run/iteration_002",
        parent_code=parent_code,
    )

    assert "连续前缀中间 JSONL" in prompt
    assert "源码、初始化参数、`run()` 参数、字段流" in prompt
    assert "模型、prompt 和 serving" in prompt
    assert "父轮对应 step JSONL 存在且可解析" in prompt
    assert "从 raw input 完整执行的 fallback" in prompt
    assert "该步及其所有下游步骤都必须重算" in prompt
    assert "不要仅因为类名相同就认定算子等价" in prompt
    assert "suffix storage" in prompt
    assert "first_entry_file_name" in prompt
    assert "不能再次调用已复用的前缀算子" in prompt
    assert "DF_PREFIX_CACHE_REUSE hit|miss" in prompt
