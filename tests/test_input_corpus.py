"""input/ 契约：范式由目录内容决定、入口选择确定可控、预览尽力而为且不会拖垮启动。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.core import input_corpus
from dataflow_evolver.core.schemas import TaskProfile
from dataflow_evolver.llm.prompts import build_pipeline_prompt

PROFILE = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")


def _empty_entry(tmp_path: Path) -> Path:
    p = tmp_path / "_agent_root" / "empty.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


# ---- 范式判定 --------------------------------------------------------------
def test_missing_dir_is_from_scratch(tmp_path):
    c = input_corpus.discover(tmp_path / "nope", _empty_entry(tmp_path))
    assert c.grounded is False
    assert c.entry_path.endswith("empty.jsonl")
    assert c.prompt_block() == ""
    assert c.sample_text() is None


def test_unset_dir_is_from_scratch(tmp_path):
    assert input_corpus.discover("", _empty_entry(tmp_path)).grounded is False


def test_empty_dir_is_from_scratch(tmp_path):
    (tmp_path / "input").mkdir()
    assert input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path)).grounded is False


def test_corpus_makes_it_grounded(tmp_path):
    src = _write_jsonl(tmp_path / "input" / "seed.jsonl", [{"q": "1+1", "a": "2"}])
    c = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path))
    assert c.grounded is True
    assert c.entry_path == str(src)


def test_only_unreadable_formats_falls_back_to_scratch(tmp_path):
    """.md 不能作 DataFlow 的 first_entry：宁可退回从零合成，也不要给一个跑不起来的入口。"""
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "notes.md").write_text("# 领域笔记\n", encoding="utf-8")
    assert input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path)).grounded is False


# ---- 入口选择 --------------------------------------------------------------
def test_entry_priority_prefers_jsonl(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.csv").write_text("q,a\n1,2\n", encoding="utf-8")
    jsonl = _write_jsonl(tmp_path / "input" / "z.jsonl", [{"q": "1"}])
    c = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path))
    assert c.entry_path == str(jsonl)      # 字典序 z>a，但格式优先级更高
    assert len(c.files) == 2               # 另一个文件仍被列出供算子自行读取


def test_explicit_entry_wins(tmp_path):
    _write_jsonl(tmp_path / "input" / "a.jsonl", [{"q": "1"}])
    b = _write_jsonl(tmp_path / "input" / "b.jsonl", [{"q": "2"}])
    c = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path), entry_name="b.jsonl")
    assert c.entry_path == str(b)


def test_explicit_entry_missing_fails_fast(tmp_path):
    """误配的入口必须启动即报错——不能默默换一个文件让人跑完才发现用错了语料。"""
    _write_jsonl(tmp_path / "input" / "a.jsonl", [{"q": "1"}])
    with pytest.raises(FileNotFoundError, match="b.jsonl"):
        input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path), entry_name="b.jsonl")


def test_nested_files_are_discovered(tmp_path):
    nested = _write_jsonl(tmp_path / "input" / "sub" / "d.jsonl", [{"q": "1"}])
    c = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path))
    assert c.entry_path == str(nested)
    assert c.files[0].rel == str(Path("sub/d.jsonl"))


def test_hidden_files_ignored(tmp_path):
    (tmp_path / "input").mkdir()
    _write_jsonl(tmp_path / "input" / ".cache.jsonl", [{"q": "1"}])
    assert input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path)).grounded is False


# ---- 预览 ------------------------------------------------------------------
def test_preview_reports_rows_and_union_of_fields(tmp_path):
    _write_jsonl(
        tmp_path / "input" / "seed.jsonl",
        [{"q": "1"}, {"q": "2", "hint": "稀疏字段"}, {"q": "3"}, {"q": "4"}],
    )
    entry = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path)).entry_file
    assert entry.rows == 4
    assert entry.fields == ("q", "hint")            # 并集：稀疏字段也要被看见
    assert len(entry.preview_lines) == input_corpus.PREVIEW_ROWS


def test_preview_truncates_long_values(tmp_path):
    _write_jsonl(tmp_path / "input" / "seed.jsonl", [{"doc": "x" * 5000, "id": 1}])
    c = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path))
    line = c.entry_file.preview_lines[0]
    assert len(line) < 1000
    assert "共 5000 字符" in line
    assert '"id": 1' in line                        # 截断的是长字段，不是整行


def test_malformed_jsonl_still_counts_rows(tmp_path):
    """语料脏一点很正常，预览解析失败只该少一段上下文，不该让整轮搜索起不来。"""
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "seed.jsonl").write_text(
        '{"q": "ok"}\nnot json at all\n', encoding="utf-8"
    )
    entry = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path)).entry_file
    assert entry.rows == 2
    assert entry.fields == ("q",)


def test_csv_preview(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "seed.csv").write_text("q,a\n1+1,2\n2+2,4\n", encoding="utf-8")
    entry = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path)).entry_file
    assert entry.rows == 2 and entry.fields == ("q", "a")


def test_json_array_preview(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "seed.json").write_text(
        json.dumps([{"q": "1"}, {"q": "2"}]), encoding="utf-8"
    )
    entry = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path)).entry_file
    assert entry.rows == 2 and entry.fields == ("q",)


def test_row_count_caps_on_huge_file(tmp_path, monkeypatch):
    monkeypatch.setattr(input_corpus, "ROW_COUNT_CAP", 5)
    _write_jsonl(tmp_path / "input" / "seed.jsonl", [{"q": str(i)} for i in range(50)])
    entry = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path)).entry_file
    assert entry.rows == 5 and entry.rows_capped is True
    assert "≥5 行" in entry.describe()


def test_sample_text_feeds_profiler(tmp_path):
    _write_jsonl(tmp_path / "input" / "seed.jsonl", [{"q": "1+1", "a": "2"}])
    c = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path))
    assert "1+1" in c.sample_text()


# ---- 注入 prompt -----------------------------------------------------------
def test_grounded_prompt_points_pipeline_at_the_corpus(tmp_path):
    src = _write_jsonl(tmp_path / "input" / "seed.jsonl", [{"passage": "光合作用……"}])
    (tmp_path / "input" / "notes.md").write_text("# 背景\n", encoding="utf-8")
    c = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path))

    prompt = build_pipeline_prompt(profile=PROFILE, corpus=c, node_dir="node_0")
    assert str(src) in prompt
    assert "corpus-grounded" in prompt
    assert "from-scratch" not in prompt
    assert "光合作用" in prompt                 # 预览进了 prompt
    assert "notes.md" in prompt                # 其余素材也列出
    assert "自行读取" in prompt


def test_scratch_prompt_keeps_from_scratch_rule(tmp_path):
    c = input_corpus.discover("", _empty_entry(tmp_path))
    prompt = build_pipeline_prompt(profile=PROFILE, corpus=c, node_dir="node_0")
    assert "from-scratch" in prompt
    assert "corpus-grounded" not in prompt
    assert "【输入语料】" not in prompt
    # 外部取数没开时，"不依赖任何输入"是准确的描述
    assert "流水线不依赖任何输入" in prompt


def test_scratch_with_external_tools_offers_fetching_real_data(tmp_path):
    """input/ 为空 + 外部取数开着 → 范式规则必须把"去取真实数据"摆成同等选项。

    这里曾经是个真实的功能缺口：范式规则是**编号硬要求**且写死"流水线不依赖任何输入、
    `first_entry_file_name` 必须指向空 jsonl、凭模型知识合成"，而外部数据说明排在编号要求
    之后、自称"（可选）"。两者直接冲突时 Agent 只会听前者，取数工具挂了也等于没挂。
    """
    c = input_corpus.discover("", _empty_entry(tmp_path))
    prompt = build_pipeline_prompt(
        profile=PROFILE, corpus=c, node_dir="node_0", data_links_dir="/ws/data_links"
    )

    # 不能再断言"不依赖任何输入"——那正是把取数堵死的那句
    assert "流水线不依赖任何输入" not in prompt
    assert "凭模型知识生成」不是唯一选项" in prompt
    # 三条路都要摆出来，并说明取真实数据时入口指向下载路径
    assert "取真实数据打底" in prompt
    assert "凭模型知识合成" in prompt
    assert "混合" in prompt
    assert "download_dataset" in prompt
    # 选哪条是设计决策，要留在 decision.json 里可供事后追溯
    assert "decision.json 的 reason" in prompt


def test_grounded_with_external_tools_keeps_the_corpus_authoritative(tmp_path):
    """有 input/ 语料时外部数据只能补充：否则 Agent 可能整个绕开用户给的语料。"""
    _write_jsonl(tmp_path / "input" / "seed.jsonl", [{"passage": "光合作用……"}])
    c = input_corpus.discover(tmp_path / "input", _empty_entry(tmp_path))
    prompt = build_pipeline_prompt(
        profile=PROFILE, corpus=c, node_dir="node_0", data_links_dir="/ws/data_links"
    )

    assert "corpus-grounded" in prompt
    assert "补充不等于替换" in prompt
    assert "不要用外部数据把它顶掉" in prompt
