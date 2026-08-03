"""外部数据溯源：节点用了哪些数据、走没走工具、绕过工具的直连引用能否被查出。

去污染回答"这个节点的产出有没有泄露"，溯源回答"这些数据从哪来"。后者的关键在于**不能假设
Agent 守规矩**——它的内置工具全放开，完全可以绕过 dataset-search 直接联网拉数据，所以这里的
重点是那条绕行路径能不能被标出来。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.core import provenance


def _node(tmp_path: Path, pipeline_src: str, **extra: str) -> Path:
    node = tmp_path / "node_3"
    (node / "operators").mkdir(parents=True)
    (node / "pipeline.py").write_text(pipeline_src, encoding="utf-8")
    for name, src in extra.items():
        (node / "operators" / f"{name}.py").write_text(src, encoding="utf-8")
    return node


# ---- 工具调用日志 ----------------------------------------------------------
def test_tool_uses_are_appended_across_phases(tmp_path):
    node = tmp_path / "node_1"
    provenance.record_tool_uses(node, [{"turn": 1, "name": "search_datasets"}], phase="generate")
    provenance.record_tool_uses(
        node, [{"turn": 2, "name": "download_dataset"}], phase="self_correct_review_1"
    )

    lines = (node / provenance.TOOL_LOG_NAME).read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(x) for x in lines]
    # 一个节点会经历生成 + 多轮自修正，合起来才是完整轨迹；覆写会只剩最后一轮。
    assert [r["name"] for r in records] == ["search_datasets", "download_dataset"]
    assert [r["phase"] for r in records] == ["generate", "self_correct_review_1"]


def test_tool_log_never_breaks_the_run(tmp_path):
    blocked = tmp_path / "afile"
    blocked.write_text("not a dir", encoding="utf-8")
    provenance.record_tool_uses(blocked / "node_1", [{"name": "Write"}])   # 不抛异常即可


def test_long_tool_args_are_truncated():
    out = provenance.truncate_arg({"content": "x" * 5000, "path": "a.py"})
    # Write/Edit 的 content 动辄上万字，全量落盘会让日志比数据本身还大
    assert len(out["content"]) < 1000 and "共 5000 字符" in out["content"]
    assert out["path"] == "a.py"


# ---- 静态扫描：走了工具的路径 ----------------------------------------------
def test_data_links_reference_is_attributed_to_the_node(tmp_path):
    links = tmp_path / "data_links"
    links.mkdir()
    data = links / "openai_gsm8k__main__train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    data.with_suffix(".meta.json").write_text(
        json.dumps({"dataset_id": "openai/gsm8k", "license": "mit"}), encoding="utf-8"
    )
    node = _node(tmp_path, f"first_entry_file_name = '{data}'\n")

    record = provenance.scan_node(node, data_links_dir=links)

    assert record["unmanaged"] is False
    assert len(record["data_links"]) == 1
    # data_links/ 是全局共享的，归属只能靠这条记录回答"哪个节点用了哪份数据"
    assert record["data_links"][0]["meta"]["dataset_id"] == "openai/gsm8k"
    assert record["data_links"][0]["exists"] is True


def test_missing_sidecar_is_flagged_not_silently_empty(tmp_path):
    links = tmp_path / "data_links"
    links.mkdir()
    data = links / "mystery.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    node = _node(tmp_path, f"path = '{data}'\n")

    entry = provenance.scan_node(node, data_links_dir=links)["data_links"][0]
    assert entry["meta_missing"] is True   # 显式标"没有出处"，而不是给个空 meta 看着像没问题


# ---- 静态扫描：绕过工具的直连引用 ------------------------------------------
def test_direct_load_dataset_is_flagged_unmanaged(tmp_path):
    node = _node(tmp_path, "ds = load_dataset('openai/gsm8k', 'main', split='train')\n")

    record = provenance.scan_node(node)

    # 这条路径 data_links/ 里什么都不留，只有静态扫描能查出来
    assert record["unmanaged"] is True
    assert record["direct_references"]["load_dataset"] == ["openai/gsm8k"]


def test_direct_reference_in_operator_file_is_caught(tmp_path):
    node = _node(
        tmp_path,
        "from operators.fetch import Fetch\n",
        fetch="url = 'https://raw.githubusercontent.com/x/y/data.jsonl'\n",
    )

    record = provenance.scan_node(node)

    # 算子文件和 pipeline.py 一样是 Agent 写的代码，只扫入口会漏掉大半
    assert record["unmanaged"] is True
    assert "operators/fetch.py" in record["scanned_files"]
    assert record["direct_references"]["urls"] == [
        "https://raw.githubusercontent.com/x/y/data.jsonl"
    ]


def test_hub_download_is_flagged(tmp_path):
    node = _node(tmp_path, "p = hf_hub_download(repo_id='some/corpus', filename='a.jsonl')\n")
    record = provenance.scan_node(node)
    assert record["direct_references"]["hub_download"] == ["some/corpus"]


def test_remote_storage_uri_is_flagged(tmp_path):
    node = _node(
        tmp_path,
        "storage = FileStorage(first_entry_file_name='hf:openai/gsm8k:main:train')\n"
        "alt = FileStorage(first_entry_file_name='ms:modelscope/gsm8k:train')\n",
    )

    record = provenance.scan_node(node)

    # FileStorage 的 hf:/ms: 写法比 load_dataset 更顺手（一个字符串就完事），而且是 dataflow-dev
    # 技能自己教的——最容易被走、也最容易被漏掉的一条绕行路径
    assert record["unmanaged"] is True
    assert record["direct_references"]["storage_uri"] == [
        "hf:openai/gsm8k:main:train", "ms:modelscope/gsm8k:train",
    ]


def test_local_first_entry_is_not_mistaken_for_a_remote_uri(tmp_path):
    node = _node(
        tmp_path,
        "storage = FileStorage(first_entry_file_name='./input/seed.jsonl')\n"
        "ts = '12:30'\n"
        "cfg = {'ms': 5}\n",
    )
    assert provenance.scan_node(node)["unmanaged"] is False


def test_llm_api_url_does_not_count_as_external_data(tmp_path):
    node = _node(tmp_path, "serving = APILLMServing_request(api_url='https://api.example.com/v1')\n")

    record = provenance.scan_node(node)

    # 每个节点都会调 LLM serving。若这也算外部数据引用，unmanaged 就恒为真、失去分辨力。
    assert record["unmanaged"] is False
    assert record["other_urls"] == ["https://api.example.com/v1"]


def test_scan_skips_generated_artifacts(tmp_path):
    node = _node(tmp_path, "x = 1\n")
    (node / "cache").mkdir()
    (node / "cache" / "leftover.py").write_text("load_dataset('a/b')\n", encoding="utf-8")

    record = provenance.scan_node(node)

    # cache/ 是算子中间产物，不是"节点写的代码"；扫它会把噪声当成引用
    assert record["unmanaged"] is False
    assert record["scanned_files"] == ["pipeline.py"]


def test_record_is_written_to_the_node_dir(tmp_path):
    node = _node(tmp_path, "load_dataset('a/b')\n")
    provenance.scan_node(node)

    written = json.loads((node / provenance.PROVENANCE_NAME).read_text(encoding="utf-8"))
    assert written["node"] == "node_3"
    assert written["method"] == "static_scan"   # 正则扫不到运行时拼出的 id，限度要写在记录里
    assert written["unmanaged"] is True


def test_scan_of_missing_node_dir_is_harmless(tmp_path):
    record = provenance.scan_node(tmp_path / "nope", write=False)
    assert record["scanned_files"] == [] and record["unmanaged"] is False


# ---- 接线：Agent 的工具调用落到节点目录 ------------------------------------
def test_tool_uses_of_a_failed_attempt_are_not_lost(tmp_path, monkeypatch):
    from dataflow_evolver.llm import agent_sdk

    attempts = iter([
        agent_sdk.AgentRunResult(
            False, error="429 rate limit", error_type="ProcessError",
            tool_uses=[{"turn": 1, "name": "download_dataset", "input": {"dataset_id": "a/b"}}],
        ),
        agent_sdk.AgentRunResult(
            True, tool_uses=[{"turn": 1, "name": "Write", "input": {"path": "pipeline.py"}}]
        ),
    ])
    monkeypatch.setattr(agent_sdk, "_build_options", lambda *a, **k: None)
    monkeypatch.setattr(agent_sdk.asyncio, "run", lambda coro: (coro.close(), next(attempts))[1])
    monkeypatch.setattr(agent_sdk.time, "sleep", lambda s: None)

    agent_sdk.run_pipeline_agent(
        "p", cwd=str(tmp_path), agent_cfg={"retry_base_delay": 0},
        tool_log_dir=str(tmp_path / "node_1"), phase="generate",
    )

    records = [
        json.loads(x)
        for x in (tmp_path / "node_1" / provenance.TOOL_LOG_NAME)
        .read_text(encoding="utf-8").strip().split("\n")
    ]
    # 失败那次的下载已经落了盘，重试不会把文件收回去；只记最后一次会漏掉它
    assert [r["name"] for r in records] == ["download_dataset", "Write"]
    assert records[0]["phase"] == "generate" and records[1]["phase"] == "generate#retry1"


# ---- 接线：评分收尾一定会扫 -------------------------------------------------
def test_evaluate_node_scans_even_when_the_node_fails(tmp_path, monkeypatch):
    from tests._ptb_stubs import bare_orchestrator

    o = bare_orchestrator(tmp_path)
    o.agent_root = tmp_path
    o.data_links_dir = ""
    node = _node(tmp_path, "load_dataset('leaky/benchmark')\n")

    class _Node:
        node_dir_name = "node_3"

    monkeypatch.setattr(
        type(o), "_score_node", lambda self, n, p: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        o._evaluate_node(_Node(), None)

    # "拉了不该拉的数据集导致失败"恰恰只发生在失败节点上，只在成功路径记录等于漏掉主要场景
    written = json.loads((node / provenance.PROVENANCE_NAME).read_text(encoding="utf-8"))
    assert written["direct_references"]["load_dataset"] == ["leaky/benchmark"]
