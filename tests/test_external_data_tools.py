"""外部数据集搜索工具：落盘纪律、复用、可溯源、失败可读，以及 prompt 注入开关。

不打真实网络：`datasets.load_dataset` 与 Hub 元信息调用统一打桩，只验证我们自己那层契约。
"""
import asyncio
import inspect
import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.schemas import TaskProfile
from dataflow_evolver.llm import mcp_datasets
from dataflow_evolver.llm.prompts import build_pipeline_prompt

PROFILE = TaskProfile("t", {"instruction": "str", "answer": "str"}, [], "sft")
SCRATCH = InputCorpus(entry_path="empty.jsonl")

ROWS = [{"question": "1+1", "answer": "2"}, {"question": "2+2", "answer": "4"}]


def _call(tool, **args) -> str:
    result = asyncio.run(tool.handler({k: str(v) for k, v in args.items()}))
    return result["content"][0]["text"], result.get("is_error", False)


def _fake_datasets(monkeypatch, rows_factory):
    """替换 `datasets` 模块，让 `_stream_to_file` 走我们自己的落盘逻辑而不打网络。

    在 sys.modules 上替换而非直接 patch 函数：`from datasets import load_dataset` 是写在
    函数体里的惰性导入，只有换掉模块才拦得住。
    """
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda ds, cfg=None, split=None, streaming=False: rows_factory()
    monkeypatch.setitem(sys.modules, "datasets", fake)


def _fake_modelscope(monkeypatch, rows_factory):
    """伪造 modelscope SDK。本环境没装它，而我们要验的是自己那层落盘契约，不是 SDK 行为。"""
    pkg = types.ModuleType("modelscope")
    sub = types.ModuleType("modelscope.msdatasets")

    class HubApi:
        def get_repo(self, repo_id, repo_type="dataset"):
            assert repo_type == "dataset"
            return types.SimpleNamespace(
                repo_id=repo_id,
                downloads=123,
                license="Apache-2.0",
                gated=False,
                tasks=["text-generation"],
                tags=["math"],
                likes=7,
                file_size=4096,
                last_modified="2026-01-01",
                description="grade school math",
            )

        def get_dataset_files(self, dataset_id):
            return [
                {"Path": "README.md", "Size": 406, "IsLFS": False},
                {"Path": "dataset_infos.json", "Size": 1768, "IsLFS": False},
            ]

    class MsDataset:
        @staticmethod
        def load(ds, subset_name=None, split=None, use_streaming=False):
            return rows_factory()

    pkg.HubApi = HubApi
    sub.MsDataset = MsDataset
    pkg.msdatasets = sub
    monkeypatch.setitem(sys.modules, "modelscope", pkg)
    monkeypatch.setitem(sys.modules, "modelscope.msdatasets", sub)


@pytest.fixture
def links(tmp_path, monkeypatch):
    d = mcp_datasets.configure(tmp_path / "data_links")
    monkeypatch.setattr(mcp_datasets, "_stream_rows", lambda hub, ds, cfg, split, n: ROWS[:n])
    _fake_datasets(monkeypatch, lambda: iter(ROWS))
    # license 查询走 Hub，测试里不打真实网络；缓存直接预置成"已知"。
    monkeypatch.setattr(mcp_datasets, "_license_cache", {"openai/gsm8k": "mit"})
    return d


# ---- 下载纪律 --------------------------------------------------------------
def test_download_writes_jsonl_into_shared_dir(links):
    text, is_error = _call(mcp_datasets.download_dataset, dataset_id="openai/gsm8k", split="train")
    assert not is_error
    payload = json.loads(text)

    path = Path(payload["path"])
    assert path.parent == links                       # 统一落共享目录，跨节点复用同一份
    assert path.is_absolute()                         # 流水线在子进程里跑，相对路径会失效
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert [json.loads(x) for x in lines] == ROWS


def test_download_writes_provenance_sidecar(links):
    text, _ = _call(mcp_datasets.download_dataset, dataset_id="openai/gsm8k", config="main", split="train")
    meta_path = Path(json.loads(text)["path"]).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["dataset_id"] == "openai/gsm8k"       # 数据从哪来，事后从文件名是猜不出来的
    assert meta["config"] == "main" and meta["split"] == "train"
    assert meta["rows"] == len(ROWS)
    assert meta["fields"] == ["answer", "question"]
    assert meta["downloaded_at"]
    # 许可要落盘：Agent 在 inspect 时看过一眼，但那只存在于它的上下文里，
    # 而论文交代数据许可、事后审计"这份数据能不能用"看的是磁盘上的边车。
    assert meta["license"] == "mit"
    assert meta["source"] == "huggingface"
    assert meta["url"].endswith("openai/gsm8k")
    # 完整性要能从记录里读出来：截断与否直接决定这份数据能不能和别的节点比
    assert meta["complete"] is True and meta["truncated_by"] is None
    assert meta["bytes"] > 0


def test_license_lookup_failure_is_recorded_as_unknown(links, monkeypatch):
    monkeypatch.setattr(mcp_datasets, "_license_cache", {})

    def _raise():
        raise ConnectionError("offline")

    monkeypatch.setattr(mcp_datasets, "_hf_api", _raise)
    text, _ = _call(mcp_datasets.download_dataset, dataset_id="a/b")
    meta = json.loads(Path(json.loads(text)["path"]).with_suffix(".meta.json").read_text())

    # 取不到就显式记 None，不要让缺失看起来像"没有许可要求"；也不能因此让下载整个失败
    assert meta["license"] is None


def test_second_download_reuses_the_same_file(links, monkeypatch):
    first, _ = _call(mcp_datasets.download_dataset, dataset_id="openai/gsm8k", split="train")

    def _boom(*a, **k):
        raise AssertionError("已下载过的数据不应再拉一次")

    monkeypatch.setattr(mcp_datasets, "_stream_to_file", _boom)
    second, _ = _call(mcp_datasets.download_dataset, dataset_id="openai/gsm8k", split="train")

    assert json.loads(second)["reused"] is True
    assert json.loads(second)["path"] == json.loads(first)["path"]
    assert json.loads(second)["dataset_id"] == "openai/gsm8k"   # 复用时也带出溯源信息
    assert "完整 split" in json.loads(second)["note"]


# ---- 全量下载与两道上限 ----------------------------------------------------
def test_whole_split_is_downloaded_by_default(links, monkeypatch):
    _fake_datasets(monkeypatch, lambda: iter([{"i": i} for i in range(500)]))

    text, _ = _call(mcp_datasets.download_dataset, dataset_id="a/b")
    payload = json.loads(text)

    # 取多少、怎么采样是流水线该做的决策（也是被搜索的对象）；工具替它先做选择等于砍掉搜索空间
    assert payload["rows"] == 500
    assert payload["complete"] is True


def test_size_cap_truncates_and_says_so(links, monkeypatch):
    _fake_datasets(monkeypatch, lambda: iter([{"text": "x" * 1000} for _ in range(1000)]))
    monkeypatch.setattr(mcp_datasets, "_max_download_bytes", 10_000)

    text, _ = _call(mcp_datasets.download_dataset, dataset_id="a/b")
    payload = json.loads(text)

    assert 0 < payload["rows"] < 1000
    assert payload["bytes"] <= 10_000            # 上限用字节：行大小跨数据集能差几个数量级
    assert payload["complete"] is False and payload["truncated_by"] == "size_cap"
    # 截断留下的是 split 头部，不是随机样本——有序数据集据此判断风格会被带偏
    assert "不是随机采样" in payload["truncation_note"]


def test_explicit_max_rows_is_honoured(links, monkeypatch):
    _fake_datasets(monkeypatch, lambda: iter([{"i": i} for i in range(500)]))

    payload = json.loads(_call(mcp_datasets.download_dataset, dataset_id="a/b", max_rows=10)[0])
    assert payload["rows"] == 10
    assert payload["complete"] is False and payload["truncated_by"] == "max_rows"


def test_max_rows_zero_means_whole_split(links, monkeypatch):
    _fake_datasets(monkeypatch, lambda: iter([{"i": i} for i in range(50)]))

    payload = json.loads(_call(mcp_datasets.download_dataset, dataset_id="a/b", max_rows=0)[0])
    assert payload["rows"] == 50 and payload["complete"] is True


# ---- 原子落盘：中断不留可复用的半份数据 ------------------------------------
def test_interrupted_download_leaves_nothing_reusable(links, monkeypatch):
    def _dies_midway():
        yield {"i": 0}
        raise ConnectionError("connection reset")

    _fake_datasets(monkeypatch, _dies_midway)
    text, is_error = _call(mcp_datasets.download_dataset, dataset_id="a/b")

    assert is_error
    # 若直接写目标名，下个节点的 exists() 照样命中，把半份数据当完整的用——不报错也查不出来
    assert list(links.glob("*.jsonl")) == []
    assert list(links.glob("*.part")) == []


def test_reuse_warns_when_the_cached_file_is_incomplete(links, monkeypatch):
    _fake_datasets(monkeypatch, lambda: iter([{"i": i} for i in range(50)]))
    _call(mcp_datasets.download_dataset, dataset_id="a/b", max_rows=10)

    payload = json.loads(_call(mcp_datasets.download_dataset, dataset_id="a/b")[0])

    # 「加大外部数据量」是个自然的变异方向；静默返回旧的截断文件会让搜索得出错误结论
    assert payload["reused"] is True
    assert "不完整" in payload["note"] and "10 行" in payload["note"]


# ---- ModelScope：第二个 hub，同一套契约 ------------------------------------
def test_modelscope_download_follows_the_same_contract(links, monkeypatch):
    _fake_modelscope(monkeypatch, lambda: iter([{"q": i} for i in range(30)]))

    payload = json.loads(_call(mcp_datasets.download_ms_dataset, dataset_id="modelscope/gsm8k")[0])

    assert payload["source"] == "modelscope"
    assert payload["rows"] == 30 and payload["complete"] is True
    assert "modelscope.cn/datasets/modelscope/gsm8k" in payload["url"]
    meta = json.loads(Path(payload["path"]).with_suffix(".meta.json").read_text())
    assert meta["source"] == "modelscope"          # 落盘纪律两个 hub 共用一套实现


def test_same_id_on_two_hubs_does_not_share_a_cache_entry(links, monkeypatch):
    _fake_modelscope(monkeypatch, lambda: iter([{"q": 1}]))

    hf_path = json.loads(_call(mcp_datasets.download_dataset, dataset_id="openai/gsm8k")[0])["path"]
    ms_path = json.loads(
        _call(mcp_datasets.download_ms_dataset, dataset_id="openai/gsm8k")[0]
    )["path"]

    # 同名数据集在两个 hub 上未必是同一份内容，共用缓存键会让节点拿到错的数据
    assert hf_path != ms_path
    assert Path(hf_path).name.startswith("hf__") and Path(ms_path).name.startswith("ms__")


def test_modelscope_missing_sdk_is_explained_not_crashed(links, monkeypatch):
    monkeypatch.setitem(sys.modules, "modelscope", None)
    text, is_error = _call(mcp_datasets.download_ms_dataset, dataset_id="a/b")
    assert is_error and "缺少依赖" in text          # 没装 modelscope 不该影响 HF 侧


def test_ms_search_result_carries_the_fields_that_decide_usability(links, monkeypatch):
    repo = types.SimpleNamespace(
        repo_id="AI-ModelScope/gsm8k", downloads=1200, license="MIT", gated=False,
        tasks=["text-generation"], tags=["math"], likes=None, file_size=0,
        last_modified="2026-01-01", description="x" * 500,
    )
    monkeypatch.setattr(mcp_datasets, "_ms_search", lambda q, n: [mcp_datasets._ms_repo_info(repo)])

    hit = json.loads(_call(mcp_datasets.search_ms_datasets, query="gsm8k")[0])[0]

    # gated 决定拿不拿得到、license 决定能不能用；两者任一不满足，预览和下载都是白费
    assert hit["dataset_id"] == "AI-ModelScope/gsm8k" and hit["license"] == "MIT"
    assert hit["tasks"] == ["text-generation"]
    assert len(hit["description"]) < 320 and hit["description"].endswith("…")
    # 空值不占位，省得几十条结果糊满 Agent 的上下文
    assert "likes" not in hit and "file_size" not in hit and "gated" not in hit


def test_modelscope_inspect_returns_metadata_without_loading_rows(links, monkeypatch):
    def _boom():
        raise AssertionError("inspect_ms_dataset must not stream dataset rows")

    _fake_modelscope(monkeypatch, _boom)

    payload = json.loads(_call(
        mcp_datasets.inspect_ms_dataset, dataset_id="modelscope/gsm8k"
    )[0])

    assert payload["dataset_id"] == "modelscope/gsm8k"
    assert payload["license"] == "Apache-2.0"
    assert payload["files"][0]["path"] == "README.md"
    assert "不会执行数据集脚本" in payload["inspect_note"]


def test_hf_tools_reject_modelscope_ids_before_network(links):
    for tool in (
        mcp_datasets.inspect_dataset,
        mcp_datasets.preview_dataset,
        mcp_datasets.download_dataset,
    ):
        text, is_error = _call(tool, dataset_id="modelscope/gsm8k")
        assert is_error
        assert "ModelScope" in text and "inspect_ms_dataset" in text


def test_script_datasets_are_refused_with_a_way_out(links, monkeypatch):
    def _boom(*a, **k):
        raise ValueError("Loading x requires ... set the option `trust_remote_code=True` ...")

    monkeypatch.setattr(mcp_datasets, "_row_iter", _boom)
    text, is_error = _call(mcp_datasets.download_ms_dataset, dataset_id="modelscope/gsm8k")

    # 加载脚本型数据集 = 在编排器进程里跑仓库里的任意代码，这条路不开；
    # 但要让 Agent 知道该换数据集，而不是以为工具坏了
    assert is_error and "脚本型数据集" in text and "换一份" in text
    assert "trust_remote_code" not in text.split("。")[0]


# ---- 契约：调用参数要对得上真实装着的 SDK ----------------------------------
# 上面所有测试都 stub 掉了 SDK，于是**上游签名漂移完全看不见**——`HfApi.list_datasets` 的
# `direction` 参数在 huggingface_hub 1.x 被移除，我们照传，单测全绿而线上每次搜索都报
# TypeError。这里不打网络，只把真实签名拿来 bind 一次，把这类漂移挡在测试阶段。
def _binds(func, *args, **kwargs):
    inspect.signature(func).bind(*args, **kwargs)


def test_huggingface_call_kwargs_match_the_installed_sdk():
    hf = pytest.importorskip("huggingface_hub")
    _binds(hf.HfApi.list_datasets, None, search="q", sort="downloads", limit=5)
    _binds(hf.HfApi.dataset_info, None, "owner/name")

    ds = pytest.importorskip("datasets")
    _binds(ds.load_dataset, "owner/name", "cfg", split="train", streaming=True)


def test_modelscope_call_kwargs_match_the_installed_sdk():
    ms = pytest.importorskip("modelscope")
    _binds(ms.HubApi.list_repos, None, "dataset", search="q", page_size=5)
    _binds(ms.HubApi.get_repo, None, "owner/name", repo_type="dataset")
    _binds(ms.HubApi.get_dataset_files, None, "owner/name")

    msd = pytest.importorskip("modelscope.msdatasets")
    _binds(
        msd.MsDataset.load, "owner/name", subset_name="cfg", split="train", use_streaming=True
    )


# ---- 连接配置 --------------------------------------------------------------
def test_connection_settings_come_from_config(tmp_path, monkeypatch):
    for key in ("HF_ENDPOINT", "HF_TOKEN", "HF_HOME"):
        monkeypatch.delenv(key, raising=False)

    mcp_datasets.configure(
        tmp_path / "dl",
        {"endpoint": "https://hf-mirror.com", "token": "secret", "cache_dir": "/tmp/hf",
         "max_download_mb": 1},
    )

    # in-process MCP 跑在编排器进程里，读不到 agent_sdk.env；连接配置只能从这里进环境
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert os.environ["HF_TOKEN"] == "secret"
    assert os.environ["HF_HOME"] == "/tmp/hf"
    assert mcp_datasets._max_download_bytes == 1024 * 1024


def test_blank_connection_settings_do_not_clobber_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://preset.example")
    mcp_datasets.configure(tmp_path / "dl", {"endpoint": "", "token": ""})
    # 留空 = 沿用已有环境/本地登录态，而不是把它清成空串
    assert os.environ["HF_ENDPOINT"] == "https://preset.example"


def test_download_requires_dataset_id(links):
    text, is_error = _call(mcp_datasets.download_dataset, dataset_id="")
    assert is_error and "dataset_id" in text


def test_failures_come_back_as_actionable_text(links, monkeypatch):
    def _raise():
        raise ConnectionError("name resolution failed")

    _fake_datasets(monkeypatch, _raise)
    text, is_error = _call(mcp_datasets.download_dataset, dataset_id="a/b")
    assert is_error
    assert "网络不可达" in text and "换一个数据集" in text   # 让 Agent 知道下一步怎么走


def test_missing_dependency_is_explained_not_crashed(links, monkeypatch):
    def _raise():
        raise ImportError("No module named 'datasets'")

    _fake_datasets(monkeypatch, _raise)
    text, is_error = _call(mcp_datasets.download_dataset, dataset_id="a/b")
    assert is_error and "缺少依赖" in text


def test_preview_truncates_long_values(links, monkeypatch):
    monkeypatch.setattr(
        mcp_datasets, "_stream_rows", lambda *a: [{"passage": "x" * 5000}]
    )
    text, _ = _call(mcp_datasets.preview_dataset, dataset_id="a/b")
    assert "共 5000 字符" in text


# ---- prompt 注入 -----------------------------------------------------------
def test_prompt_block_only_when_tools_are_mounted():
    off = build_pipeline_prompt(profile=PROFILE, corpus=SCRATCH, node_dir="node_1")
    assert "外部数据搜索" not in off

    on = build_pipeline_prompt(
        profile=PROFILE, corpus=SCRATCH, node_dir="node_1", data_links_dir="/ws/data_links"
    )
    assert "外部数据搜索" in on
    assert "search_datasets" in on and "download_dataset" in on
    assert "/ws/data_links" in on
    assert "绝对路径" in on           # 相对路径在子进程里会失效
    assert "license" in on            # 用别人的数据前先看许可
    assert "严禁引入评测基准的测试集" in on
    # 绕开工具拉来的数据没有 .meta.json，事后无从追溯，所以要明说这条路不许走
    assert "所有外部数据必须经这些工具获取" in on
    assert "load_dataset" in on
    # FileStorage 的 hf:/ms: 远程写法比 load_dataset 更顺手，而且是我们自己的技能文档教的
    assert "hf:owner/name:config:split" in on
    # 默认全量下载是刻意的：采样属于流水线的决策，不该被工具替它做掉
    assert "整个 split" in on
    assert "不要传 `max_rows` 去限制下载量" in on
    # 两个 hub + 三个发现渠道
    assert "inspect_ms_dataset" in on and "HuggingFace 连不上时改用 ModelScope" in on
    assert "ModelScope ID" in on and "*_ms_dataset" in on
    assert "search_papers" in on and "只找线索、不取数" in on


def test_prompt_forbids_external_data_when_tools_are_off():
    off = build_pipeline_prompt(profile=PROFILE, corpus=SCRATCH, node_dir="node_1")

    # 工具没挂载 ≠ Agent 没法联网：它的 Bash/WebFetch 和 datasets 库一直都在手边。
    # 少了这段，"本轮不引入外部数据"就只存在于配置里，prompt 对 Agent 只字未提。
    assert "本轮不启用" in off
    assert "不要联网拉取任何外部数据" in off
    assert "LLM serving API 的调用不受此限" in off
