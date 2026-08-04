"""in-process MCP server：让 PipelineAgent 去 HuggingFace Hub 上找、看、取真实数据集。

为什么要有这组工具：光靠模型知识从零合成，数据的多样性上限就是模型自己的分布；而真实公开
语料是搜索空间里一整个维度——「换一个数据源」往往比「再调一版 prompt」的增益大得多。Agent
SDK 自带 WebSearch/WebFetch 能找到线索，但没有「按结构化元信息挑数据集 → 预览 → 取样落盘」
这条链路，本模块补的就是这一段。

四条纪律，写死在工具行为里而不是只写在 prompt 里：

1. **落盘集中**：所有下载都进共享的 `data_links/`，跨节点复用同一份文件，不在每个节点目录下
   各存一份几百 MB 的副本。
2. **可溯源**：每个文件旁边落一份 `.meta.json`（来源 dataset_id / config / split / URL /
   license / 行数 / 字节 / 是否完整 / 时间）。数据从哪来是论文和去污染审计都要回答的问题，
   事后从文件名是猜不出来的。
3. **默认取全量 split**：不做"前 N 行"这种默认截断。取多少、怎么采样、按什么过滤，本就是
   **流水线该做的决策**，也正是被搜索的对象；把它固化在工具里，等于替搜索先做了一个它无从
   推翻的选择。而且"前 N 行"不是随机样本——数据集若按难度/来源/时间有序，头部就是有偏子集。
4. **只取一个 split，不整仓库拉**：整仓库会把 `test` split 一起拖下来，那正是我们花力气防的
   泄露入口。落盘仍受 `max_download_mb` 字节上限保护，触顶即截断并在 meta 里标明。

依赖（huggingface_hub / datasets）**惰性导入**：装没装都不影响框架启动，缺了只是这组工具
返回一条说明。网络不通同理——这些是可选增强，不该让主循环起不来。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from dataflow_evolver.utils.logging import get_logger

logger = get_logger("llm.mcp_datasets")

DEFAULT_PREVIEW_ROWS = 3
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_MAX_DOWNLOAD_MB = 2048     # 单份数据集落盘上限；触顶截断而非报错，见 _stream_to_file

HF = "huggingface"
MS = "modelscope"
_HUB_CODE = {HF: "hf", MS: "ms"}
DEFAULT_MS_ENDPOINT = "https://modelscope.cn"

# YAML `dataset_search` 段 → 两个 hub 认的环境变量名
_HUB_ENV = {
    "endpoint": "HF_ENDPOINT", "token": "HF_TOKEN", "cache_dir": "HF_HOME",
    "modelscope_endpoint": "MODELSCOPE_ENDPOINT", "modelscope_token": "MODELSCOPE_API_TOKEN",
}
_SECRET_ENV = {"HF_TOKEN", "MODELSCOPE_API_TOKEN"}

_data_links_dir: Path = Path("workspace/data_links")
_max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_MB * 1024 * 1024
_ms_endpoint: str = DEFAULT_MS_ENDPOINT
# dataset_id -> license，由 inspect_dataset 填、download_dataset 取（见 _license_of）
_license_cache: dict[str, str | None] = {}


def configure(data_links_dir: str | Path, cfg: Any = None) -> Path:
    """由编排器在启动时指定共享下载目录与连接配置（`dataset_search` 配置段）。"""
    global _data_links_dir, _max_download_bytes, _ms_endpoint
    _data_links_dir = Path(data_links_dir).expanduser().resolve()
    _data_links_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg if cfg is not None else {}
    _apply_hub_env(cfg)
    mb = float(cfg.get("max_download_mb", 0) or DEFAULT_MAX_DOWNLOAD_MB)
    _max_download_bytes = int(mb * 1024 * 1024)
    _ms_endpoint = str(
        cfg.get("modelscope_endpoint", "") or DEFAULT_MS_ENDPOINT
    ).rstrip("/")
    logger.info("外部数据下载目录：%s（单份上限 %.0f MB）", _data_links_dir, mb)
    return _data_links_dir


def _apply_hub_env(cfg: Any) -> None:
    """把 `dataset_search` 里的连接配置写进环境变量。

    **必须在启动时设、且两个 hub 的依赖必须惰性导入**：`huggingface_hub` 在 import 期就把
    `HF_ENDPOINT` 等读成模块常量，导入之后再改 `os.environ` 不生效。本模块所有 hub 导入都在
    函数内部，`configure()` 又在 `Orchestrator.__init__` 里调用，早于任何一次工具调用。

    另一个坑：`agent_sdk.env` 那一段是发给 Agent SDK **子进程**的，而 dataset-search 是
    in-process MCP，工具跑在编排器自己的进程里，读不到那些变量。连接配置只能走这里。
    """
    for key, env_key in _HUB_ENV.items():
        value = str(cfg.get(key, "") or "").strip()
        if value:
            os.environ[env_key] = value
            logger.info(
                "外部数据：%s=%s", env_key, "***" if env_key in _SECRET_ENV else value
            )


def data_links_dir() -> Path:
    _data_links_dir.mkdir(parents=True, exist_ok=True)
    return _data_links_dir


# ---- 结果封装：MCP 工具永远返回文本，失败也要给出可行动的下一步 ------------
def _ok(payload: Any) -> dict[str, Any]:
    # default=str：Hub SDK 会在元信息里塞 datetime 等非 JSON 类型，序列化异常会让一次本来成功的
    # 搜索变成工具崩溃——宁可把它转成字符串，也不能因为一个展示字段丢掉整个结果。
    text = (
        payload if isinstance(payload, str)
        else json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )
    return {"content": [{"type": "text", "text": text}]}


def _err(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "is_error": True}


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi()


def _looks_like_modelscope_id(dataset_id: str) -> bool:
    """Heuristic guard for IDs returned by ModelScope tools.

    The HF Hub can technically have an org named "modelscope", but in practice an ID
    beginning with this prefix usually came from `search_ms_datasets`. Routing it to
    HF `inspect_dataset` sends the agent back through a slow/unreachable HF endpoint.
    """
    return dataset_id.lower().startswith("modelscope/")


def _modelscope_route_error(tool_name: str) -> dict[str, Any]:
    return _err(
        f"{tool_name} 是 HuggingFace-only 工具，不能处理 ModelScope 数据集 ID。"
        "如果 ID 来自 search_ms_datasets，请改用 inspect_ms_dataset；随后用 "
        "preview_ms_dataset / download_ms_dataset。"
    )


def _safe_name(hub: str, dataset_id: str, config: str, split: str) -> str:
    """落盘文件名。带 hub 短码：同名数据集在两个 hub 上未必是同一份内容，不能共用一个缓存键。"""
    parts = [p for p in (_HUB_CODE[hub], dataset_id, config, split) if p]
    return "__".join(p.replace("/", "_").replace(" ", "_") for p in parts) + ".jsonl"


# ---- 工具 ------------------------------------------------------------------
@tool(
    "search_datasets",
    "Search HuggingFace Hub for datasets by keyword. Returns id, downloads, likes and tags. "
    "Hub search is keyword-based (no semantic matching), so use precise terms.",
    {
        "query": Annotated[str, "Keyword query, e.g. 'grade school math word problems'"],
        "limit": Annotated[str, f"Max results (default {DEFAULT_SEARCH_LIMIT})"],
    },
)
async def search_datasets(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query is required")
    limit = _as_int(args.get("limit"), DEFAULT_SEARCH_LIMIT, cap=50)
    try:
        # 按下载量排序：作为"这个数据集是否被用过、是否可信"的粗代理，比默认相关度更实用。
        # 不传 `direction`：huggingface_hub 1.x 已移除该参数，`sort="downloads"` 本身即为降序。
        items = list(_hf_api().list_datasets(search=query, sort="downloads", limit=limit))
    except Exception as exc:
        return _err(_explain(exc, "搜索 HuggingFace 数据集"))
    if not items:
        return _ok(f"没有匹配 '{query}' 的数据集。Hub 搜索不支持语义匹配，请换更具体的关键词。")
    return _ok([
        {
            "dataset_id": d.id,
            "downloads": getattr(d, "downloads", None),
            "likes": getattr(d, "likes", None),
            "tags": list(getattr(d, "tags", []) or [])[:12],
        }
        for d in items
    ])


@tool(
    "inspect_dataset",
    "Inspect a HuggingFace dataset without downloading it: license, size, tags, available "
    "configs/splits and file list. Call this before download_dataset.",
    {"dataset_id": Annotated[str, "e.g. 'openai/gsm8k'"]},
)
async def inspect_dataset(args: dict[str, Any]) -> dict[str, Any]:
    dataset_id = (args.get("dataset_id") or "").strip()
    if not dataset_id:
        return _err("dataset_id is required")
    if _looks_like_modelscope_id(dataset_id):
        return _modelscope_route_error("inspect_dataset")
    try:
        info = _hf_api().dataset_info(dataset_id)
    except Exception as exc:
        return _err(_explain(exc, f"获取数据集 {dataset_id} 的元信息"))

    card = getattr(info, "cardData", None) or {}
    _license_cache[dataset_id] = card.get("license")
    detail = {
        "dataset_id": info.id,
        "license": card.get("license"),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "tags": list(getattr(info, "tags", []) or [])[:20],
        "configs_and_splits": _configs_and_splits(dataset_id),
        "files": [s.rfilename for s in (getattr(info, "siblings", None) or [])][:40],
    }
    return _ok(detail)


@tool(
    "preview_dataset",
    "Fetch a few real rows of a HuggingFace dataset to check its fields and style before "
    "committing to it. Streams only the first rows; nothing is written to disk.",
    {
        "dataset_id": Annotated[str, "e.g. 'openai/gsm8k'"],
        "config": Annotated[str, "Config/subset name (optional; see inspect_dataset)"],
        "split": Annotated[str, "Split name, default 'train'"],
        "rows": Annotated[str, f"How many rows (default {DEFAULT_PREVIEW_ROWS}, max 20)"],
    },
)
async def preview_dataset(args: dict[str, Any]) -> dict[str, Any]:
    dataset_id = (args.get("dataset_id") or "").strip()
    if not dataset_id:
        return _err("dataset_id is required")
    if _looks_like_modelscope_id(dataset_id):
        return _modelscope_route_error("preview_dataset")
    config = (args.get("config") or "").strip() or None
    split = (args.get("split") or "train").strip()
    rows = _as_int(args.get("rows"), DEFAULT_PREVIEW_ROWS, cap=20)
    try:
        sample = _stream_rows(HF, dataset_id, config, split, rows)
    except Exception as exc:
        return _err(_explain(exc, f"预览数据集 {dataset_id}"))
    if not sample:
        return _ok(f"数据集 {dataset_id}（config={config}, split={split}）没有取到数据行。")
    return _ok({
        "source": HF, "dataset_id": dataset_id, "config": config, "split": split,
        "fields": sorted({k for row in sample for k in row}),
        "rows": [_truncate_row(r) for r in sample],
    })


@tool(
    "download_dataset",
    "Download a HuggingFace dataset split into the shared data_links directory as a .jsonl "
    "file (with a .meta.json provenance sidecar), and return its absolute path. Downloads the "
    "ENTIRE split by default — decide how to sample/filter it inside your pipeline, not here. "
    "Use the returned absolute path inside the pipeline you write.",
    {
        "dataset_id": Annotated[str, "e.g. 'openai/gsm8k'"],
        "config": Annotated[str, "Config/subset name (optional)"],
        "split": Annotated[str, "Split name, default 'train'"],
        "max_rows": Annotated[
            str,
            "Optional row cap. Omit or 0 = download the whole split (recommended). "
            "A cap keeps only the FIRST N rows, which is not a random sample.",
        ],
    },
)
async def download_dataset(args: dict[str, Any]) -> dict[str, Any]:
    return _download(HF, args)


def _download(hub: str, args: dict[str, Any]) -> dict[str, Any]:
    """两个 hub 共用的落盘实现：复用检查 → 流式写临时文件 → 原子改名 → 写溯源边车。

    共用而非各写一份：字节上限、原子落盘、溯源字段、复用提示这几条纪律，任何一个 hub 漏掉都
    等于开了个后门，而"两份实现慢慢长歪"是这类重复最典型的下场。
    """
    dataset_id = (args.get("dataset_id") or "").strip()
    if not dataset_id:
        return _err("dataset_id is required")
    if hub == HF and _looks_like_modelscope_id(dataset_id):
        return _modelscope_route_error("download_dataset")
    config = (args.get("config") or "").strip() or None
    split = (args.get("split") or "train").strip()
    max_rows = _as_int(args.get("max_rows"), 0)     # 0 = 全量

    target = data_links_dir() / _safe_name(hub, dataset_id, config or "", split)
    if target.exists():
        # 跨节点复用：同一份数据不重复下载，也让不同节点用的确实是同一份文件。
        meta = _read_meta(target)
        return _ok({
            "path": str(target), "reused": True, "note": _reuse_note(meta), **meta,
        })

    # 先写临时文件再原子改名：下载中途被打断（超时/中断）会留下半份数据，若直接写目标名，
    # 下一个节点的 target.exists() 照样命中，把截断的文件当完整数据用——不报错，也查不出来。
    tmp = target.with_name(target.name + ".part")
    try:
        stats = _stream_to_file(
            hub, dataset_id, config, split, tmp, max_rows, _max_download_bytes
        )
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return _err(_explain(exc, f"从 {hub} 下载数据集 {dataset_id}"))
    if not stats.rows:
        tmp.unlink(missing_ok=True)
        return _err(f"数据集 {dataset_id}（config={config}, split={split}）没有取到任何数据行。")
    os.replace(tmp, target)

    meta = {
        "source": hub,
        "dataset_id": dataset_id, "config": config, "split": split,
        "url": _dataset_url(hub, dataset_id),
        "license": _license_of(dataset_id) if hub == HF else None,
        "rows": stats.rows, "bytes": stats.written,
        "complete": stats.reason is None,
        "truncated_by": stats.reason,
        "truncation_note": _TRUNCATION_NOTE if stats.reason else None,
        "fields": sorted(stats.fields),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    target.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "外部数据落盘：%s（%d 行 / %.1f MB，来自 %s:%s%s）",
        target, stats.rows, stats.written / 1024 / 1024, _HUB_CODE[hub], dataset_id,
        f"，因 {stats.reason} 截断" if stats.reason else "，完整 split",
    )
    return _ok({"path": str(target), "reused": False, **meta})


def _dataset_url(hub: str, dataset_id: str) -> str:
    if hub == MS:
        return f"{_ms_endpoint}/datasets/{dataset_id}"
    return f"https://huggingface.co/datasets/{dataset_id}"


# ---- ModelScope：与 HF 同一套契约的第二个数据源 ----------------------------
# 存在的理由不是"多一个源"，而是**在国内网络下 HF 常常直连不通**。这时可选项只有配镜像或换源，
# 而 ModelScope 上中文语料的覆盖本来也更好。三个工具与 HF 侧一一对应、落盘纪律完全一致。
@tool(
    "search_ms_datasets",
    "Search ModelScope for datasets by keyword. Use this when HuggingFace is unreachable, or "
    "when looking for Chinese-language corpora (ModelScope covers them better). Next call "
    "inspect_ms_dataset on the selected dataset_id, not HuggingFace inspect_dataset.",
    {
        "query": Annotated[str, "Keyword query, e.g. '数学 推理'"],
        "limit": Annotated[str, f"Max results (default {DEFAULT_SEARCH_LIMIT})"],
    },
)
async def search_ms_datasets(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query is required")
    limit = _as_int(args.get("limit"), DEFAULT_SEARCH_LIMIT, cap=50)
    try:
        items = _ms_search(query, limit)[:limit]
    except Exception as exc:
        return _err(_explain(exc, "搜索 ModelScope 数据集"))
    if not items:
        return _ok(f"没有匹配 '{query}' 的 ModelScope 数据集，换个关键词或改用 HuggingFace。")
    return _ok(items)


@tool(
    "inspect_ms_dataset",
    "Inspect a ModelScope dataset without downloading rows or executing dataset scripts. Returns "
    "license, tags, tasks, downloads, visibility, description and repository file list. Use this "
    "after search_ms_datasets and before preview_ms_dataset/download_ms_dataset.",
    {"dataset_id": Annotated[str, "e.g. 'modelscope/gsm8k'"]},
)
async def inspect_ms_dataset(args: dict[str, Any]) -> dict[str, Any]:
    dataset_id = (args.get("dataset_id") or "").strip()
    if not dataset_id:
        return _err("dataset_id is required")
    try:
        detail = _ms_inspect(dataset_id)
    except Exception as exc:
        return _err(_explain(exc, f"获取 ModelScope 数据集 {dataset_id} 的元信息"))
    return _ok(detail)


@tool(
    "preview_ms_dataset",
    "Fetch a few real rows of a ModelScope dataset to check its fields and style. "
    "Streams only the first rows; nothing is written to disk.",
    {
        "dataset_id": Annotated[str, "e.g. 'modelscope/gsm8k'"],
        "config": Annotated[str, "Subset name (optional)"],
        "split": Annotated[str, "Split name, default 'train'"],
        "rows": Annotated[str, f"How many rows (default {DEFAULT_PREVIEW_ROWS}, max 20)"],
    },
)
async def preview_ms_dataset(args: dict[str, Any]) -> dict[str, Any]:
    dataset_id = (args.get("dataset_id") or "").strip()
    if not dataset_id:
        return _err("dataset_id is required")
    config = (args.get("config") or "").strip() or None
    split = (args.get("split") or "train").strip()
    rows = _as_int(args.get("rows"), DEFAULT_PREVIEW_ROWS, cap=20)
    try:
        sample = _stream_rows(MS, dataset_id, config, split, rows)
    except Exception as exc:
        return _err(_explain(exc, f"预览 ModelScope 数据集 {dataset_id}"))
    if not sample:
        return _ok(f"数据集 {dataset_id}（config={config}, split={split}）没有取到数据行。")
    return _ok({
        "source": MS, "dataset_id": dataset_id, "config": config, "split": split,
        "fields": sorted({k for row in sample for k in row}),
        "rows": [_truncate_row(r) for r in sample],
    })


@tool(
    "download_ms_dataset",
    "Download a ModelScope dataset split into the shared data_links directory as a .jsonl file "
    "(with a .meta.json provenance sidecar), and return its absolute path. Downloads the ENTIRE "
    "split by default. Same contract as download_dataset, different hub.",
    {
        "dataset_id": Annotated[str, "e.g. 'modelscope/gsm8k'"],
        "config": Annotated[str, "Subset name (optional)"],
        "split": Annotated[str, "Split name, default 'train'"],
        "max_rows": Annotated[
            str,
            "Optional row cap. Omit or 0 = download the whole split (recommended). "
            "A cap keeps only the FIRST N rows, which is not a random sample.",
        ],
    },
)
async def download_ms_dataset(args: dict[str, Any]) -> dict[str, Any]:
    return _download(MS, args)


def _ms_search(query: str, limit: int) -> list[dict[str, Any]]:
    """走 modelscope SDK 的 `list_repos`，而不是自己拼 HTTP。

    最初这里手写了一条 `PUT /api/v1/dolphin/datasets`，实测直接 404——那是社区流传的旧端点。
    SDK 的 `HubApi.list_repos("dataset", search=...)` 是官方接口，且一次就把 license / tags /
    tasks / downloads / gated 都带回来，等于 HF 侧 search + inspect 两步的信息量。
    """
    from modelscope import HubApi

    page = HubApi().list_repos("dataset", search=query, page_size=limit)
    return [_ms_repo_info(r) for r in page.items]


def _ms_inspect(dataset_id: str) -> dict[str, Any]:
    """Metadata-only ModelScope inspect.

    Do not call `MsDataset.load()` here: script-based ModelScope datasets may require
    `trust_remote_code=True`, which would execute repository code in the orchestrator process.
    """
    from modelscope import HubApi

    api = HubApi()
    repo = api.get_repo(dataset_id, repo_type="dataset")
    info = _ms_repo_info(repo)
    try:
        files = api.get_dataset_files(dataset_id)
    except Exception as exc:  # noqa: BLE001 - metadata is still useful without file listing.
        files_detail: Any = f"（无法列出文件：{type(exc).__name__}）"
    else:
        files_detail = [
            {
                "path": f.get("Path") or f.get("Name"),
                "size": f.get("Size"),
                "is_lfs": f.get("IsLFS"),
            }
            for f in files[:40]
        ]
    info["files"] = files_detail
    info["inspect_note"] = (
        "这是 metadata-only inspect：不会下载样本行，也不会执行数据集脚本。"
        "下一步请用 preview_ms_dataset 查看字段/样式，或用 download_ms_dataset 落盘。"
    )
    return info


def _ms_repo_info(repo: Any) -> dict[str, Any]:
    """RepoInfo → 精简 dict。只挑对"这个数据集能不能用"有决定作用的字段。

    `gated` 与 `license` 单列：前者决定拿不拿得到，后者决定能不能用；两者任一不满足，
    后面的预览和下载都是白费。
    """
    # dataset_id 放第一位：它是后续每个工具都要回填的那个值，被长 description 顶到末尾会难找。
    fields = ("downloads", "likes", "license", "gated", "tasks", "tags",
              "file_size", "last_modified", "description")
    info: dict[str, Any] = {"dataset_id": getattr(repo, "repo_id", None)}
    info.update({f: getattr(repo, f, None) for f in fields})
    desc = info.get("description")
    if isinstance(desc, str) and len(desc) > 300:
        info["description"] = desc[:300] + "…"
    # 只报"确有其值"的字段：`gated: false`、`file_size: 0`（未知）这类占位对判断毫无帮助，
    # 而搜索一次返回几十条，每条多几个空字段就是几百行糊进 Agent 的上下文。
    return {k: v for k, v in info.items() if v}


# 截断样本为何要额外说明：它不是随机子集，而是 split 的头部。数据集若按难度/来源/时间排过序，
# 头部就是有偏的一段，而 Agent 会拿它当"这个数据源的样子"来判断适不适用。
_TRUNCATION_NOTE = (
    "本文件只包含该 split 的**前若干行**（触发上限后截断），不是随机采样。"
    "若数据集本身有序（按难度/来源/时间等），这段头部会有系统性偏差；"
    "据此判断数据源风格时请留意，必要时调大 dataset_search.max_download_mb 重取。"
)


@dataclass
class _DownloadStats:
    rows: int = 0
    written: int = 0
    fields: set[str] = field(default_factory=set)
    reason: str | None = None      # None=完整 split；否则为 "max_rows" / "size_cap"


# ---- 内部实现 --------------------------------------------------------------
def _row_iter(hub: str, dataset_id: str, config: str | None, split: str) -> Any:
    """按 hub 取一个**流式**行迭代器。两个 hub 只在这里分叉，落盘逻辑完全共用。"""
    if hub == MS:
        from modelscope.msdatasets import MsDataset

        return MsDataset.load(
            dataset_id, subset_name=config, split=split, use_streaming=True
        )
    from datasets import load_dataset

    return load_dataset(dataset_id, config, split=split, streaming=True)


def _stream_rows(
    hub: str, dataset_id: str, config: str | None, split: str, n: int
) -> list[dict]:
    """流式取前 n 行，供 preview 使用（不落盘）。"""
    out: list[dict] = []
    for row in _row_iter(hub, dataset_id, config, split):
        out.append(dict(row))
        if len(out) >= n:
            break
    return out


def _stream_to_file(
    hub: str,
    dataset_id: str,
    config: str | None,
    split: str,
    target: Path,
    max_rows: int,
    max_bytes: int,
) -> _DownloadStats:
    """流式把整个 split 边读边写进 jsonl，返回落盘统计。

    流式 + 增量写盘（而非整体加载后一次性写出）：内存占用与数据集大小无关。非流式加载一个几 GB
    的 split 会把编排器进程直接撑爆，而那时候树里可能已经跑了几十个节点。

    上限用**字节**而不是行数：一行的大小在不同数据集之间能差几个数量级，行数上限对"会不会
    拉爆磁盘"这个真正要防的问题几乎没有约束力。
    """
    stats = _DownloadStats()
    ds = _row_iter(hub, dataset_id, config, split)
    with target.open("w", encoding="utf-8") as f:
        for row in ds:
            row = dict(row)
            line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
            encoded = len(line.encode("utf-8"))
            if max_bytes and stats.written + encoded > max_bytes:
                stats.reason = "size_cap"
                break
            f.write(line)
            stats.rows += 1
            stats.written += encoded
            stats.fields.update(row)
            if max_rows and stats.rows >= max_rows:
                stats.reason = "max_rows"
                break
    return stats


def _reuse_note(meta: dict) -> str:
    """复用提示。文件完整与否必须说清楚，否则复用会静默改变后续节点拿到的数据量。"""
    if not meta:
        return (
            "该数据此前已下载，直接复用同一份文件。但它缺少 .meta.json 溯源记录，"
            "无法确认是否完整——如需确保完整，请删掉该文件后重新下载。"
        )
    if meta.get("complete"):
        return "该数据此前已下载（完整 split），直接复用同一份文件。"
    return (
        f"该数据此前已下载，直接复用同一份文件。**注意它不完整**："
        f"因 {meta.get('truncated_by')} 截断，只有 {meta.get('rows')} 行。"
        f"{_TRUNCATION_NOTE}"
    )


def _configs_and_splits(dataset_id: str) -> dict[str, list[str]] | str:
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names
    except ImportError:
        return "（datasets 未安装，无法列出 configs/splits）"
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception as exc:
        return f"（无法列出 configs：{type(exc).__name__}）"
    out: dict[str, list[str]] = {}
    for cfg in configs[:10]:
        try:
            out[cfg] = list(get_dataset_split_names(dataset_id, cfg))
        except Exception:
            out[cfg] = []
    return out


def _license_of(dataset_id: str) -> str | None:
    """取数据集 license 写进溯源边车。

    Agent 在 inspect_dataset 里看过一眼 license，但那只存在于它的上下文里；论文交代数据许可、
    或事后审计某份数据能不能用，看的是磁盘上的边车。所以这里要独立取一次，并复用 inspect
    阶段已经拿到的结果（按流程 download 前必先 inspect，通常不产生额外请求）。
    取不到就记 None——宁可显式标"未知"，也不要让缺失看起来像"没有许可要求"。
    """
    if dataset_id in _license_cache:
        return _license_cache[dataset_id]
    try:
        card = getattr(_hf_api().dataset_info(dataset_id), "cardData", None) or {}
        _license_cache[dataset_id] = card.get("license")
    except Exception as exc:
        logger.warning("获取 %s 的 license 失败：%s: %s", dataset_id, type(exc).__name__, exc)
        return None
    return _license_cache[dataset_id]


def _read_meta(target: Path) -> dict:
    meta_path = target.with_suffix(".meta.json")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _truncate_row(row: dict, maxlen: int = 300) -> dict:
    out = {}
    for k, v in row.items():
        s = v if isinstance(v, (int, float, bool, type(None))) else str(v)
        if isinstance(s, str) and len(s) > maxlen:
            s = s[:maxlen] + f"…(共 {len(s)} 字符)"
        out[str(k)] = s
    return out


def _as_int(raw: Any, default: int, cap: int = 0) -> int:
    """解析工具入参里的整数；`cap<=0` 表示不设上限，返回 0 表示"不限制"。"""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return 0 if default == 0 else default
    return min(value, cap) if cap > 0 else value


def _explain(exc: Exception, what: str) -> str:
    """把异常翻成 Agent 能据以决策的下一步，而不是丢一行栈。"""
    if isinstance(exc, ImportError):
        return f"{what}失败：缺少依赖（{exc}）。这组外部数据工具不可用，请改用其它数据来源。"
    logger.warning("%s失败：%s: %s", what, type(exc).__name__, exc)
    if "trust_remote_code" in str(exc):
        # 脚本型数据集在 ModelScope 上很常见（HF 已基本淘汰）。加载它等于在编排器进程里执行
        # 仓库里的任意代码，所以这条路我们不开——让 Agent 换一份数据，而不是让它以为工具坏了。
        return (
            f"{what}失败：这是**脚本型数据集**，加载需要执行仓库里的远程代码。"
            "出于安全考虑本工具不启用 trust_remote_code。请换一份以 parquet/jsonl 直接存储的"
            "数据集——同一份语料通常有多个镜像仓库，换一个 owner 往往就能拿到。"
        )
    return (
        f"{what}失败：{type(exc).__name__}: {exc}。"
        "可能是网络不可达、数据集不存在或需要授权；换一个数据集或改用从零合成。"
    )


dataset_mcp_server = create_sdk_mcp_server(
    name="dataset-search",
    version="2.0.0",
    tools=[
        search_datasets, inspect_dataset, preview_dataset, download_dataset,
        search_ms_datasets, inspect_ms_dataset, preview_ms_dataset, download_ms_dataset,
    ],
)
