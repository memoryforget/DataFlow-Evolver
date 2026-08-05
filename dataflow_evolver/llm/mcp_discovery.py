"""in-process MCP server：找线索。与 mcp_datasets 的分工是「发现 vs 取数」。

`dataset-search` 只能按关键词查 Hub 的索引，而 Hub 搜索不支持语义匹配——"想要小学数学应用题
的 CoT 语料"这种意图，往往是先在论文/博客/GitHub 里看到某个数据集被用过，才知道它叫什么。这
一段在此之前只能靠 Agent SDK 自带的 WebSearch/WebFetch：能用，但返回的是自由文本，Agent 得自
己从网页里把 dataset id 抠出来，而且没有统一的"下一步该干什么"的指引。

三个工具都**只发现、不下载**。这是刻意的：任何真正把数据取到本地的动作都必须经 dataset-search
落 `data_links/` + `.meta.json`，否则数据来源无从审计（见 core/provenance.py）。所以每个工具的
返回里都带一句"拿到 id 后回到 download_dataset / download_ms_dataset"。

凭据缺失不报错、不崩溃，只让**对应那一个**工具返回一条说明：
- `search_web` 需要 Serper key（付费），没有就用不了；
- `search_github` 无 token 也能用（GitHub 对匿名搜索限流 10 次/分钟），有 token 更宽松；
- `search_papers` 走 arXiv 公开 API，永远不需要凭据。
按工具而不是按整台 server 降级——少配一个 key 不该让另外两个也用不了。
"""
from __future__ import annotations

import json
import os
from typing import Annotated, Any
from xml.etree import ElementTree

from claude_agent_sdk import create_sdk_mcp_server, tool

from dataflow_evolver.utils.logging import get_logger

logger = get_logger("llm.mcp_discovery")

DEFAULT_LIMIT = 8
_TIMEOUT = 30

_SERPER_URL = "https://google.serper.dev/search"
_GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
# 必须 https：arXiv 官方示例给的是 http，但明文 http 在不少受限网络里会被直接挡掉（本环境即是
# 如此，https 通、http 返回 000），而那种失败看起来只是"网络不通"，很难联想到是协议问题。
_ARXIV_URL = "https://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

_NEXT_STEP = (
    "这些工具只负责找线索。确定要用某个数据集后，回到 dataset-search 的 "
    "`download_dataset`（HuggingFace）或 `download_ms_dataset`（ModelScope）取数——"
    "只有走那两个工具，数据才会落到共享目录并留下溯源记录。"
)

_serper_key: str = ""
_github_token: str = ""


def configure(cfg: Any = None) -> None:
    """由编排器在启动时注入凭据（`discovery` 配置段）。"""
    global _serper_key, _github_token
    cfg = cfg if cfg is not None else {}
    _serper_key = str(cfg.get("serper_api_key", "") or "").strip()
    _github_token = str(cfg.get("github_token", "") or "").strip()
    # OpenCode 的 local MCP 是独立进程，凭据只经继承环境传递，不写入 opencode.json。
    if _serper_key:
        os.environ["DF_DISCOVERY_SERPER_API_KEY"] = _serper_key
    else:
        os.environ.pop("DF_DISCOVERY_SERPER_API_KEY", None)
    if _github_token:
        os.environ["DF_DISCOVERY_GITHUB_TOKEN"] = _github_token
    else:
        os.environ.pop("DF_DISCOVERY_GITHUB_TOKEN", None)
    logger.info(
        "发现工具：web=%s，github=%s，papers=可用（arXiv 无需凭据）",
        "可用" if _serper_key else "未配 serper_api_key",
        "已认证" if _github_token else "匿名（限流 10 次/分钟）",
    )


def _ok(payload: Any) -> dict[str, Any]:
    text = (
        payload if isinstance(payload, str)
        else json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )
    return {"content": [{"type": "text", "text": text}]}


def _err(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "is_error": True}


def _explain(exc: Exception, what: str) -> str:
    if isinstance(exc, ImportError):
        return f"{what}失败：缺少 requests 依赖（{exc}）。该工具不可用，请改用其它发现渠道。"
    logger.warning("%s失败：%s: %s", what, type(exc).__name__, exc)
    return (
        f"{what}失败：{type(exc).__name__}: {exc}。"
        "可能是网络不可达、凭据无效或触发限流；换一个发现渠道，或直接用 dataset-search 按关键词搜。"
    )


def _limit(raw: Any) -> int:
    try:
        return max(1, min(int(str(raw).strip()), 25))
    except (TypeError, ValueError):
        return DEFAULT_LIMIT


@tool(
    "search_web",
    "Google web search (via Serper) for dataset leads: blog posts, dataset cards, papers that "
    "mention a corpus by name. Returns titles, links and snippets — it does NOT download "
    "anything. Use it when Hub keyword search fails because you don't know the dataset's name.",
    {
        "query": Annotated[str, "e.g. 'open source grade school math CoT dataset huggingface'"],
        "limit": Annotated[str, f"Max results (default {DEFAULT_LIMIT}, max 25)"],
    },
)
async def search_web(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query is required")
    if not _serper_key:
        return _err(
            "网页搜索未启用：未配置 discovery.serper_api_key。"
            "请改用 search_papers（arXiv，无需凭据）、search_github，或内置的 WebSearch 工具。"
        )
    try:
        import requests

        resp = requests.post(
            _SERPER_URL,
            headers={"X-API-KEY": _serper_key, "Content-Type": "application/json"},
            json={"q": query, "num": _limit(args.get("limit"))},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return _err(_explain(exc, "网页搜索"))

    hits = [
        {"title": x.get("title"), "link": x.get("link"), "snippet": x.get("snippet")}
        for x in (data.get("organic") or [])
    ]
    if not hits:
        return _ok(f"没有搜到 '{query}' 的结果，换个说法或改用 search_papers。")
    return _ok({"results": hits, "next_step": _NEXT_STEP})


@tool(
    "search_github",
    "Search GitHub repositories for dataset releases and data-processing projects. Returns repo "
    "metadata only — it does NOT clone or download. Useful for finding corpora that live in a "
    "repo rather than on a dataset hub.",
    {
        "query": Annotated[str, "e.g. 'math reasoning dataset jsonl'"],
        "limit": Annotated[str, f"Max results (default {DEFAULT_LIMIT}, max 25)"],
    },
)
async def search_github(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query is required")
    headers = {"Accept": "application/vnd.github+json"}
    if _github_token:
        headers["Authorization"] = f"Bearer {_github_token}"
    try:
        import requests

        resp = requests.get(
            _GITHUB_SEARCH_URL,
            headers=headers,
            params={"q": query, "sort": "stars", "order": "desc",
                    "per_page": _limit(args.get("limit"))},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        hint = "" if _github_token else "（当前为匿名请求，限流 10 次/分钟；配 discovery.github_token 可放宽）"
        return _err(_explain(exc, f"GitHub 搜索{hint}"))

    hits = [
        {
            "repo": x.get("full_name"), "stars": x.get("stargazers_count"),
            "description": x.get("description"), "url": x.get("html_url"),
            "license": ((x.get("license") or {}).get("spdx_id")),
        }
        for x in (data.get("items") or [])
    ]
    if not hits:
        return _ok(f"没有搜到 '{query}' 的仓库，换个关键词。")
    return _ok({"results": hits, "next_step": _NEXT_STEP})


@tool(
    "search_papers",
    "Search arXiv for papers about datasets or data-construction methods. Returns title, "
    "abstract and links. No credentials needed. Papers are often the fastest way to learn what "
    "a good corpus for your task is called, and how it was built.",
    {
        "query": Annotated[str, "e.g. 'synthetic math instruction tuning data'"],
        "limit": Annotated[str, f"Max results (default {DEFAULT_LIMIT}, max 25)"],
    },
)
async def search_papers(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query is required")
    try:
        import requests

        resp = requests.get(
            _ARXIV_URL,
            params={
                "search_query": f"all:{query}", "start": 0,
                "max_results": _limit(args.get("limit")),
                "sortBy": "relevance", "sortOrder": "descending",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        hits = _parse_arxiv(resp.text)
    except Exception as exc:
        return _err(_explain(exc, "arXiv 论文搜索"))

    if not hits:
        return _ok(f"arXiv 上没有搜到 '{query}' 的论文，换个关键词。")
    return _ok({"results": hits, "next_step": _NEXT_STEP})


def _parse_arxiv(xml_text: str) -> list[dict[str, Any]]:
    """解析 arXiv 的 Atom 响应。

    用标准库 ElementTree 而非 feedparser：只为读四个字段引入一个新依赖不划算，而这组工具本就
    要在依赖不全的环境里尽量还能用。
    """
    root = ElementTree.fromstring(xml_text)
    out: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        summary = (entry.findtext("atom:summary", "", _ARXIV_NS) or "").strip()
        out.append({
            "title": " ".join((entry.findtext("atom:title", "", _ARXIV_NS) or "").split()),
            "published": entry.findtext("atom:published", "", _ARXIV_NS),
            "url": entry.findtext("atom:id", "", _ARXIV_NS),
            "summary": summary[:600] + ("…" if len(summary) > 600 else ""),
        })
    return out


discovery_mcp_server = create_sdk_mcp_server(
    name="data-discovery",
    version="1.0.0",
    tools=[search_web, search_github, search_papers],
)
