"""外部数据溯源：记录「这个节点的数据到底从哪来」。

去污染（core/decontamination.py）扫的是**数据层**——产出里有没有 benchmark 的题。它能查出
泄露发生了，但查不出泄露从哪来。这两件事都要能回答：前者决定这个节点算不算数，后者决定
下一步该封掉哪个数据源、以及论文里怎么交代数据许可与出处。

外部数据进入流水线有两条路，记录方式不同：

1. **走 dataset-search MCP**（`download_dataset`）：落盘时就在 `data_links/` 里写了
   `.meta.json` 溯源边车，本模块只负责把它归到用了它的节点名下。
2. **绕过 MCP**：Agent 的内置工具是全放开的（`claude_code` 预设），它完全可以直接在流水线里
   写 `load_dataset("openai/gsm8k")`、`hf_hub_download(...)` 或 `requests.get(url)`，数据在
   执行时才拉进来，`data_links/` 里什么都不会留下。prompt 里已要求一律走工具，但 prompt 是
   软约束——**审计不能建立在 Agent 守规矩的前提上**。所以这里对节点代码做一遍静态扫描，把
   直连引用挖出来标成 `unmanaged`。

扫描是正则做的，因此有已知边界：运行时拼出来的 dataset id / URL（f-string、变量拼接）扫不到。
这不是可以靠加正则补上的——真要堵死得靠沙箱断网。定位是「把绝大多数情况变得可查」，而不是
「保证无遗漏」，`data_provenance.json` 里的 `method: "static_scan"` 就是在标这个限度。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from dataflow_evolver.utils.logging import get_logger

logger = get_logger("core.provenance")

TOOL_LOG_NAME = "agent_tools.jsonl"
PROVENANCE_NAME = "data_provenance.json"

# 工具参数里最长保留多少字符。Write/Edit 的 content 动辄上万字，全量落盘会让日志比数据还大，
# 而审计要看的是「调了什么工具、目标是谁」，不是把文件内容再抄一遍。
_MAX_ARG_CHARS = 600

# 扫描跳过的子目录：cache/ 是算子中间产物，tier2_out/ 是训练产物，都不是「节点写的代码」。
_SKIP_DIRS = {"cache", "tier2_out", "__pycache__", ".ipynb_checkpoints"}

_LOAD_DATASET = re.compile(r"""load_dataset\s*\(\s*['"]([^'"]+)['"]""")
_HUB_DOWNLOAD = re.compile(
    r"""(?:hf_hub_download|snapshot_download)\s*\([^)]*?repo_id\s*=\s*['"]([^'"]+)['"]""",
    re.DOTALL,
)
_URL = re.compile(r"""https?://[^\s'"<>)\\]+""")
# DataFlow 的 FileStorage 原生支持 `hf:` / `ms:` 前缀当 first_entry，dataflow-dev 技能的
# knowledge_base 还专门教了这个写法。它比 load_dataset 更顺手（一个字符串就完事），却完全不长
# 得像"下载"，是最容易被走、也最容易被漏掉的一条绕行路径。
_STORAGE_URI = re.compile(r"""['"](hf|ms):([^'"\s]+)['"]""")

# URL 是否「看起来在取数据」。LLM serving 的 base_url 也是 http，若一律算作外部数据引用，
# 每个节点都会被标成 unmanaged，这个标记就没有分辨力了。判据取两条：域名是公开数据托管站，
# 或路径以数据文件扩展名结尾。
_DATA_HOSTS = ("huggingface.co", "raw.githubusercontent.com", "github.com", "modelscope.cn")
_DATA_SUFFIXES = (".jsonl", ".json", ".csv", ".tsv", ".parquet", ".txt", ".zip", ".gz", ".tar")


def record_tool_uses(
    node_dir: str | Path, tool_uses: list[dict[str, Any]], phase: str = ""
) -> None:
    """把一次 Agent 运行的工具调用追加到 `node_dir/agent_tools.jsonl`。

    追加而非覆写：一个节点会经历生成 + 若干轮自修正，每轮都是独立的 Agent 运行，合起来才是
    这个节点完整的行为轨迹。`phase` 标出是哪一轮（generate / self_correct_review_1 / …）。

    永不抛异常：这是旁路记录，不该让写日志失败连累主循环。
    """
    if not tool_uses:
        return
    path = Path(node_dir) / TOOL_LOG_NAME
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for use in tool_uses:
                record = {"ts": ts, "phase": phase, **use}
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("写工具调用日志失败（%s）：%s", path, exc)


def truncate_arg(value: Any, maxlen: int = _MAX_ARG_CHARS) -> Any:
    """递归截断工具参数里的长字符串，保留结构。"""
    if isinstance(value, str):
        return value if len(value) <= maxlen else value[:maxlen] + f"…(共 {len(value)} 字符)"
    if isinstance(value, dict):
        return {str(k): truncate_arg(v, maxlen) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [truncate_arg(v, maxlen) for v in value[:20]]
    return value


def scan_node(
    node_dir: str | Path, data_links_dir: str | Path = "", write: bool = True
) -> dict[str, Any]:
    """扫描节点代码里的外部数据引用，返回（并可选落盘）溯源记录。

    `data_links_dir` 为空表示外部数据工具没开——此时**仍要扫**：Agent 绕过工具直连拉数据这条
    路和工具开没开无关，恰恰是工具关闭时才更需要知道有没有人翻墙。
    """
    node_path = Path(node_dir)
    record: dict[str, Any] = {
        "node": node_path.name,
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "static_scan",
        "scanned_files": [],
        "data_links": [],
        "direct_references": {
            "load_dataset": [], "hub_download": [], "storage_uri": [], "urls": [],
        },
        "other_urls": [],
        "unmanaged": False,
    }
    try:
        sources = _source_files(node_path)
    except OSError as exc:
        logger.warning("扫描节点 %s 的源码失败：%s", node_path, exc)
        return record

    links_prefix = str(Path(data_links_dir).resolve()) if data_links_dir else ""
    link_re = re.compile(re.escape(links_prefix) + r"""[\w./\\-]+""") if links_prefix else None

    linked: set[str] = set()
    for src in sources:
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        record["scanned_files"].append(str(src.relative_to(node_path)))
        if link_re:
            linked.update(link_re.findall(text))
        _collect(text, record["direct_references"], record["other_urls"])

    record["data_links"] = [_link_entry(p) for p in sorted(linked)]
    direct = record["direct_references"]
    for key in direct:
        direct[key] = sorted(set(direct[key]))
    record["other_urls"] = sorted(set(record["other_urls"]))
    record["unmanaged"] = any(direct.values())

    if record["unmanaged"]:
        logger.warning(
            "节点 %s 绕过下载工具直连外部数据（%s）；这些来源没有 .meta.json 溯源，已记入 %s。",
            node_path.name,
            "，".join(f"{k}={v}" for k, v in direct.items() if v),
            PROVENANCE_NAME,
        )
    elif record["data_links"]:
        logger.info(
            "节点 %s 引用了 %d 份 data_links 外部数据。", node_path.name, len(record["data_links"])
        )

    if write:
        try:
            (node_path / PROVENANCE_NAME).write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("写 %s 失败：%s", PROVENANCE_NAME, exc)
    return record


# ---- 内部实现 --------------------------------------------------------------
def _source_files(node_path: Path) -> list[Path]:
    """节点目录下 Agent 写的 .py 文件（pipeline.py + operators/ 等），跳过产物目录。"""
    out: list[Path] = []
    if not node_path.is_dir():
        return out
    for p in sorted(node_path.rglob("*.py")):
        if _SKIP_DIRS.intersection(p.relative_to(node_path).parts[:-1]):
            continue
        out.append(p)
    return out


def _collect(text: str, direct: dict[str, list[str]], other_urls: list[str]) -> None:
    direct["load_dataset"].extend(_LOAD_DATASET.findall(text))
    direct["hub_download"].extend(_HUB_DOWNLOAD.findall(text))
    direct["storage_uri"].extend(f"{scheme}:{ref}" for scheme, ref in _STORAGE_URI.findall(text))
    for url in _URL.findall(text):
        (direct["urls"] if _looks_like_data_url(url) else other_urls).append(url)


def _looks_like_data_url(url: str) -> bool:
    lowered = url.lower().rstrip("/")
    if any(host in lowered for host in _DATA_HOSTS):
        return True
    return any(lowered.endswith(suffix) for suffix in _DATA_SUFFIXES)


def _link_entry(path: str) -> dict[str, Any]:
    """data_links 文件 + 它的 .meta.json 溯源（缺失时显式标出，而不是静默给空）。"""
    p = Path(path)
    meta_path = p.with_suffix(".meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = None
    entry: dict[str, Any] = {"path": path, "exists": p.exists()}
    if meta is None:
        entry["meta_missing"] = True
    else:
        entry["meta"] = meta
    return entry
