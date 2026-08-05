"""Turn the user-provided raw corpus into a deterministic entry and prompt preview.

不做成配置项，是因为「配了 grounded 却没语料」「放了语料却按 scratch 跑」两种组合都只可能是
误配：前者必崩，后者白白浪费了用户准备的数据。让目录内容本身作为唯一事实来源，就没有可漂移的
第二处状态。

预览是**尽力而为**的：语料格式千奇百怪，解析失败只会少一段 prompt 上下文，绝不该让整轮搜索起不来。
The review-only loop requires a real structured entry and fails fast when it is absent.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from dataflow_evolver.utils.logging import get_logger

logger = get_logger("core.input_corpus")

# DataFlow FileStorage 能直接作为 first_entry 读取的格式；其余文件只作为「算子可自行读取的素材」列出。
ENTRY_SUFFIX_PRIORITY = (".jsonl", ".json", ".csv", ".parquet")
TEXT_SUFFIXES = (".txt", ".md")

PREVIEW_ROWS = 3            # 预览行数：够看清字段与风格，又不至于挤占 prompt
PREVIEW_VALUE_MAXLEN = 200  # 单字段截断长度：长文档字段（如整篇正文）不该淹没其它字段
ROW_COUNT_CAP = 200_000     # 行数统计上限：超大语料只报「≥上限」，避免启动时全量扫盘
JSON_PARSE_MAX_BYTES = 32 * 1024 * 1024   # 整体式 .json 超过此大小不解析，只报文件信息


@dataclass(frozen=True)
class CorpusFile:
    """输入目录里的一个文件及其尽力而为的预览。"""

    path: Path
    rel: str                            # 相对输入根目录的路径，用于展示
    size_bytes: int
    rows: int | None = None             # None = 未知（不可解析格式或解析失败）
    rows_capped: bool = False           # True 表示行数统计触顶，实际 ≥ rows
    fields: tuple[str, ...] = ()
    preview_lines: tuple[str, ...] = ()

    @property
    def is_entry_candidate(self) -> bool:
        return self.path.suffix.lower() in ENTRY_SUFFIX_PRIORITY

    def describe(self) -> str:
        parts = [self.rel, _human_size(self.size_bytes)]
        if self.rows is not None:
            parts.append(f"{'≥' if self.rows_capped else ''}{self.rows} 行")
        if self.fields:
            parts.append("字段: " + ", ".join(self.fields))
        return "  ".join(parts)


@dataclass(frozen=True)
class InputCorpus:
    """一次运行的固定输入契约：入口文件、全部素材文件及 prompt 预览。"""

    entry_path: str
    root: Path | None = None
    files: tuple[CorpusFile, ...] = ()
    entry_file: CorpusFile | None = field(default=None)

    @property
    def grounded(self) -> bool:
        """是否已绑定真实语料。轻量入口要求此值始终为真。"""
        return self.entry_file is not None

    def sample_text(self) -> str | None:
        """返回入口文件的前几行；无预览时退回其他素材文件。"""
        for f in ((self.entry_file,) if self.entry_file else ()) + self.files:
            if f is not None and f.preview_lines:
                return "\n".join(f.preview_lines)
        return None

    def prompt_block(self) -> str:
        """渲染注入 PipelineAgent prompt 的输入语料段。"""
        if not self.grounded:
            return ""
        lines = [f"\n【输入语料】目录 {self.root}，流水线的 Entry file 即下表中标注 ← 的那个："]
        for f in self.files:
            mark = "   ← Entry file" if f is self.entry_file else ""
            lines.append(f"  - {f.describe()}{mark}")
        preview = self.entry_file.preview_lines if self.entry_file else ()
        if preview:
            lines.append(f"  Entry file 前 {len(preview)} 行（长字段已截断，仅供了解字段与风格）：")
            lines.extend(f"    {line}" for line in preview)
        others = [f for f in self.files if f is not self.entry_file]
        if others:
            lines.append(
                "  其余文件不会被自动读取；确有需要时在自定义算子里按上面的路径自行读取。"
            )
        return "\n".join(lines) + "\n"


def discover(
    input_dir: str | Path | None,
    entry_name: str = "",
) -> InputCorpus:
    """Scan a required raw-data directory and choose one deterministic entry.

    `entry_name`：多文件时显式指定入口（相对输入目录的路径）。不指定则按
    `ENTRY_SUFFIX_PRIORITY` + 路径字典序确定性地选一个，并在文件多于一个时告警——
    静默挑一个再让用户去日志里猜是哪个，比多打一行告警糟糕得多。
    """
    if not input_dir:
        raise ValueError("project.input_dir 不能为空：review-only loop 需要固定 raw data")
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw data 目录不存在：{root}")

    files = tuple(_scan(root))
    if not files:
        raise ValueError(f"raw data 目录为空：{root}")

    entry = _pick_entry(files, entry_name, root)
    if entry is None:
        raise ValueError(
            f"raw data 目录 {root} 中没有可作为 Entry file 的结构化文件："
            f"{', '.join(ENTRY_SUFFIX_PRIORITY)}"
        )

    logger.info(
        "固定 raw corpus：Entry file=%s（%s），目录内共 %d 个文件。",
        entry.rel, entry.describe(), len(files),
    )
    return InputCorpus(
        entry_path=str(entry.path), root=root, files=files, entry_file=entry
    )


def _scan(root: Path) -> list[CorpusFile]:
    out: list[CorpusFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        rel = str(path.relative_to(root))
        try:
            size = path.stat().st_size
        except OSError:
            continue
        out.append(_profile_file(path, rel, size))
    return out


def _pick_entry(files: tuple[CorpusFile, ...], entry_name: str, root: Path) -> CorpusFile | None:
    if entry_name:
        wanted = str(Path(entry_name))
        for f in files:
            if f.rel == wanted:
                return f
        raise FileNotFoundError(
            f"配置指定的输入入口文件 {entry_name} 不在 {root} 中；"
            f"该目录下的文件有：{', '.join(f.rel for f in files)}"
        )
    candidates = [f for f in files if f.is_entry_candidate]
    if not candidates:
        return None
    candidates.sort(key=lambda f: (ENTRY_SUFFIX_PRIORITY.index(f.path.suffix.lower()), f.rel))
    entry = candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "输入目录有 %d 个可作入口的数据文件，按「格式优先级+字典序」选用 %s；"
            "若不是想要的那个，请用 project.input_entry 显式指定。",
            len(candidates), entry.rel,
        )
    return entry


# ---- 各格式的尽力而为预览 --------------------------------------------------
def _profile_file(path: Path, rel: str, size: int) -> CorpusFile:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            return _profile_jsonl(path, rel, size)
        if suffix == ".json":
            return _profile_json(path, rel, size)
        if suffix == ".csv":
            return _profile_csv(path, rel, size)
        if suffix in TEXT_SUFFIXES:
            return _profile_text(path, rel, size)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.warning("预览输入文件 %s 失败（%s），仅按文件名列出。", rel, exc)
    return CorpusFile(path=path, rel=rel, size_bytes=size)


def _profile_jsonl(path: Path, rel: str, size: int) -> CorpusFile:
    rows = 0
    capped = False
    sample: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            if len(sample) < PREVIEW_ROWS:
                obj = _loads(line)
                if obj is not None:
                    sample.append(obj)
            if rows >= ROW_COUNT_CAP:
                capped = True
                break
    return _build(path, rel, size, rows, capped, sample)


def _profile_json(path: Path, rel: str, size: int) -> CorpusFile:
    if size > JSON_PARSE_MAX_BYTES:
        logger.info("输入文件 %s 过大（%s），跳过预览解析。", rel, _human_size(size))
        return CorpusFile(path=path, rel=rel, size_bytes=size)
    obj = _loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(obj, list):
        sample = [r for r in obj[:PREVIEW_ROWS] if isinstance(r, dict)]
        return _build(path, rel, size, len(obj), False, sample)
    if isinstance(obj, dict):
        return _build(path, rel, size, 1, False, [obj])
    return CorpusFile(path=path, rel=rel, size_bytes=size)


def _profile_csv(path: Path, rel: str, size: int) -> CorpusFile:
    rows = 0
    capped = False
    sample: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            if len(sample) < PREVIEW_ROWS:
                sample.append({k: v for k, v in row.items() if k is not None})
            if rows >= ROW_COUNT_CAP:
                capped = True
                break
    return _build(path, rel, size, rows, capped, sample)


def _profile_text(path: Path, rel: str, size: int) -> CorpusFile:
    """纯文本没有字段可言，预览取前几行原文即可。"""
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(_truncate(line.rstrip("\n")))
            if len(lines) >= PREVIEW_ROWS:
                break
    return CorpusFile(
        path=path, rel=rel, size_bytes=size, preview_lines=tuple(lines)
    )


def _build(
    path: Path, rel: str, size: int, rows: int, capped: bool, sample: list[dict]
) -> CorpusFile:
    fields: list[str] = []
    for row in sample:                      # 并集而非首行：稀疏字段也该被 Agent 看见
        for k in row:
            if k not in fields:
                fields.append(str(k))
    preview = tuple(
        json.dumps({str(k): _truncate_value(v) for k, v in row.items()}, ensure_ascii=False)
        for row in sample
    )
    return CorpusFile(
        path=path, rel=rel, size_bytes=size, rows=rows, rows_capped=capped,
        fields=tuple(fields), preview_lines=preview,
    )


def _loads(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _truncate_value(value):
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, list):
        return [_truncate_value(v) for v in value[:3]]
    if isinstance(value, dict):
        return {str(k): _truncate_value(v) for k, v in list(value.items())[:5]}
    return value


def _truncate(text: str) -> str:
    if len(text) <= PREVIEW_VALUE_MAXLEN:
        return text
    return text[:PREVIEW_VALUE_MAXLEN] + f"…(共 {len(text)} 字符)"


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"
