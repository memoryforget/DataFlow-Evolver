"""行集的读取与定量抽样：dev 题集与训练集共用的两件小事。

两处都要「把一个文件读成 dict 列表」和「取固定条数的确定性子集」。各写一份必然漂移——
抽样种子、是否保序、不足条数时的行为，只要有一处不同，两边就在不同口径下固定数据，
而这类差异既不报错也不进日志。
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def read_rows(path: str | Path) -> list[dict]:
    """读 jsonl 或 json 数组，统一成 dict 列表。无法解析的行跳过。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        rows = data if isinstance(data, list) else [data]
        return [r for r in rows if isinstance(r, dict)]
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def subsample(rows: list[dict], size: int, seed: int) -> list[dict]:
    """定量抽样。`size<=0` 或不足 size 条时返回全量。

    用 `sorted(sample(range(n), size))` 而不是 `sample(rows, size)`：保留源文件顺序，
    使结果在人眼比对与 diff 下可读，抽中的集合本身不受影响。
    """
    if size <= 0 or len(rows) <= size:
        return rows
    idx = sorted(random.Random(seed).sample(range(len(rows)), size))
    return [rows[i] for i in idx]


def dump_rows(rows: list[dict], path: str | Path) -> None:
    """按目标后缀写回：`.json` 写数组，其余按 jsonl 逐行写。

    保持与来源同一种容器格式——训练脚本按扩展名决定怎么读，换了格式它会直接读不出数据。
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return
    p.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
