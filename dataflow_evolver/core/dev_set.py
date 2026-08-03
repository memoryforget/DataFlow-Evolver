"""dev 评测集物化：把驱动 MCTS 的验证集固定成一份可审计的文件。

为什么这件事该由框架做，而不是丢给评测脚本：

1. **可比性**。奖励是节点之间比出来的，前提是所有节点在同一批题上打分。若抽样发生在每个
   节点的评测子进程里，任何人中途改一下抽样条数或种子，前后节点就在不同题集上比，MCTS 的
   排序被静默污染——不报错、日志里也看不出来。物化成一份文件后，"同一批题"从约定变成事实。
2. **可去污染**。dev 是搜索直接优化的目标，把 dev 题混进合成训练数据是最省力的刷分方式，
   比混 test 还有效（test 全程不参与 loop）。但只短暂存在于子进程内存里的题集无法作为
   去污染参考集，这个洞根本堵不上。落盘之后才扫得到。
3. **一处切分、处处复用**。每个评测脚本各写一套抽样（MATH 用 seed=42 抽 500、
   PostTrainBench 用 `--dev-limit`）必然各自漂移。物化后脚本退化成"在这个文件上评测"。

纪律本身取自 DataMaster，执行层不同：它把"所有节点必须评在同一份验证集上"写在 prompt 里，但没有
任何代码生成那份 `input/val.csv`，且同一份 prompt 给了"文件不存在就自己切"的兜底——于是默认情况下
每个节点各切各的。这里把它落成代码而不是约定（见 docs/design-decisions.md §4.1.1）。

另有一层结构差异：它的 val 从任务自带训练集里切出来，而我们没有可切的训练集——训练数据是每个节点
现合成的。所以我们固定的是**评测子集**，不是 train/val split。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dataflow_evolver.core import rows as rowlib
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("core.dev_set")

DEFAULT_SEED = 42
MANIFEST_NAME = "dev.manifest.json"
DEV_NAME = "dev.jsonl"


@dataclass(frozen=True)
class DevSet:
    """一份已物化的 dev 集及其来源指纹。"""

    path: str
    source: str
    source_sha: str
    seed: int
    size: int          # 请求条数；0 表示全量
    count: int         # 实际条数

    def recipe(self) -> dict:
        """决定这份 dev 集内容的三要素——比对它即可判断两次运行是否同口径。"""
        return {"source_sha": self.source_sha, "seed": self.seed, "size": self.size}


class DevSetChanged(RuntimeError):
    """workspace 里已有一份不同口径的 dev 集。继续跑会让新旧节点分数不可比。"""


def materialize(
    source: str, out_dir: str | Path, size: int = 0, seed: int = DEFAULT_SEED
) -> DevSet:
    """把 `source` 按 (size, seed) 固定成 `out_dir/dev.jsonl`，并落一份 manifest。

    幂等：已有同口径的 dev 集就直接复用（续跑必须拿到同一批题）。口径变了则抛
    `DevSetChanged`——静默改写会让 checkpoint 之前与之后的节点分数不可比，而这种污染
    事后无从追查。
    """
    src = Path(source).expanduser()
    if not src.is_file():
        raise FileNotFoundError(
            f"trainer.dev_set.source 指向的文件不存在：{src}\n"
            f"它是 MCTS 的奖励信号来源，缺了整轮搜索没有意义。\n"
            f"评测数据不入库，获取方式见 data/README.md。\n"
            f"（留空 source 只会跳过物化，不会让评测跑起来：评测脚本按契约不自行挑题，"
            f"拿不到 --dev-set 会直接拒绝。见 README「脚本负责什么、不负责什么」。）"
        )

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dev_path = out / DEV_NAME
    manifest_path = out / MANIFEST_NAME

    source_sha = _sha256(src)
    wanted = {"source_sha": source_sha, "seed": int(seed), "size": int(size)}

    existing = _read_manifest(manifest_path)
    if existing is not None and dev_path.is_file():
        if existing.recipe() == wanted:
            logger.info(
                "复用已物化的 dev 集：%s（%d 条，seed=%d）",
                dev_path, existing.count, existing.seed,
            )
            return existing
        raise DevSetChanged(
            f"{out} 下已有一份不同口径的 dev 集，继续跑会让新旧节点分数不可比。\n"
            f"  已有：{existing.recipe()}\n"
            f"  本次：{wanted}\n"
            f"确实要换口径的话，删掉 {manifest_path} 与 {dev_path}（旧节点分数随之作废），"
            f"或换一个 workspace 重新开一轮。"
        )

    rows = rowlib.read_rows(src)
    if not rows:
        raise ValueError(f"dev 集源文件没有可用数据行：{src}")

    picked = rowlib.subsample(rows, size=int(size), seed=int(seed))
    dev_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked), encoding="utf-8"
    )
    dev = DevSet(
        path=str(dev_path),
        source=str(src),
        source_sha=source_sha,
        seed=int(seed),
        size=int(size),
        count=len(picked),
    )
    manifest_path.write_text(
        json.dumps(asdict(dev), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "dev 集已物化：%s（自 %s 抽 %d/%d 条，seed=%d）",
        dev_path, src.name, len(picked), len(rows), seed,
    )
    return dev


def build(cfg_block, workspace: str | Path) -> DevSet | None:
    """按 `trainer.dev_set` 配置物化；未配 source 则返回 None（评测脚本沿用自身默认）。"""
    if cfg_block is None:
        return None
    source = str(cfg_block.get("source", "") or "")
    if not source:
        return None
    return materialize(
        source=source,
        out_dir=Path(workspace) / "eval",
        size=int(cfg_block.get("size", 0) or 0),
        seed=int(cfg_block.get("seed", DEFAULT_SEED)),
    )


# ---- 内部工具 --------------------------------------------------------------
def _read_manifest(path: Path) -> DevSet | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DevSet(**data)
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        logger.warning("dev 集 manifest 读取失败，将重新物化：%s", exc)
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
