"""训练集定量物化：把真正喂进训练脚本的数据切齐到固定条数。

为什么这件事必须由框架强制，而不是写在 prompt 里请 Agent 配合：

训练配方在框架外的一份脚本里冻结（epochs / lr / batch 全是常量，见 §2.5），而总优化步数是
`epochs × N / global_batch`。于是**数据条数直接决定算力预算**：一个多产三倍数据的节点白拿
三倍梯度步数，它分高了也说不清是数据更好，还是单纯训得更久。奖励归因就此失效，而 MCTS 会
如实地把这条捷径学走——"多产数据"是比"改进数据"容易得多的涨分方式，且与数据质量无关。

`DF_TARGET_SAMPLES` 只是注入给 pipeline 的一个**建议值**：算子可以不读它，过滤步骤会让实际
产出低于它，增强步骤会让它翻倍。想让"同规模"成立，只能在数据进训练脚本前切齐。

DataMaster 面对的是同一个局面——template 冻结、Red node 下载外部数据后训练集可任意膨胀——
但两端都没有约束，prompt 里只有一句"控制数据规模"，给的理由还是训练时间而非可比性。这里把
它落成代码（见 docs/design-decisions.md §2.5）。

**截断救不了反方向**：产出不足 cap 是同一个问题的另一半（欠训），同样污染比较，但补不出数据来。
只能记录并告警，让它在日志和 manifest 里可见，而不是安静地混进分数里。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dataflow_evolver.core import rows as rowlib
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("core.train_set")

DEFAULT_SEED = 42
MANIFEST_NAME = "train_set.manifest.json"
CAPPED_NAME = "train_set_capped"


@dataclass(frozen=True)
class TrainSet:
    """一次训练实际使用的数据及其与目标规模的偏差。"""

    path: str        # 交给训练脚本的文件（未截断时即 pipeline 原始产出）
    source: str      # pipeline 产出路径
    produced: int    # pipeline 实际产出条数
    used: int        # 训练实际使用条数
    cap: int         # 目标规模（0=不设限）
    seed: int

    @property
    def truncated(self) -> bool:
        return self.used < self.produced

    @property
    def short(self) -> bool:
        """产出不足目标规模——截断补不回来的那一半问题。"""
        return self.cap > 0 and self.produced < self.cap


def materialize(
    dataset_path: str | Path,
    out_dir: str | Path,
    cap: int,
    seed: int = DEFAULT_SEED,
) -> TrainSet:
    """把 `dataset_path` 切齐到至多 `cap` 条，返回该交给训练脚本的路径。

    只在确实超量时才写新文件；未超量则原样透传，manifest 照写——差额本身是要留痕的信息。
    manifest 落在 `out_dir`（训练输出目录），与 `run_manifest.json` 并排，事后一处就能核对
    "这个分数是多少条数据、什么训练配方跑出来的"。

    读不出行时不阻断：让训练脚本自己对空数据集报错，那里的信息比这里更具体。
    """
    src = Path(dataset_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        rows = rowlib.read_rows(src)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("训练集读取失败，原样交给训练脚本：%s（%s）", src, exc)
        return _record(out, TrainSet(str(src), str(src), 0, 0, int(cap), int(seed)))

    produced = len(rows)
    cap = int(cap)
    if cap <= 0 or produced <= cap:
        ts = TrainSet(str(src), str(src), produced, produced, cap, int(seed))
        if ts.short:
            logger.warning(
                "训练集产出不足目标规模：%d/%d 条（%.0f%%）。总步数随之偏低，该节点分数与"
                "足量节点不完全可比——截断补不回来，仅记录。",
                produced, cap, 100.0 * produced / cap,
            )
        return _record(out, ts)

    picked = rowlib.subsample(rows, size=cap, seed=int(seed))
    capped = out / f"{CAPPED_NAME}{src.suffix or '.jsonl'}"
    rowlib.dump_rows(picked, capped)
    logger.info(
        "训练集切齐：%d → %d 条（seed=%d）→ %s", produced, len(picked), seed, capped
    )
    return _record(
        out, TrainSet(str(capped), str(src), produced, len(picked), cap, int(seed))
    )


def _record(out_dir: Path, ts: TrainSet) -> TrainSet:
    """落 manifest。留痕失败不该影响训练本身，只告警。"""
    try:
        data = {**asdict(ts), "truncated": ts.truncated, "short": ts.short}
        (out_dir / MANIFEST_NAME).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("train_set manifest 落盘失败（不影响训练）：%s", exc)
    return ts
