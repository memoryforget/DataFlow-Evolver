"""ReviewAgent：Phase-1 快反馈 / Tier-1 打分器。

历史职责：训练前抽样评审生成数据集的 schema 符合度与任务相关性，作为「是否放行训练」的门槛。

PostTrainBench 适配后升级为 **Tier-1 复合打分器**（1 次 LLM 调用 + 纯代码硬指标）：
- 五维 LLM-judge（各 0-1）：correctness / relevance / difficulty / diversity / schema。
- 固定权重复合分：``llm_composite = 0.35·correctness + 0.20·relevance
  + 0.15·difficulty + 0.15·diversity + 0.15·schema``。
- 硬指标惩罚（扫全份文件，不花 LLM）：``dup_rate`` / ``null_rate`` / ``parse_valid_rate``
  / ``contamination_rate``，``tier1 = llm_composite × (1-dup)(1-null)(1-contam) × parse_valid``。
  其中 ``dup_rate`` 取「整记录哈希」与「训练字段级」两种口径的较大值——只看整记录会漏掉
  同一道题配不同措辞 CoT 的情形（详见 ``_scan_hard_metrics``）；``contamination_rate``
  是与 benchmark 测试集的 n-gram 重合率，超阈值直接判 ``passed=False``（见 core/decontamination.py）。
- 抽样从「取前 N 行」改为**随机蓄水池采样**（一趟遍历、O(k) 内存、全份等概率）。
- ``passed`` 门槛保留（schema/relevance 阈值），用于剔除垃圾节点。

合规：五维 + 硬指标只看我们**自己合成**的数据。唯一"接触" benchmark test 的是去污染扫描，
且只做 n-gram 命中判定、不读分数、不把测试内容回流给任何 Agent——它的作用恰恰是阻止泄露。
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

from dataflow_evolver.agents.base import AgentABC
from dataflow_evolver.core.decontamination import ContaminationIndex
from dataflow_evolver.core.schemas import ReviewResult, TaskProfile
from dataflow_evolver.llm.prompts import FAILURE_SIGNALS, build_review_prompt

# Tier-1 复合分的固定权重（对应设计文档 DEITA/IFD/CoT-Self-Instruct/InsTag 的经验分工）。
DEFAULT_REVIEW_WEIGHTS: dict[str, float] = {
    "correctness": 0.35,
    "relevance": 0.20,
    "difficulty": 0.15,
    "diversity": 0.15,
    "schema": 0.15,
}


class ReviewAgent(AgentABC):
    name = "review"

    def __init__(
        self,
        serving,
        schema_threshold: float = 0.6,
        relevance_threshold: float = 0.6,
        sample_size: int = 80,
        max_retries: int = 2,
        weights: dict[str, float] | None = None,
        hard_metrics: bool = True,
        seed: int | None = None,
        contamination: ContaminationIndex | None = None,
        null_threshold: float = 0.5,
    ) -> None:
        super().__init__(serving, max_retries=max_retries)
        self.schema_threshold = schema_threshold
        self.relevance_threshold = relevance_threshold
        # 关键字段空缺率的一票否决线。字段名整体对不上时空缺率≈1，取 0.5 能果断拦住，
        # 又不会因个别行缺字段就毙掉整个节点（那种情况仍按比例扣 tier1）。
        self.null_threshold = null_threshold
        self.sample_size = sample_size
        self.weights = _normalize_weights(weights)
        self.hard_metrics = hard_metrics
        self.seed = seed
        # 未配置基准参考集时是一个 disabled 的空索引，扫描恒返回干净——调用点无需分支。
        self.contamination = contamination or ContaminationIndex()

    def build_prompt(self, profile: TaskProfile, samples_text: str) -> str:
        return build_review_prompt(profile, samples_text)

    def compute_composite(self, result: ReviewResult) -> float:
        """五维加权复合分（仅依赖 LLM 维度与固定权重）。"""
        w = self.weights
        return (
            w["correctness"] * result.correctness_score
            + w["relevance"] * result.relevance_score
            + w["difficulty"] * result.difficulty_score
            + w["diversity"] * result.diversity_score
            + w["schema"] * result.schema_score
        )

    def parse(self, raw: dict) -> ReviewResult:
        """把 LLM 返回的五维分解析进 ReviewResult（尚未叠加硬指标惩罚）。"""
        schema_score = _clip01(raw.get("schema_score", 0.0))
        relevance_score = _clip01(raw.get("relevance_score", 0.0))
        # 新增三维：缺省回退保证旧 prompt / 部分返回也不崩。
        correctness_score = _clip01(raw.get("correctness_score", 0.0))
        difficulty_score = _clip01(raw.get("difficulty_score", 0.0))
        diversity_score = _clip01(raw.get("diversity_score", 0.0))
        passed = (
            schema_score >= self.schema_threshold
            and relevance_score >= self.relevance_threshold
        )
        # ANDES F1-F7 失败信号：只规整为合法编号并去重，**不并入 issues**。
        # 两者是异质数据——issues 是针对本份数据的具体证据，信号是标准化标签。混进一个列表后，
        # 每个消费点都得再把它们过滤出来（曾为此写了共享过滤函数 + 属性 + 四处调用）；
        # 保持分离后，需要信号文本的渲染点自己查 FAILURE_SIGNALS 拼即可。
        signals = _normalize_signals(raw.get("failure_signals", []))
        issues = list(raw.get("issues", []))
        result = ReviewResult(
            schema_score=schema_score,
            relevance_score=relevance_score,
            passed=passed,
            issues=issues,
            correctness_score=correctness_score,
            difficulty_score=difficulty_score,
            diversity_score=diversity_score,
            failure_signals=signals,
        )
        result.llm_composite = self.compute_composite(result)
        # 未叠加硬指标前，tier1 先等于复合分（review() 会用硬指标覆盖）。
        result.tier1_score = result.llm_composite
        return result

    def apply_hard_metrics(
        self, result: ReviewResult, dataset_path: str, profile: TaskProfile
    ) -> ReviewResult:
        """扫全份文件得硬指标并叠加惩罚，写回 result 并返回。"""
        required = list(profile.target_schema.keys()) if profile.target_schema else []
        metrics = _scan_hard_metrics(dataset_path, required)
        result.dup_rate = metrics.dup_rate
        result.null_rate = metrics.null_rate
        result.parse_valid_rate = metrics.parse_valid_rate
        if metrics.field_dup_rate > metrics.record_dup_rate:
            self.logger.info(
                "重复率由字段 %s 主导：字段级 %.3f > 整记录 %.3f"
                "（同一内容配不同措辞时整记录哈希会失效）",
                metrics.dup_field,
                metrics.field_dup_rate,
                metrics.record_dup_rate,
            )
        if metrics.total == 0:
            # 空文件：复合分再高也没有可训练数据，tier1 直接归零。
            result.tier1_score = 0.0
            return result

        # schema 契约：字段名对不上时，训练脚本会把整份数据丢光并非零退出，而那时 GPU 已经
        # 烧完，报错还会被记成"训练脚本非零退出"，看不出根因是字段名。这里用纯代码提前拦下，
        # 并把缺失的字段名写进 issues——评审自修正循环据此就能改对，不必等训练。
        if required and metrics.null_rate >= self.null_threshold:
            result.passed = False
            result.issues.append(
                f"产出字段与下游训练脚本的契约不符：要求 {required}，"
                f"实测关键字段空缺率 {metrics.null_rate:.1%}（上限 {self.null_threshold:.1%}）。"
                f"请让流水线最终产出**恰好这些字段名**（可用算子做字段映射），"
                f"字段名不同的话训练脚本会一条都读不到。"
            )

        # 去污染：与 benchmark 测试集重合的数据不是"质量差"，是把评测答案抄进了训练集。
        # 既按比例打折（少量疑似重合可能是假阳，不该一票否决），又设硬门槛（超过阈值就是
        # 在刷分，必须判不通过并把证据喂回 Agent，否则搜索会持续往这个方向走）。
        contam = self.contamination.scan(dataset_path)
        result.contamination_rate = contam.rate
        if contam.hits:
            self.logger.warning(
                "数据与基准测试集重合 %d/%d（%.2f%%）", contam.hits, contam.total, contam.rate * 100
            )
            if contam.rate > self.contamination.max_rate:
                result.passed = False
                result.issues.append(
                    f"检出与评测基准测试集重合 {contam.rate:.1%}（上限 "
                    f"{self.contamination.max_rate:.1%}）：这是数据泄露。命中片段示例："
                    f"{'；'.join(contam.examples)}。请去掉直接取自基准测试集的数据来源。"
                )

        result.tier1_score = (
            result.llm_composite
            * (1.0 - metrics.dup_rate)
            * (1.0 - metrics.null_rate)
            * (1.0 - contam.rate)
            * metrics.parse_valid_rate
        )
        return result

    def review(
        self,
        dataset_path: str,
        profile: TaskProfile,
        evidence_path: str | None = None,
        phase: str = "search",
    ) -> ReviewResult:
        """随机抽样 → LLM 五维打分 → 硬指标惩罚 → ReviewResult。失败时保守返回未通过。

        `evidence_path`：给定则把本次评审的证据（抽样原文 + LLM 原始响应 + 解析结果）**追加**
        写入该 JSONL。tier1 是搜索期唯一的 MCTS reward，但过去只留下一个标量分：抽样和原始
        响应用完即弃，事后既无法判断这个分是否可信，也无法离线复算/换权重重打。
        `phase`：区分同一节点的多次评审（首评 / 自纠后重评 / Tier-2 放大后重评）。
        """
        samples_text = _sample_jsonl(dataset_path, self.sample_size, seed=self.seed)
        if not samples_text:
            result = ReviewResult(0.0, 0.0, False, ["数据集为空或不可读"])
            self._dump_evidence(evidence_path, phase, dataset_path, "", None, result)
            return result
        prompt = self.build_prompt(profile, samples_text)
        raw = self.run_json(prompt)
        if raw is None:
            result = ReviewResult(0.0, 0.0, False, ["评审 LLM 未返回有效 JSON"])
            self._dump_evidence(evidence_path, phase, dataset_path, samples_text, None, result)
            return result
        result = self.parse(raw)
        if self.hard_metrics:
            self.apply_hard_metrics(result, dataset_path, profile)
        self.logger.info(
            "Review：corr=%.2f rel=%.2f diff=%.2f div=%.2f schema=%.2f "
            "| dup=%.2f null=%.2f parse=%.2f | composite=%.3f tier1=%.3f passed=%s | signals=%s",
            result.correctness_score, result.relevance_score, result.difficulty_score,
            result.diversity_score, result.schema_score,
            result.dup_rate, result.null_rate, result.parse_valid_rate,
            result.llm_composite, result.tier1_score, result.passed,
            ",".join(result.failure_signals) or "-",
        )
        self._dump_evidence(evidence_path, phase, dataset_path, samples_text, raw, result)
        return result

    def _dump_evidence(
        self,
        evidence_path: str | None,
        phase: str,
        dataset_path: str,
        samples_text: str,
        parsed: dict | None,
        result: ReviewResult,
    ) -> None:
        """把一次评审的完整证据追加为一行 JSON。落盘失败绝不影响主流程。

        用**追加**而非覆盖：同一节点会被评审多次（首评、自纠后重评、Tier-2 放大后重评），
        覆盖会丢掉自纠轨迹——而"改了之后哪一维真的涨了"正是最需要留存的证据。
        """
        if not evidence_path:
            return
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "dataset_path": dataset_path,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "weights": self.weights,
            "samples_jsonl": samples_text,     # 抽样原文：LLM 实际看到的字节，本身即合法 JSONL
            "raw_response": self.last_raw_response,
            "parsed": parsed,                  # 抽取后的 JSON（None = 未返回有效 JSON）
            "result": asdict(result),
        }
        try:
            path = Path(evidence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self.logger.warning("评审证据落盘失败（不影响主流程）：%s", exc)


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """合并用户权重与默认权重（缺项回退默认）；不强制归一，保持固定权重语义。"""
    merged = dict(DEFAULT_REVIEW_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in merged:
                merged[k] = float(v)
    return merged


def _normalize_signals(raw_signals) -> list[str]:
    """把 LLM 返回的 failure_signals 规整为合法编号（F1-F7）、大写、去重、保序。"""
    if not isinstance(raw_signals, (list, tuple)):
        return []
    out: list[str] = []
    for x in raw_signals:
        sid = str(x).strip().upper()
        if sid in FAILURE_SIGNALS and sid not in out:
            out.append(sid)
    return out


def _clip01(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _sample_jsonl(path: str, n: int, seed: int | None = None) -> str:
    """随机蓄水池采样：一趟遍历、O(n) 内存、全份非空行等概率抽 n 条。

    相比旧的「取前 N 行」，蓄水池采样避免只看到数据集开头（生成类算子常按类别/难度
    顺序产出，前 N 行有偏），使 Tier-1 打分更能代表整份数据。
    """
    p = Path(path)
    if not p.exists():
        return ""
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = line
    return "\n".join(reservoir)


class HardMetrics(NamedTuple):
    """``_scan_hard_metrics`` 的返回值。dup_rate 是对外口径，另两个 dup 仅供日志溯源。"""

    dup_rate: float
    record_dup_rate: float
    field_dup_rate: float
    dup_field: str
    null_rate: float
    parse_valid_rate: float
    total: int


# 内容字段判定门槛：中位长度低于此值的一律当标签字段看待。answer 常常只是一个数字、
# difficulty_level 只有 easy/medium/hard，这类字段天然大量重复，算进重复率会把每份数据集
# 都判成重灾区。40 字符落在标签字段（个位数长度）和题干/推理链（数百字符）之间的空档里。
CONTENT_FIELD_MIN_MEDIAN_LEN = 40

# 求中位数需要留样本。只留前 N 条：区分「3 字符的答案」和「300 字符的题干」根本不需要
# 无偏样本，而阈值离两种量级都很远，前缀偏置翻不动这个判定。
_LENGTH_SAMPLE_CAP = 4096


def _scan_hard_metrics(path: str, required_fields: list[str]) -> HardMetrics:
    """扫全份 jsonl 得硬指标。

    - dup_rate: 重复占比，取 ``max(整记录哈希重复率, 训练字段级重复率)``。
      整记录哈希只抓「整行一模一样」，会漏掉同一道题配不同措辞 CoT 的情形——两行哈希不同，
      但对模型而言是同一份监督信号。实测某份 4000 条数据整记录只报 0.06，而 question
      字段唯一率仅 0.518。反过来字段级也补不了整记录：题干各异而整行重复同样是重复。
      两种口径互不覆盖，所以取较大值。
    - null_rate: 关键字段空缺率（字段级）= 空缺槽位 / (可解析行数 × 字段数)；无 required
      字段时为 0。
    - parse_valid_rate: 合法 JSON 对象行 / total。
    """
    p = Path(path)
    if not p.exists():
        return HardMetrics(0.0, 0.0, 0.0, "", 0.0, 1.0, 0)

    total = 0
    parseable = 0
    hashes: set[str] = set()
    null_slots = 0
    field_slots = 0
    # 只对 required_fields 逐字段留哈希，字段数有界，仍是一趟流式扫描。
    field_hashes: dict[str, set[str]] = {fld: set() for fld in required_fields}
    field_lengths: dict[str, list[int]] = {fld: [] for fld in required_fields}

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = None
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                obj = None

            if isinstance(obj, dict):
                parseable += 1
                canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False)
                hashes.add(hashlib.md5(canonical.encode("utf-8")).hexdigest())
                for fld in required_fields:
                    field_slots += 1
                    value = obj.get(fld)
                    if _is_empty(value):
                        null_slots += 1
                    text = _as_text(value)
                    field_hashes[fld].add(
                        hashlib.md5(text.encode("utf-8")).hexdigest()
                    )
                    if len(field_lengths[fld]) < _LENGTH_SAMPLE_CAP:
                        field_lengths[fld].append(len(text))
            else:
                hashes.add(hashlib.md5(line.encode("utf-8")).hexdigest())

    if total == 0:
        return HardMetrics(0.0, 0.0, 0.0, "", 0.0, 0.0, 0)

    record_dup = _clip01(1.0 - (len(hashes) / total))
    field_dup, dup_field = _worst_content_field_dup(
        field_hashes, field_lengths, parseable
    )
    parse_valid_rate = parseable / total
    null_rate = (null_slots / field_slots) if field_slots > 0 else 0.0
    return HardMetrics(
        dup_rate=max(record_dup, field_dup),
        record_dup_rate=record_dup,
        field_dup_rate=field_dup,
        dup_field=dup_field,
        null_rate=_clip01(null_rate),
        parse_valid_rate=_clip01(parse_valid_rate),
        total=total,
    )


def _worst_content_field_dup(
    field_hashes: dict[str, set[str]],
    field_lengths: dict[str, list[int]],
    rows: int,
) -> tuple[float, str]:
    """训练字段里塌陷最严重那一维的重复率，返回 (dup_rate, 字段名)。

    只看内容字段（中位长度 ≥ CONTENT_FIELD_MIN_MEDIAN_LEN），标签字段一律跳过。
    """
    if rows <= 0:
        return 0.0, ""
    worst = 0.0
    worst_field = ""
    for fld, hashes in field_hashes.items():
        lengths = field_lengths.get(fld) or []
        if not lengths or statistics.median(lengths) < CONTENT_FIELD_MIN_MEDIAN_LEN:
            continue
        dup = 1.0 - (len(hashes) / rows)
        if dup > worst:
            worst, worst_field = dup, fld
    return _clip01(worst), worst_field


def _as_text(value) -> str:
    """把字段值折成可哈希、可量长度的字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False
