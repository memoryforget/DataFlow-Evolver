"""ReviewAgent：Phase-1 快反馈 / Tier-1 打分器。

历史职责：训练前抽样评审生成数据集的 schema 符合度与任务相关性，作为「是否放行训练」的门槛。

PostTrainBench 适配后升级为 **Tier-1 复合打分器**（1 次 LLM 调用 + 纯代码硬指标）：
- 五维 LLM-judge（各 0-1）：correctness / relevance / difficulty / diversity / schema。
- 固定权重复合分：``llm_composite = 0.35·correctness + 0.20·relevance
  + 0.15·difficulty + 0.15·diversity + 0.15·schema``。
- 硬指标惩罚（扫全份文件，不花 LLM）：``dup_rate`` / ``null_rate`` / ``parse_valid_rate``，
  ``tier1 = llm_composite * (1-dup_rate) * (1-null_rate) * parse_valid_rate``。
- 抽样从「取前 N 行」改为**随机蓄水池采样**（一趟遍历、O(k) 内存、全份等概率）。
- ``passed`` 门槛保留（schema/relevance 阈值），用于剔除垃圾节点。

合规：五维 + 硬指标只看我们**自己合成**的数据，不接触任何 benchmark test。
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from dataflow_evolver.agents.base import AgentABC
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
    ) -> None:
        super().__init__(serving, max_retries=max_retries)
        self.schema_threshold = schema_threshold
        self.relevance_threshold = relevance_threshold
        self.sample_size = sample_size
        self.weights = _normalize_weights(weights)
        self.hard_metrics = hard_metrics
        self.seed = seed

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
        # ANDES F1-F7 失败信号：规整为合法编号并去重，再把人类可读描述并入 issues
        # 供 PipelineAgent 自纠 prompt 复用（不参与打分）。
        signals = _normalize_signals(raw.get("failure_signals", []))
        issues = list(raw.get("issues", []))
        issues.extend(f"[{sid}] {FAILURE_SIGNALS[sid]}" for sid in signals)
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
        dup_rate, null_rate, parse_valid_rate, total = _scan_hard_metrics(
            dataset_path, required
        )
        result.dup_rate = dup_rate
        result.null_rate = null_rate
        result.parse_valid_rate = parse_valid_rate
        if total == 0:
            # 空文件：复合分再高也没有可训练数据，tier1 直接归零。
            result.tier1_score = 0.0
            return result
        result.tier1_score = (
            result.llm_composite
            * (1.0 - dup_rate)
            * (1.0 - null_rate)
            * parse_valid_rate
        )
        return result

    def review(self, dataset_path: str, profile: TaskProfile) -> ReviewResult:
        """随机抽样 → LLM 五维打分 → 硬指标惩罚 → ReviewResult。失败时保守返回未通过。"""
        samples_text = _sample_jsonl(dataset_path, self.sample_size, seed=self.seed)
        if not samples_text:
            return ReviewResult(0.0, 0.0, False, ["数据集为空或不可读"])
        prompt = self.build_prompt(profile, samples_text)
        raw = self.run_json(prompt)
        if raw is None:
            return ReviewResult(0.0, 0.0, False, ["评审 LLM 未返回有效 JSON"])
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
        return result


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


def _scan_hard_metrics(
    path: str, required_fields: list[str]
) -> tuple[float, float, float, int]:
    """扫全份 jsonl，返回 (dup_rate, null_rate, parse_valid_rate, total_nonempty_rows)。

    - dup_rate: 内容哈希重复占比 = 1 - unique/total（对可解析行用规范化 JSON 哈希，
      不可解析行用原始串哈希）。
    - null_rate: 关键字段空缺率（字段级）= 空缺槽位 / (可解析行数 × 字段数)；无 required
      字段时为 0。
    - parse_valid_rate: 合法 JSON 对象行 / total。
    """
    p = Path(path)
    if not p.exists():
        return 0.0, 0.0, 1.0, 0

    total = 0
    parseable = 0
    hashes: set[str] = set()
    null_slots = 0
    field_slots = 0

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
                if required_fields:
                    for fld in required_fields:
                        field_slots += 1
                        if _is_empty(obj.get(fld)):
                            null_slots += 1
            else:
                hashes.add(hashlib.md5(line.encode("utf-8")).hexdigest())

    if total == 0:
        return 0.0, 0.0, 0.0, 0

    dup_rate = 1.0 - (len(hashes) / total)
    parse_valid_rate = parseable / total
    null_rate = (null_slots / field_slots) if field_slots > 0 else 0.0
    return _clip01(dup_rate), _clip01(null_rate), _clip01(parse_valid_rate), total


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False
