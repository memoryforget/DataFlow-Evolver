"""所有核心数据结构（dataclass）。先行严格定义，供各 Agent / Wrapper / 树搜索共享。

对应 TODO.md §4。这些类型刻意保持"纯数据"，不含业务逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskProfile:
    """TaskProfilerAgent 输出，全局唯一。任务描述的形式化数据需求。"""

    task_description: str
    target_schema: dict[str, str]          # 期望字段→类型，如 {"instruction": "str", "answer": "str"}
    quality_criteria: list[str]
    downstream_task: str                   # 例如 "math-QA-SFT"


@dataclass
class PipelineConfig:
    """PipelineAgent 产物：一个节点对应一份流水线配置 + 完整可执行代码。"""

    operators: list[dict[str, Any]]        # [{name, params}, ...]
    field_flow: str                        # "text -> chunk -> qa_pairs"
    code: str                              # 完整可执行的 DataFlow 流水线 Python 源码
    rationale: str
    recipe: dict[str, Any] = field(default_factory=dict)  # 已由框架校验并冻结的数据设计契约
    recipe_hash: str = ""                  # canonical recipe 的 SHA-256，便于审计/续跑


@dataclass
class ReviewResult:
    """Phase-1 快反馈：训练前的数据评审结果。

    历史上仅有 schema/relevance 两维 + passed 门槛。PostTrainBench 适配把它升级为
    Tier-1 复合打分器：新增 correctness/difficulty/diversity 三个 LLM 维度，以及纯代码
    扫描得到的硬指标（dup_rate/null_rate/parse_valid_rate），并合成 tier1_score 作为
    MCTS 每节点 reward。所有新增字段均带默认值，旧的两维调用保持兼容。
    """

    schema_score: float                    # 0-1
    relevance_score: float                 # 0-1
    passed: bool
    issues: list[str] = field(default_factory=list)

    # --- Tier-1 升级：新增 LLM 维度（0-1，缺省 0 便于向后兼容）---
    correctness_score: float = 0.0         # 解答/标签正确性、推理-答案一致性
    difficulty_score: float = 0.0          # 对目标模型的非平凡且可学难度（中间难度带最高）
    diversity_score: float = 0.0           # 抽样批内多样性、非近重复

    # --- 硬指标（纯代码扫全份文件，不花 LLM）---
    dup_rate: float = 0.0                  # 重复占比 = max(整记录哈希重复, 训练字段级重复)
    null_rate: float = 0.0                 # schema 关键字段空缺率
    parse_valid_rate: float = 1.0          # 可解析（合法 JSON + 字段齐全）占比
    contamination_rate: float = 0.0        # 与 benchmark 测试集的 n-gram 重合率（未配参考集时恒 0）
    normalized_dup_rate: float = 0.0       # 规范化内容近重率（去格式/数字表面差异）
    structural_dup_rate: float = 0.0       # 相同 operation graph/archetype 的集中度
    coverage_score: float = 1.0            # recipe 预期 coverage bucket 的覆盖率
    largest_bucket_fraction: float = 0.0   # 最大 bucket 占比，监测单一模板坍塌
    external_fraction: float = 0.0         # 最终数据中声明为 external 的比例
    synthetic_fraction: float = 0.0        # 最终数据中声明为 synthetic 的比例
    source_id_valid_rate: float = 1.0      # provenance source_id 合法率
    seed_link_rate: float = 1.0            # hybrid 合成行追溯到保留 external seed 的比例
    source_compliance_score: float = 1.0   # mode 来源/比例/追溯契约的确定性复合分

    # --- 复合分：作为 Tier-1 reward 回传 MCTS ---
    llm_composite: float = 0.0             # 五维加权（固定权重）
    tier1_score: float = 0.0               # llm_composite × (1-dup)(1-null)(1-contam) × parse_valid

    # --- 失败信号（借鉴 ANDES F1-F7 坏味道分类；仅观测/可解释，不参与打分）---
    failure_signals: list[str] = field(default_factory=list)  # 命中的信号编号，如 ["F3", "F5"]


@dataclass
class BadCase:
    """来自真实下游评估的单条失败样例。"""

    input: str
    prediction: str                        # 模型完整输出（不截断，供诊断看清推理与最终答案）
    ground_truth: str
    error_type: str | None = None
    extracted_answer: str | None = None    # 从 prediction 解析出的最终答案（与 ground_truth 直接对照）


@dataclass
class TrainResult:
    """Phase-2 慢反馈：DownstreamTrainerWrapper 解析外部脚本结果得到。"""

    score: float
    metric_name: str
    bad_cases: list[BadCase] = field(default_factory=list)
    raw_log_path: str = ""
    failed: bool = False                   # 训练/评估系统级失败（OOM/timeout/非零退出）
    error_msg: str | None = None


@dataclass
class DiagnosticReport:
    """DiagnosticAgent 产物：回写节点，指导下一轮有的放矢的变异。"""

    summary: str                           # 例："模型在含分数的几何题上失败"
    failure_patterns: list[str] = field(default_factory=list)
    actionable_suggestions: list[str] = field(default_factory=list)
