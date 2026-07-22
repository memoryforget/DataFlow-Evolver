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


@dataclass
class ReviewResult:
    """Phase-1 快反馈：训练前的数据评审结果。"""

    schema_score: float                    # 0-1
    relevance_score: float                 # 0-1
    passed: bool
    issues: list[str] = field(default_factory=list)


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
