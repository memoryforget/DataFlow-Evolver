"""Core data contracts for the review-only feedback loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskProfile:
    task_description: str
    target_schema: dict[str, str]
    quality_criteria: list[str]


@dataclass
class PipelineConfig:
    operators: list[dict[str, Any]]
    field_flow: str
    code: str
    rationale: str


@dataclass
class DatasetQualityResult:
    """Deterministic dataset-distribution evidence produced by DAS/MMD."""

    enabled: bool = False
    status: str = "disabled"
    metric: str = "DAS"
    proxy_name: str = ""

    candidate_total: int = 0
    candidate_sample_size: int = 0
    proxy_sample_size: int = 0

    embedding_model: str = ""
    embedding_dim: int = 0
    normalized: bool = True
    kernel: str = "RBF"
    sigma: float = 1.0
    estimator: str = "biased"

    mmd: float | None = None
    das: float | None = None
    delta_mmd_vs_parent: float | None = None
    embedding_fail_rate: float = 0.0
    kernel_terms: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ReviewResult:
    """LLM quality dimensions plus deterministic full-dataset metrics."""

    schema_score: float
    relevance_score: float
    passed: bool
    issues: list[str] = field(default_factory=list)

    correctness_score: float = 0.0
    difficulty_score: float = 0.0
    diversity_score: float = 0.0

    dup_rate: float = 0.0
    null_rate: float = 0.0
    parse_valid_rate: float = 1.0
    contamination_rate: float = 0.0

    llm_composite: float = 0.0
    review_score: float = 0.0
    failure_signals: list[str] = field(default_factory=list)
    dataset_quality: DatasetQualityResult = field(default_factory=DatasetQualityResult)
