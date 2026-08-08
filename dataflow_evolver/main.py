"""Incumbent-challenger raw-data pipeline improvement driven by review feedback.

One challenger is produced per iteration. Only a better candidate replaces the
best-so-far incumbent; compact attempt history discourages repeated failed strategies.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dataflow_evolver.agents.pipeline_agent import PipelineAgent
from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.core import decontamination, input_corpus
from dataflow_evolver.core.dataset_quality import (
    DatasetQualityEvaluator,
    OpenAIChatEmbeddingBackend,
    ProxyDatasetConfig,
)
from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult, TaskSpec
from dataflow_evolver.engine.execution_wrapper import (
    ExecutionResult,
    ExecutionWrapperABC,
    SubprocessExecutionWrapper,
)
from dataflow_evolver.llm.serving import build_serving
from dataflow_evolver.utils.config import Config, load_config
from dataflow_evolver.utils.logging import get_logger, setup_logging

logger = get_logger("main")


@dataclass
class Candidate:
    """One pipeline/data/review observation in the incumbent feedback loop."""

    iteration: int
    iteration_dir: Path
    config: PipelineConfig
    dataset_path: str | None = None
    review: ReviewResult | None = None
    execution_error: str | None = None
    failure_stage: str | None = None
    parent_iteration: int | None = None
    accepted_as_incumbent: bool = False

    @property
    def score(self) -> float:
        return self.review.review_score if self.review is not None else 0.0


class ReviewOnlyLoop:
    """A best-so-far incumbent challenged by one generated pipeline per iteration."""

    def __init__(
        self,
        *,
        pipeline_agent: PipelineAgent,
        executor: ExecutionWrapperABC,
        reviewer: ReviewAgent,
        task: TaskSpec,
        workspace: Path,
        run_name: str,
        max_iterations: int,
        execution_repairs: int = 1,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("loop.max_iterations 必须至少为 1（包含初始流水线）")
        self.pipeline_agent = pipeline_agent
        self.executor = executor
        self.reviewer = reviewer
        self.task = task
        self.workspace = workspace.resolve()
        self.run_name = _safe_run_name(run_name)
        self.max_iterations = max_iterations
        self.execution_repairs = max(0, execution_repairs)
        self.candidates: list[Candidate] = []

    @property
    def run_dir(self) -> Path:
        return self.workspace / "runs" / self.run_name

    def run(self) -> Candidate | None:
        """Run up to ``max_iterations`` candidates while retaining the best incumbent."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"运行目录已存在且非空：{self.run_dir}。请设置新的 project.run_name，"
                "避免旧 iteration 产物污染本次评审。"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        config = self.pipeline_agent.generate_initial(
            task=self.task,
            iteration_dir=self._iteration_dir(0),
        )
        if config is None:
            logger.error("初始 PipelineAgent 未产出有效流水线。")
            return None

        incumbent: Candidate | None = None
        for iteration in range(self.max_iterations):
            iteration_dir = self._iteration_dir(iteration)
            logger.info("==== Review-only 迭代 %d/%d ====", iteration + 1, self.max_iterations)
            parent_incumbent = incumbent
            candidate = self._execute_and_review(
                iteration,
                iteration_dir,
                config,
                parent_review=(
                    parent_incumbent.review if parent_incumbent is not None else None
                ),
            )
            candidate.parent_iteration = (
                parent_incumbent.iteration if parent_incumbent is not None else None
            )
            if _is_better(candidate, incumbent):
                candidate.accepted_as_incumbent = True
                incumbent = candidate
                logger.info(
                    "接受 challenger 为 incumbent：%s score=%.4f",
                    candidate.iteration_dir.name,
                    candidate.score,
                )
            else:
                logger.info(
                    "拒绝 challenger，保留 incumbent=%s",
                    incumbent.iteration_dir.name if incumbent is not None else "none",
                )
            self.candidates.append(candidate)
            self._append_trace(candidate)

            if iteration + 1 >= self.max_iterations:
                break
            if incumbent is None or incumbent.review is None:
                logger.error("尚无可用 incumbent，无法继续生成 challenger。")
                break

            # Always branch from best-so-far. Rejected attempts enter only a compact
            # decision history, never as inherited pipeline code.
            config = self.pipeline_agent.generate(
                task=self.task,
                iteration_dir=self._iteration_dir(iteration + 1),
                parent_code=incumbent.config.code,
                parent_rationale=incumbent.config.rationale,
                parent_review=incumbent.review,
                evolution_history=self._evolution_history(),
            )
            if config is None:
                logger.error("第 %d 轮 PipelineAgent 未产出有效流水线。", iteration + 2)
                break

        best = incumbent or select_best(self.candidates)
        if best is not None:
            self._write_manifest(best)
            logger.info(
                "Review-only 最优候选：%s review_score=%.4f passed=%s",
                best.iteration_dir.name,
                best.score,
                best.review.passed if best.review else False,
            )
        return best

    def _iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / f"iteration_{iteration + 1:03d}"

    def _execute_and_review(
        self,
        iteration: int,
        iteration_dir: Path,
        config: PipelineConfig,
        parent_review: ReviewResult | None = None,
    ) -> Candidate:
        result: ExecutionResult = self.executor.run(config, iteration_dir)
        retries = 0
        while not result.success and retries < self.execution_repairs:
            retries += 1
            logger.info("流水线执行失败，进行第 %d/%d 次代码修复。", retries, self.execution_repairs)
            fixed = self.pipeline_agent.self_correct(
                iteration_dir,
                result.traceback or "",
                result.diagnostics,
                failure_stage=result.failure_stage or "execution",
            )
            if fixed is None:
                break
            config = fixed
            result = self.executor.run(config, iteration_dir)

        if not result.success or not result.dataset_path:
            logger.error("流水线执行最终失败：%s", result.traceback)
            return Candidate(
                iteration=iteration,
                iteration_dir=iteration_dir,
                config=config,
                execution_error=result.traceback or "unknown execution failure",
                failure_stage=result.failure_stage,
            )

        evidence_path = iteration_dir / "review.jsonl"
        review = self.reviewer.review(
            result.dataset_path,
            self.task,
            evidence_path=str(evidence_path),
            phase=f"iteration_{iteration}",
            parent_review=parent_review,
            usage_dir=str(iteration_dir),
        )
        return Candidate(
            iteration=iteration,
            iteration_dir=iteration_dir,
            config=config,
            dataset_path=result.dataset_path,
            review=review,
        )

    def _append_trace(self, candidate: Candidate) -> None:
        record = _candidate_record(candidate)
        path = self.run_dir / "iterations.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _evolution_history(self) -> list[dict[str, Any]]:
        """Return compact decisions, not prior pipeline code or full reviews."""
        history: list[dict[str, Any]] = []
        for candidate in self.candidates:
            history.append(
                {
                    "iteration": candidate.iteration + 1,
                    "outcome": (
                        "accepted" if candidate.accepted_as_incumbent else "rejected"
                    ),
                    "review_score": candidate.score if candidate.review is not None else None,
                    "passed": candidate.review.passed if candidate.review is not None else False,
                    "operators": [
                        str(operator.get("name", "?"))
                        for operator in candidate.config.operators[:8]
                    ],
                    "decision": _compact_text(candidate.config.rationale, 240),
                    "issues": (
                        [_compact_text(issue, 160) for issue in candidate.review.issues[:2]]
                        if candidate.review is not None
                        else [_compact_text(candidate.execution_error or "execution failed", 160)]
                    ),
                }
            )
        return history

    def _write_manifest(self, best: Candidate) -> None:
        manifest = {
            "mode": "pipeline_review",
            "run_name": self.run_name,
            "max_iterations": self.max_iterations,
            "completed_iterations": len(self.candidates),
            "selection_rule": "best-so-far incumbent: passed first, then highest review_score",
            "best": _candidate_record(best),
            "evolution_history": self._evolution_history(),
            "all_candidates": [_candidate_record(c) for c in self.candidates],
            "manual_next_step": "Use best.dataset_path for downstream training/evaluation outside this loop.",
        }
        path = self.run_dir / "manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )


def select_best(candidates: list[Candidate]) -> Candidate | None:
    """Prefer a passed review, then maximize its Review score."""
    reviewed = [candidate for candidate in candidates if candidate.review is not None]
    if not reviewed:
        return None
    return max(reviewed, key=lambda c: (bool(c.review and c.review.passed), c.score))


def _is_better(challenger: Candidate, incumbent: Candidate | None) -> bool:
    if challenger.review is None:
        return False
    if incumbent is None or incumbent.review is None:
        return True
    challenger_rank = (challenger.review.passed, challenger.score)
    incumbent_rank = (incumbent.review.passed, incumbent.score)
    return challenger_rank > incumbent_rank


def build_loop(cfg: Config) -> ReviewOnlyLoop:
    """Build the pipeline-review loop from the default YAML contract."""
    project = cfg.project
    workspace = Path(project.workspace_dir).resolve()
    if not (workspace / "dataflow").is_dir():
        raise FileNotFoundError(f"project.workspace_dir 缺少 vendored dataflow：{workspace}")
    if not (workspace / ".claude/skills/dataflow-dev/SKILL.md").is_file():
        raise FileNotFoundError(f"project.workspace_dir 缺少 dataflow-dev skill：{workspace}")
    backend = str(cfg.agent.get("backend", "claude") or "claude").strip().lower()
    if backend == "codex" and not (
        workspace / ".agents/skills/dataflow-dev/SKILL.md"
    ).is_file():
        raise FileNotFoundError(
            f"Codex backend requires .agents/skills/dataflow-dev/SKILL.md: {workspace}"
        )

    corpus = input_corpus.discover(
        project.get("input_dir", "") or "",
        entry_name=str(project.get("input_entry", "") or ""),
    )

    task_cfg = cfg.task
    target_schema = _plain_dict(task_cfg.get("target_schema"))
    if not target_schema:
        raise ValueError("task.target_schema 不能为空；入口不会自动猜测输出字段")
    task = TaskSpec(
        task_description="",  # CLI --task is injected in main() after construction.
        target_schema={str(k): str(v) for k, v in target_schema.items()},
        quality_criteria=list(task_cfg.get("quality_criteria", []) or []),
    )

    serving = build_serving(cfg.llm)
    review_cfg = cfg.review
    decon_cfg = review_cfg.get("decontamination") or Config({})
    dataset_quality_cfg = review_cfg.get("dataset_quality") or Config({})
    dataset_quality_evaluator, dataset_quality_fields = _build_dataset_quality_evaluator(
        dataset_quality_cfg,
        workspace=workspace,
    )
    reviewer = ReviewAgent(
        serving,
        contamination=decontamination.build(
            reference_files=list(decon_cfg.get("reference_files", []) or []),
            n=int(decon_cfg.get("ngram", decontamination.DEFAULT_NGRAM)),
            max_rate=float(decon_cfg.get("max_rate", decontamination.DEFAULT_MAX_RATE)),
            fields=list(decon_cfg.get("fields", []) or []) or None,
        ),
        schema_threshold=float(review_cfg.get("schema_threshold", 0.6)),
        relevance_threshold=float(review_cfg.get("relevance_threshold", 0.6)),
        sample_size=int(review_cfg.get("sample_size", 80)),
        weights=_plain_dict(review_cfg.get("weights")) or None,
        hard_metrics=bool(review_cfg.get("hard_metrics", True)),
        seed=review_cfg.get("sample_seed", None),
        null_threshold=float(review_cfg.get("null_threshold", 0.5)),
        dataset_quality_evaluator=dataset_quality_evaluator,
        dataset_quality_fields=dataset_quality_fields,
        dataset_quality_weight=float(dataset_quality_cfg.get("score_weight", 0.25)),
        dataset_quality_score_weights=_plain_dict(
            dataset_quality_cfg.get("score_weights")
        ) or None,
    )

    agent_cfg = cfg.agent

    pipeline_llm = _resolve_pipeline_llm(cfg.llm)
    llm_block, extra_env = _pipeline_runtime_contract(pipeline_llm)
    pipeline_agent = PipelineAgent(
        agent_cfg=agent_cfg,
        workspace_dir=str(workspace),
        corpus=corpus,
        llm_block=llm_block,
    )

    loop_cfg = cfg.loop
    executor = SubprocessExecutionWrapper(
        workspace_dir=str(workspace),
        pipeline_filename=str(agent_cfg.get("pipeline_filename", "pipeline.py")),
        compile_timeout_sec=int(loop_cfg.get("compile_timeout_sec", 120)),
        timeout_sec=int(loop_cfg.get("execution_timeout_sec", 1200)),
        python_exe=str(loop_cfg.get("python_exe", "") or "") or None,
        extra_env=extra_env or None,
    )
    return ReviewOnlyLoop(
        pipeline_agent=pipeline_agent,
        executor=executor,
        reviewer=reviewer,
        task=task,
        workspace=workspace,
        run_name=str(project.get("run_name", "run") or "run"),
        max_iterations=int(loop_cfg.get("max_iterations", 4)),
        execution_repairs=int(loop_cfg.get("execution_repairs", 1)),
    )


def _candidate_record(candidate: Candidate) -> dict[str, Any]:
    pipeline_path = candidate.iteration_dir / "pipeline.py"
    return {
        "iteration": candidate.iteration,
        "iteration_dir": str(candidate.iteration_dir),
        "pipeline_path": str(pipeline_path),
        "dataset_path": candidate.dataset_path,
        "execution_error": candidate.execution_error,
        "failure_stage": candidate.failure_stage,
        "parent_iteration": candidate.parent_iteration,
        "accepted_as_incumbent": candidate.accepted_as_incumbent,
        "field_flow": candidate.config.field_flow,
        "operators": candidate.config.operators,
        "review": asdict(candidate.review) if candidate.review is not None else None,
    }


def _safe_run_name(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise ValueError("project.run_name 必须是不含路径分隔符的非空名称")
    return path.name


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Config):
        return dict(value.to_dict())
    return dict(value or {}) if isinstance(value, dict) else {}


def _build_dataset_quality_evaluator(
    value: Config,
    *,
    workspace: Path,
) -> tuple[DatasetQualityEvaluator | None, tuple[str, str]]:
    """Build the optional deterministic DAS evaluator from the review config."""
    candidate_cfg = value.get("candidate") or Config({})
    fields = (
        str(candidate_cfg.get("user_field", "instruction")),
        str(candidate_cfg.get("assistant_field", "output")),
    )
    if not _as_bool(value.get("enabled", False)):
        return None, fields

    proxy_cfg = value.get("proxy") or Config({})
    embedding_cfg = value.get("embedding") or Config({})
    proxy_path = str(proxy_cfg.get("path", "") or "").strip()
    base_url = str(embedding_cfg.get("base_url", "") or "").strip()
    model_name = str(embedding_cfg.get("model_name", "") or "").strip()
    if not proxy_path:
        raise ValueError("review.dataset_quality.proxy.path 不能为空")
    if not base_url:
        raise ValueError("review.dataset_quality.embedding.base_url 不能为空")
    if not model_name:
        raise ValueError("review.dataset_quality.embedding.model_name 不能为空")

    cache_value = str(value.get("cache_dir", "") or "").strip()
    cache_dir = Path(cache_value) if cache_value else workspace / "cache" / "das"
    proxy = ProxyDatasetConfig(
        source=str(proxy_cfg.get("source", "huggingface")),
        path=proxy_path,
        name=str(proxy_cfg.get("name", "") or ""),
        split=str(proxy_cfg.get("split", "train")),
        user_field=str(proxy_cfg.get("user_field", "question")),
        assistant_field=str(proxy_cfg.get("assistant_field", "response")),
    )
    embedder = OpenAIChatEmbeddingBackend(
        model_name=model_name,
        base_url=base_url,
        api_key=str(embedding_cfg.get("api_key", "EMPTY") or "EMPTY"),
        max_concurrent_requests=int(embedding_cfg.get("max_concurrent_requests", 128)),
        max_retries=int(embedding_cfg.get("max_retries", 3)),
        truncate_prompt_tokens=int(embedding_cfg.get("truncate_prompt_tokens", 40960)),
        truncation_side=str(embedding_cfg.get("truncation_side", "right")),
    )
    evaluator = DatasetQualityEvaluator(
        proxy=proxy,
        embedder=embedder,
        sample_size=int(value.get("sample_size", 5000)),
        seed=int(value.get("sample_seed", 42)),
        sigma=float(value.get("rbf_sigma", 1.0)),
        biased=_as_bool(value.get("biased", True)),
        normalize_embeddings=_as_bool(value.get("normalize_embeddings", True)),
        max_embedding_fail_rate=float(value.get("max_embedding_fail_rate", 0.02)),
        cache_dir=cache_dir,
        require_full_sample=_as_bool(value.get("require_full_sample", True)),
    )
    return evaluator, fields


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _resolve_pipeline_llm(cfg_llm: Config) -> dict[str, dict[str, Any]]:
    api = cfg_llm.api
    sections = cfg_llm.get("pipeline_llm")

    def resolve(section: Config | None) -> dict[str, Any]:
        def pick(key: str, default: Any = None) -> Any:
            value = section.get(key, None) if section is not None else None
            return api.get(key, default) if value is None else value

        return {
            "api_url": _normalize_pipeline_api_url(pick("api_url")),
            "model_name": pick("model_name"),
            "api_key": pick("api_key", ""),
            "max_workers": pick("max_workers", 10),
        }

    if sections is None:
        return {"default": resolve(None)}
    raw = sections.to_dict() if isinstance(sections, Config) else sections
    return {
        str(name): resolve(Config(value) if isinstance(value, dict) else None)
        for name, value in raw.items()
    }


def _normalize_pipeline_api_url(value: Any) -> Any:
    """Accept an OpenAI base URL while emitting the endpoint DataFlow posts to."""
    if not isinstance(value, str) or not value.strip():
        return value
    url = value.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


def _pipeline_runtime_contract(
    pipeline_llm: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    lines = ["   按用途区分的 Pipeline 算子 LLM Serving："]
    env: dict[str, str] = {}
    for purpose, item in pipeline_llm.items():
        env_name = f"DF_PIPELINE_API_KEY_{purpose.upper()}"
        lines.append(
            f"   [{purpose}] model={item['model_name']}, api_url={item['api_url']}, "
            f"key_name_of_api_key={env_name}, max_workers={item['max_workers']}"
        )
        if item.get("api_key"):
            env[env_name] = str(item["api_key"])
    lines.append("   key_name_of_api_key 是环境变量名；禁止把密钥写入 pipeline.py。")
    return "\n".join(lines), env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed raw data + PipelineAgent + ReviewAgent loop")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(args.log_level, Path(cfg.project.workspace_dir) / "run.log")
    loop = build_loop(cfg)
    loop.task.task_description = args.task
    best = loop.run()
    if best is None:
        print("\n没有得到可评审的数据集。")
        return
    print(f"\n最优流水线：{best.iteration_dir / 'pipeline.py'}")
    print(f"最优数据：{best.dataset_path}")
    print(f"Review score：{best.score:.4f}（passed={best.review.passed}）")
    print(f"运行清单：{loop.run_dir / 'manifest.json'}")
    print("训练和下游评测未执行；请根据清单中的 best.dataset_path 人工启动。")


if __name__ == "__main__":
    main()
