"""Lightweight raw-data pipeline evolution driven only by ReviewAgent feedback.

This entry point intentionally does not import MCTS, RecipeAgent, external-data MCP,
downstream training, Tier-2 selection, or benchmark evaluation.  One candidate is
produced per iteration and the previous candidate's structured ReviewResult is fed
directly into the next PipelineAgent prompt.  Training/evaluation remains an explicit
manual step after the loop has finished.
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
from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult, TaskProfile
from dataflow_evolver.engine.execution_wrapper import (
    ExecutionResult,
    ExecutionWrapperABC,
    SubprocessExecutionWrapper,
)
from dataflow_evolver.llm.serving import build_serving
from dataflow_evolver.utils.config import Config, load_config
from dataflow_evolver.utils.logging import get_logger, setup_logging

logger = get_logger("lightweight")


@dataclass
class Candidate:
    """One pipeline/data/review observation in the linear feedback loop."""

    iteration: int
    node_dir_name: str
    config: PipelineConfig
    dataset_path: str | None = None
    review: ReviewResult | None = None
    execution_error: str | None = None

    @property
    def score(self) -> float:
        return self.review.tier1_score if self.review is not None else 0.0


class ReviewOnlyLoop:
    """A linear ``Pipeline -> Data -> Review -> next prompt`` loop."""

    def __init__(
        self,
        *,
        pipeline_agent: PipelineAgent,
        executor: ExecutionWrapperABC,
        reviewer: ReviewAgent,
        profile: TaskProfile,
        workspace: Path,
        agent_root: Path,
        run_name: str,
        max_iterations: int,
        execution_retries: int = 1,
        target_samples: int = 0,
        init_reference_code: str | None = None,
        init_instruction: str = "",
    ) -> None:
        if max_iterations < 1:
            raise ValueError("loop.max_iterations 必须至少为 1（包含初始流水线）")
        self.pipeline_agent = pipeline_agent
        self.executor = executor
        self.reviewer = reviewer
        self.profile = profile
        self.workspace = workspace.resolve()
        self.agent_root = agent_root.resolve()
        self.run_name = _safe_run_name(run_name)
        self.max_iterations = max_iterations
        self.execution_retries = max(0, execution_retries)
        self.target_samples = max(0, target_samples)
        self.init_reference_code = init_reference_code
        self.init_instruction = init_instruction
        self.candidates: list[Candidate] = []

    @property
    def run_dir(self) -> Path:
        return self.agent_root / self.run_name

    def run(self) -> Candidate | None:
        """Run exactly ``max_iterations`` candidates unless generation/execution stops."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"运行目录已存在且非空：{self.run_dir}。请设置新的 project.run_name，"
                "避免旧节点产物污染本次评审。"
            )

        config = self.pipeline_agent.generate_initial(
            profile=self.profile,
            node_dir_name=self._node_name(0),
            reference_code=self.init_reference_code,
            instruction=self.init_instruction,
        )
        if config is None:
            logger.error("初始 PipelineAgent 未产出有效流水线。")
            return None

        for iteration in range(self.max_iterations):
            node_name = self._node_name(iteration)
            logger.info("==== Review-only 迭代 %d/%d ====", iteration + 1, self.max_iterations)
            candidate = self._execute_and_review(iteration, node_name, config)
            self.candidates.append(candidate)
            self._append_trace(candidate)

            if iteration + 1 >= self.max_iterations:
                break
            if candidate.review is None:
                logger.error("本轮没有可回传的 ReviewResult，终止线性反馈链。")
                break

            # The feedback loop is deliberately direct: no RecipeContext, global memory,
            # sibling summary, diagnostics, train result, or hidden reward shaping.
            config = self.pipeline_agent.generate(
                profile=self.profile,
                node_dir_name=self._node_name(iteration + 1),
                parent_code=candidate.config.code,
                parent_rationale=candidate.config.rationale,
                parent_review=candidate.review,
            )
            if config is None:
                logger.error("第 %d 轮 PipelineAgent 未产出有效流水线。", iteration + 2)
                break

        best = select_best(self.candidates)
        if best is not None:
            self._write_manifest(best)
            logger.info(
                "Review-only 最优候选：%s tier1=%.4f passed=%s",
                best.node_dir_name,
                best.score,
                best.review.passed if best.review else False,
            )
        return best

    def _node_name(self, iteration: int) -> str:
        return f"{self.run_name}/node_{iteration}"

    def _execute_and_review(
        self, iteration: int, node_name: str, config: PipelineConfig
    ) -> Candidate:
        env = (
            {"DF_TARGET_SAMPLES": str(self.target_samples)}
            if self.target_samples > 0
            else None
        )
        result: ExecutionResult = self.executor.run(config, node_name, env_overrides=env)
        retries = 0
        while not result.success and retries < self.execution_retries:
            retries += 1
            logger.info("流水线执行失败，进行第 %d/%d 次代码修复。", retries, self.execution_retries)
            fixed = self.pipeline_agent.self_correct(node_name, result.traceback or "")
            if fixed is None:
                break
            config = fixed
            result = self.executor.run(config, node_name, env_overrides=env)

        if not result.success or not result.dataset_path:
            logger.error("流水线执行最终失败：%s", result.traceback)
            return Candidate(
                iteration=iteration,
                node_dir_name=node_name,
                config=config,
                execution_error=result.traceback or "unknown execution failure",
            )

        evidence_path = self.agent_root / node_name / "review.jsonl"
        review = self.reviewer.review(
            result.dataset_path,
            self.profile,
            evidence_path=str(evidence_path),
            phase=f"iteration_{iteration}",
        )
        return Candidate(
            iteration=iteration,
            node_dir_name=node_name,
            config=config,
            dataset_path=result.dataset_path,
            review=review,
        )

    def _append_trace(self, candidate: Candidate) -> None:
        record = _candidate_record(candidate, self.agent_root)
        path = self.workspace / f"{self.run_name}_iterations.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _write_manifest(self, best: Candidate) -> None:
        manifest = {
            "mode": "review_only",
            "run_name": self.run_name,
            "max_iterations": self.max_iterations,
            "completed_iterations": len(self.candidates),
            "selection_rule": "passed first, then highest tier1_score",
            "best": _candidate_record(best, self.agent_root),
            "all_candidates": [_candidate_record(c, self.agent_root) for c in self.candidates],
            "manual_next_step": "Use best.dataset_path for downstream training/evaluation outside this loop.",
        }
        path = self.workspace / f"{self.run_name}_best.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )


def select_best(candidates: list[Candidate]) -> Candidate | None:
    """Prefer a passed review, then maximize its Tier-1 score."""
    reviewed = [candidate for candidate in candidates if candidate.review is not None]
    if not reviewed:
        return None
    return max(reviewed, key=lambda c: (bool(c.review and c.review.passed), c.score))


def build_loop(cfg: Config) -> ReviewOnlyLoop:
    """Build the lightweight loop from its intentionally small YAML contract."""
    project = cfg.project
    workspace = Path(project.workspace_dir)
    agent_root = Path(project.agent_root)
    agent_root.mkdir(parents=True, exist_ok=True)

    empty_entry = agent_root / "empty.jsonl"
    empty_entry.touch(exist_ok=True)
    corpus = input_corpus.discover(
        project.get("input_dir", "") or "",
        empty_entry,
        entry_name=str(project.get("input_entry", "") or ""),
    )
    if not corpus.grounded:
        raise ValueError(
            "lightweight review loop 要求用户提供固定 raw data：请设置 project.input_dir，"
            "且目录中至少包含一个 jsonl/json/csv/parquet 文件。"
        )

    task_cfg = cfg.task
    target_schema = _plain_dict(task_cfg.get("target_schema"))
    if not target_schema:
        raise ValueError("task.target_schema 不能为空；轻量入口不会额外调用 TaskProfiler 猜测字段")
    profile = TaskProfile(
        task_description="",  # CLI --task is injected in main() after construction.
        target_schema={str(k): str(v) for k, v in target_schema.items()},
        quality_criteria=list(task_cfg.get("quality_criteria", []) or []),
        downstream_task=str(task_cfg.get("downstream_task", "") or ""),
    )

    serving = build_serving(cfg.llm)
    review_cfg = cfg.review
    decon_cfg = review_cfg.get("decontamination") or Config({})
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
    )

    # Force external-data capabilities off even if a broader config is accidentally reused.
    agent_data = dict(cfg.agent_sdk.to_dict())
    agent_data["external_data"] = False
    agent_data["discovery"] = False
    agent_cfg = Config(agent_data)

    pipeline_llm = _resolve_pipeline_llm(cfg.llm)
    llm_block, extra_env = _pipeline_runtime_contract(pipeline_llm)
    generation_cfg = cfg.get("generation") or Config({})
    target_samples = int(generation_cfg.get("target_samples", 0) or 0)
    pipeline_agent = PipelineAgent(
        agent_cfg=agent_cfg,
        agent_root=str(agent_root),
        corpus=corpus,
        llm_block=llm_block,
        target_samples=target_samples,
        data_links_dir="",
    )

    loop_cfg = cfg.loop
    executor = SubprocessExecutionWrapper(
        agent_root=str(agent_root),
        pipeline_filename=str(agent_cfg.get("pipeline_filename", "pipeline.py")),
        timeout_sec=int(loop_cfg.get("execution_timeout_sec", 1200)),
        python_exe=str(loop_cfg.get("python_exe", "") or "") or None,
        extra_env=extra_env or None,
    )
    init_cfg = cfg.get("init") or Config({})
    reference_path = str(init_cfg.get("reference_pipeline", "") or "")
    reference_code = Path(reference_path).read_text(encoding="utf-8") if reference_path else None
    return ReviewOnlyLoop(
        pipeline_agent=pipeline_agent,
        executor=executor,
        reviewer=reviewer,
        profile=profile,
        workspace=workspace,
        agent_root=agent_root,
        run_name=str(project.get("run_name", "review_only") or "review_only"),
        max_iterations=int(loop_cfg.get("max_iterations", 4)),
        execution_retries=int(loop_cfg.get("execution_retries", 1)),
        target_samples=target_samples,
        init_reference_code=reference_code,
        init_instruction=str(init_cfg.get("instruction", "") or ""),
    )


def _candidate_record(candidate: Candidate, agent_root: Path | None = None) -> dict[str, Any]:
    pipeline_path = Path(candidate.node_dir_name) / "pipeline.py"
    if agent_root is not None:
        pipeline_path = agent_root / pipeline_path
    return {
        "iteration": candidate.iteration,
        "node_dir": candidate.node_dir_name,
        "pipeline_path": str(pipeline_path),
        "dataset_path": candidate.dataset_path,
        "execution_error": candidate.execution_error,
        "field_flow": candidate.config.field_flow,
        "operators": candidate.config.operators,
        "review": asdict(candidate.review) if candidate.review is not None else None,
    }


def _safe_run_name(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("project.run_name 必须是 agent_root 下的非空相对路径，且不能包含 '..'")
    return path.as_posix().strip("/")


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Config):
        return dict(value.to_dict())
    return dict(value or {}) if isinstance(value, dict) else {}


def _resolve_pipeline_llm(cfg_llm: Config) -> dict[str, dict[str, Any]]:
    api = cfg_llm.api
    sections = cfg_llm.get("pipeline_llm")

    def resolve(section: Config | None) -> dict[str, Any]:
        def pick(key: str, default: Any = None) -> Any:
            value = section.get(key, None) if section is not None else None
            return api.get(key, default) if value is None else value

        return {
            "api_url": pick("api_url"),
            "model_name": pick("model_name"),
            "api_key": pick("api_key", ""),
            "max_workers": pick("max_workers", 100),
        }

    if sections is None:
        return {"default": resolve(None)}
    raw = sections.to_dict() if isinstance(sections, Config) else sections
    return {
        str(name): resolve(Config(value) if isinstance(value, dict) else None)
        for name, value in raw.items()
    }


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
    parser.add_argument("--config", default="configs/lightweight_review.yaml")
    parser.add_argument("--task", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(args.log_level, Path(cfg.project.workspace_dir) / "run.log")
    loop = build_loop(cfg)
    loop.profile.task_description = args.task
    best = loop.run()
    if best is None:
        print("\n没有得到可评审的数据集。")
        return
    print(f"\n最优流水线：{loop.agent_root / best.node_dir_name / 'pipeline.py'}")
    print(f"最优数据：{best.dataset_path}")
    print(f"Review Tier-1：{best.score:.4f}（passed={best.review.passed}）")
    print(f"运行清单：{loop.workspace / f'{loop.run_name}_best.json'}")
    print("训练和下游评测未执行；请根据清单中的 best.dataset_path 人工启动。")


if __name__ == "__main__":
    main()
