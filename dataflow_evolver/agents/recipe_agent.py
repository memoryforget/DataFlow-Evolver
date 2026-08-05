"""RecipeAgent: evidence-guided data strategy planning, separate from code."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.recipe import load_and_validate_recipe, write_canonical_recipe
from dataflow_evolver.core.recipe_context import RecipeContext
from dataflow_evolver.core.schemas import TaskProfile
from dataflow_evolver.llm.agent_runtime import AgentRuntime, build_agent_runtime
from dataflow_evolver.llm.agent_sdk import RECIPE_SYSTEM_PROMPT_APPEND, AgentInfraError
from dataflow_evolver.llm.prompts import build_recipe_prompt, build_recipe_repair_prompt
from dataflow_evolver.utils.config import Config
from dataflow_evolver.utils.logging import get_logger


class RecipeAgent:
    """Use an agent SDK to propose recipes; use deterministic code to approve them."""

    name = "recipe"

    def __init__(
        self,
        agent_cfg: Config,
        agent_root: str,
        corpus: InputCorpus,
        target_samples: int,
        external_data: bool,
        data_links_dir: str = "",
        runtime: AgentRuntime | None = None,
    ) -> None:
        self.agent_cfg = agent_cfg
        self.agent_root = Path(agent_root).resolve()
        self.corpus = corpus
        self.target_samples = target_samples or 1
        self.external_data = external_data
        self.data_links_dir = data_links_dir
        self.max_revisions = int(agent_cfg.get("recipe_max_revisions", 2))
        self.runtime = runtime or build_agent_runtime(agent_cfg)
        self.logger = get_logger(f"agents.{self.name}")

    def plan(
        self,
        profile: TaskProfile,
        node_dir_name: str,
        context: RecipeContext,
    ) -> tuple[dict[str, Any], str] | None:
        """Produce, validate, repair, then freeze one recipe for PipelineAgent."""
        node_dir = self.agent_root / node_dir_name
        node_dir.mkdir(parents=True, exist_ok=True)
        recipe_path = node_dir / "recipe.json"
        prompt = build_recipe_prompt(
            profile,
            self.corpus,
            node_dir_name,
            self.target_samples,
            self.external_data,
            context=context,
        )
        for attempt in range(self.max_revisions + 1):
            result = self.runtime.run(
                prompt,
                cwd=str(self.agent_root),
                agent_cfg=self.agent_cfg,
                tool_log_dir=str(node_dir),
                phase="plan_recipe" if attempt == 0 else f"repair_recipe_{attempt}",
                system_prompt_append=RECIPE_SYSTEM_PROMPT_APPEND,
                tool_profile="recipe",
            )
            if not result.success:
                raise AgentInfraError(
                    result.error or "recipe planning SDK run failed",
                    retryable=result.retryable,
                )
            checked = load_and_validate_recipe(
                recipe_path,
                grounded=self.corpus.grounded,
                external_data=self.external_data,
                default_target_samples=self.target_samples,
                data_links_dir=self.data_links_dir,
                input_corpus_path=self.corpus.entry_path if self.corpus.grounded else "",
            )
            leaked = self._implementation_artifacts(node_dir)
            errors = list(checked.errors)
            if leaked:
                errors.append(
                    "Recipe 阶段越权创建了实现产物，请删除后重试：" + ", ".join(leaked)
                )
            if checked.valid and not leaked:
                write_canonical_recipe(recipe_path, checked)
                self.logger.info("recipe validated and frozen: %s", checked.recipe_hash)
                return checked.recipe, checked.recipe_hash
            self.logger.warning("recipe validation failed: %s", errors)
            prompt = build_recipe_repair_prompt(node_dir_name, errors)
        return None

    @staticmethod
    def _implementation_artifacts(node_dir: Path) -> list[str]:
        leaked: list[str] = []
        for name in ("pipeline.py", "decision.json"):
            if (node_dir / name).exists():
                leaked.append(name)
        operators = node_dir / "operators"
        if operators.exists():
            leaked.append("operators/")
        return leaked
