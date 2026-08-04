"""PipelineAgent：构建者/变异者，由 Claude Agent SDK 驱动。

它不直接产出文本，而是启动一次无头 Agent SDK 运行，让 SDK 内的 Claude 借助 MCP 算子发现、
源码阅读、dataflow-dev 技能把完整可运行的 DataFlow 流水线写到节点 workspace 并自测通过，
然后框架读取生成的 pipeline.py + decision.json 组装 PipelineConfig。

注意：PipelineAgent 不继承 AgentABC（那是非-SDK 辅助 Agent 的基类）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.recipe import load_and_validate_recipe, write_canonical_recipe
from dataflow_evolver.core.schemas import (
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TaskProfile,
    TrainResult,
)
from dataflow_evolver.llm.agent_sdk import AgentInfraError, run_pipeline_agent
from dataflow_evolver.llm.mcp_operators import _recommend_categories
from dataflow_evolver.llm.prompts import (
    build_pipeline_prompt,
    build_recipe_prompt,
    build_recipe_repair_prompt,
    build_review_selfcorrect_prompt,
    build_selfcorrect_prompt,
)
from dataflow_evolver.utils.config import Config
from dataflow_evolver.utils.logging import get_logger


class PipelineAgent:
    name = "pipeline"

    def __init__(
        self,
        agent_cfg: Config,
        agent_root: str,
        corpus: InputCorpus,
        llm_block: str = "",
        target_samples: int = 0,
        data_links_dir: str = "",
    ) -> None:
        self.agent_cfg = agent_cfg
        self.agent_root = Path(agent_root).resolve()
        self.corpus = corpus
        self.pipeline_filename = agent_cfg.get("pipeline_filename", "pipeline.py")
        self.llm_block = llm_block
        self.target_samples = target_samples
        self.recipe_max_revisions = int(agent_cfg.get("recipe_max_revisions", 2))
        # 非空 = 外部数据搜索工具已挂载，prompt 里要告诉 Agent 怎么用、落到哪。
        self.data_links_dir = data_links_dir
        self.logger = get_logger(f"agents.{self.name}")

    # ---- 生成（Init node：整棵树的起点，无父可继承）----
    def generate_initial(
        self,
        profile: TaskProfile,
        node_dir_name: str,
        reference_code: str | None = None,
        instruction: str | None = None,
    ) -> PipelineConfig | None:
        """构建 node_0 的初始流水线。

        与 `generate` 的差别只在 prompt 里多一段初始节点指引（参考实现改写 / 按指令构建 /
        自主设计三档），产物契约与组装逻辑完全一致——Init node 是一个普通实验节点。
        """
        node_dir_abs = self.agent_root / node_dir_name
        node_dir_abs.mkdir(parents=True, exist_ok=True)
        planned = self._plan_recipe(profile, node_dir_name)
        if planned is None:
            return None
        recipe, recipe_hash = planned
        prompt = build_pipeline_prompt(
            profile=profile,
            corpus=self.corpus,
            node_dir=node_dir_name,
            llm_block=self.llm_block,
            target_samples=self.target_samples,
            recommended_categories=_recommend_categories(profile.task_description),
            data_links_dir=self.data_links_dir,
            is_init=True,
            init_reference_code=reference_code,
            init_instruction=instruction,
            validated_recipe=recipe,
            recipe_hash=recipe_hash,
        )
        return self._run_and_assemble(
            prompt, node_dir_name, phase="generate_init", recipe=recipe, recipe_hash=recipe_hash
        )

    # ---- 生成（在父节点基础上的变异）----
    def generate(
        self,
        profile: TaskProfile,
        node_dir_name: str,
        diagnostic: DiagnosticReport | None = None,
        memory_summary: str | None = None,
        parent_code: str | None = None,
        parent_rationale: str | None = None,
        parent_review: ReviewResult | None = None,
        parent_train_result: TrainResult | None = None,
        siblings: list[str] | None = None,
        parent_recipe: dict[str, Any] | None = None,
    ) -> PipelineConfig | None:
        """运行一次 Agent SDK 生成流水线；成功则组装 PipelineConfig，否则 None。

        `parent_code`：父节点（首层为 Init node）的流水线代码，作为变异起点注入 prompt。
        `parent_rationale`：父节点 decision.json 的设计理由，避免子节点误改父节点刻意的取舍。
        `parent_review`：父节点 Tier-1 评审结果（**无论是否通过**都注入），提供五维/硬指标/
            失败信号等可行动的变异靶子——搜索期这是唯一存在的质量信号。
        `siblings`：同父兄弟已试过的方向摘要，促使多个子节点分散探索而非重复同一变异。
        """
        node_dir_abs = self.agent_root / node_dir_name
        node_dir_abs.mkdir(parents=True, exist_ok=True)
        planned = self._plan_recipe(profile, node_dir_name, parent_recipe=parent_recipe)
        if planned is None:
            return None
        recipe, recipe_hash = planned
        # 预调用推荐引擎，把结果直接注入 prompt 强制引导算子选择
        recommended = _recommend_categories(profile.task_description)
        prompt = build_pipeline_prompt(
            profile=profile,
            corpus=self.corpus,
            node_dir=node_dir_name,
            diagnostic=diagnostic,
            memory_summary=memory_summary,
            llm_block=self.llm_block,
            parent_code=parent_code,
            target_samples=self.target_samples,
            recommended_categories=recommended,
            data_links_dir=self.data_links_dir,
            parent_rationale=parent_rationale,
            parent_review=parent_review,
            parent_train_result=parent_train_result,
            siblings=siblings,
            validated_recipe=recipe,
            recipe_hash=recipe_hash,
        )
        return self._run_and_assemble(
            prompt, node_dir_name, recipe=recipe, recipe_hash=recipe_hash
        )

    # ---- 自修正（执行崩溃后重试）----
    def self_correct(self, node_dir_name: str, traceback: str) -> PipelineConfig | None:
        prompt = build_selfcorrect_prompt(node_dir_name, traceback)
        return self._run_and_assemble(
            prompt, node_dir_name, warn_stale_decision=True, phase="self_correct_exec"
        )

    # ---- 自修正（Phase-1 评审打回后重试；prompt 提示可查看 cache 下各算子中间产物定位）----
    def self_correct_review(
        self,
        node_dir_name: str,
        issues: list[str],
        history: list[str] | None = None,
        failure_signals: list[str] | None = None,
    ) -> PipelineConfig | None:
        """`history`：本节点此前各轮修复的目标与成效，避免推翻上一轮已生效的改动。

        `failure_signals`：命中的 F 编号，prompt 渲染时查表拼成标准化标签段落。
        """
        prompt = build_review_selfcorrect_prompt(
            node_dir_name, issues, history=history, failure_signals=failure_signals
        )
        return self._run_and_assemble(
            prompt,
            node_dir_name,
            warn_stale_decision=True,
            phase=f"self_correct_review_{len(history or [])+ 1}",
        )

    # ---- 内部：跑 SDK + 读产物 ----
    def _run_and_assemble(
        self,
        prompt: str,
        node_dir_name: str,
        warn_stale_decision: bool = False,
        phase: str = "generate",
        recipe: dict[str, Any] | None = None,
        recipe_hash: str = "",
    ) -> PipelineConfig | None:
        decision_path = self.agent_root / node_dir_name / "decision.json"
        fp_before = _fingerprint(decision_path) if warn_stale_decision else None
        result = run_pipeline_agent(
            prompt,
            cwd=str(self.agent_root),
            agent_cfg=self.agent_cfg,
            tool_log_dir=str(self.agent_root / node_dir_name),
            phase=phase,
        )
        if not result.success:
            # SDK/基础设施级失败（429/5xx/连接/超时/进程崩溃）——非流水线质量问题。
            # 抛给主循环区分处理：不惩罚节点、不消耗迭代预算（详见 main._step）。
            self.logger.error(
                "PipelineAgent SDK 运行失败（基础设施级，retryable=%s）：%s",
                result.retryable, result.error,
            )
            raise AgentInfraError(result.error or "Agent SDK 运行失败", retryable=result.retryable)

        node_dir_abs = self.agent_root / node_dir_name
        pipeline_path = node_dir_abs / self.pipeline_filename
        if not pipeline_path.exists():
            self.logger.error("未在 %s 找到生成的流水线文件，视为扩展失败。", pipeline_path)
            return None

        code = pipeline_path.read_text(encoding="utf-8")
        # 自修正后 decision.json 未被改动 = 算子构成/字段流转/设计意图仍是修复前那一版，
        # 而 code 已经变了。只告警不阻断（旧版本可能确实没有结构变化），但要能在日志里查到。
        if warn_stale_decision and _fingerprint(decision_path) == fp_before:
            self.logger.warning(
                "自修正后 %s 未更新：本节点记录的算子/字段流转/设计意图可能与新代码不一致"
                "（会传染到历史记忆与子节点继承）。", decision_path,
            )
        decision = _read_decision(decision_path)
        operators, field_flow, rationale = _parse_decision(decision)
        self.logger.info(
            "PipelineAgent 产出流水线：%s（%d 个算子，flow=%s）",
            pipeline_path, len(operators), field_flow,
        )
        if recipe is None:
            recipe, existing_hash = self._load_existing_recipe(node_dir_name)
            recipe_hash = recipe_hash or existing_hash
        return PipelineConfig(
            operators=operators,
            field_flow=field_flow,
            code=code,
            rationale=rationale,
            recipe=recipe or {},
            recipe_hash=recipe_hash,
        )

    def _plan_recipe(
        self,
        profile: TaskProfile,
        node_dir_name: str,
        parent_recipe: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str] | None:
        """Ask the PipelineAgent to propose; let deterministic framework code approve."""
        recipe_path = self.agent_root / node_dir_name / "recipe.json"
        prompt = build_recipe_prompt(
            profile,
            self.corpus,
            node_dir_name,
            self.target_samples or 1,
            bool(self.data_links_dir),
            parent_recipe=parent_recipe,
        )
        for attempt in range(self.recipe_max_revisions + 1):
            result = run_pipeline_agent(
                prompt,
                cwd=str(self.agent_root),
                agent_cfg=self.agent_cfg,
                tool_log_dir=str(self.agent_root / node_dir_name),
                phase="plan_recipe" if attempt == 0 else f"repair_recipe_{attempt}",
            )
            if not result.success:
                raise AgentInfraError(
                    result.error or "recipe planning SDK run failed", retryable=result.retryable
                )
            checked = load_and_validate_recipe(
                recipe_path,
                grounded=self.corpus.grounded,
                external_data=bool(self.data_links_dir),
                default_target_samples=self.target_samples or 1,
            )
            if checked.valid:
                write_canonical_recipe(recipe_path, checked)
                self.logger.info("recipe validated and frozen: %s", checked.recipe_hash)
                return checked.recipe, checked.recipe_hash
            self.logger.warning("recipe validation failed: %s", checked.errors)
            prompt = build_recipe_repair_prompt(node_dir_name, checked.errors)
        return None

    def _load_existing_recipe(self, node_dir_name: str) -> tuple[dict[str, Any], str]:
        corpus = getattr(self, "corpus", None)
        if corpus is None:
            return {}, ""
        checked = load_and_validate_recipe(
            self.agent_root / node_dir_name / "recipe.json",
            grounded=corpus.grounded,
            external_data=bool(getattr(self, "data_links_dir", "")),
            default_target_samples=getattr(self, "target_samples", 0) or 1,
        )
        return (checked.recipe, checked.recipe_hash) if checked.valid else ({}, "")


def _read_decision(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _fingerprint(path: Path) -> str | None:
    """文件内容指纹（不存在返回 None）。

    用内容而非 mtime 判断"有没有被改"：mtime 有文件系统粒度问题（同一 tick 内的改写测不出），
    而且内容原样重写在语义上本就等于"描述没更新"。
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _first_present(d: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    """返回第一个存在且非空的键值（decision.json 由 LLM 自由生成，键名会漂移）。"""
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return default


# decision.json 的 schema 虽已写进 prompt，但它由 LLM 生成，键名仍会漂移（实测同一次运行里
# 有节点写 "operators"/"description"/"design_rationale"，有节点写 "ops" 字符串数组）。
# 严格按单一键名读取会静默丢掉整份算子列表与 rationale，进而让 global memory 失去去重信息，
# 故这里对同义键做宽容匹配。
_OPS_KEYS = ("ops", "operators", "operator_list", "pipeline_ops")
_PURPOSE_KEYS = ("purpose", "description", "desc", "role")
_RATIONALE_KEYS = ("reason", "rationale", "design_rationale", "reasoning")
_FLOW_KEYS = ("field_flow", "fields_flow", "flow")


def _parse_decision(decision: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    """把 Stage-1 决策 JSON 归一化为 (operators, field_flow, rationale)。

    ops 可能是字符串列表或对象列表，统一成 [{"name", "custom", "purpose", "params"}]。
    """
    raw_ops = _first_present(decision, _OPS_KEYS, default=[])
    operators: list[dict[str, Any]] = []
    if isinstance(raw_ops, dict):        # 少数情况写成 {算子名: {...}} 映射
        raw_ops = [{"name": k, **(v if isinstance(v, dict) else {})} for k, v in raw_ops.items()]
    if isinstance(raw_ops, (list, tuple)):
        for op in raw_ops:
            if isinstance(op, str):
                operators.append({"name": op, "custom": True, "purpose": "", "params": {}})
            elif isinstance(op, dict):
                operators.append({
                    "name": _first_present(op, ("name", "operator", "class"), "?"),
                    "custom": op.get("custom", True),
                    "purpose": _first_present(op, _PURPOSE_KEYS),
                    "params": op.get("params", {}),
                })
    field_flow = _first_present(decision, _FLOW_KEYS)
    rationale = _first_present(decision, _RATIONALE_KEYS)
    return operators, field_flow, rationale
