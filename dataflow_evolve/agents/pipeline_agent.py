"""PipelineAgent：构建者/变异者，由 Claude Agent SDK 驱动。

它不直接产出文本，而是启动一次无头 Agent SDK 运行，让 SDK 内的 Claude 借助 MCP 算子发现、
源码阅读、dataflow-dev 技能把完整可运行的 DataFlow 流水线写到节点 workspace 并自测通过，
然后框架读取生成的 pipeline.py + decision.json 组装 PipelineConfig。

注意：PipelineAgent 不继承 AgentABC（那是非-SDK 辅助 Agent 的基类）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataflow_evolve.core.schemas import DiagnosticReport, PipelineConfig, TaskProfile
from dataflow_evolve.llm.agent_sdk import run_pipeline_agent
from dataflow_evolve.llm.mcp_operators import _recommend_categories
from dataflow_evolve.llm.prompts import (
    build_pipeline_prompt,
    build_review_selfcorrect_prompt,
    build_selfcorrect_prompt,
)
from dataflow_evolve.utils.config import Config
from dataflow_evolve.utils.logging import get_logger


class PipelineAgent:
    name = "pipeline"

    def __init__(
        self,
        agent_cfg: Config,
        agent_root: str,
        entry_path: str,
        llm_block: str = "",
        target_samples: int = 0,
    ) -> None:
        self.agent_cfg = agent_cfg
        self.agent_root = Path(agent_root).resolve()
        self.entry_path = str(Path(entry_path).resolve())
        self.pipeline_filename = agent_cfg.get("pipeline_filename", "pipeline.py")
        self.llm_block = llm_block
        self.target_samples = target_samples
        self.logger = get_logger(f"agents.{self.name}")

    # ---- 生成（首次构建 / 有诊断的变异）----
    def generate(
        self,
        profile: TaskProfile,
        node_dir_name: str,
        diagnostic: DiagnosticReport | None = None,
        memory_summary: str | None = None,
        review_issues: list[str] | None = None,
        parent_code: str | None = None,
    ) -> PipelineConfig | None:
        """运行一次 Agent SDK 生成流水线；成功则组装 PipelineConfig，否则 None。

        `review_issues`：父节点上一轮被 Phase-1 评审打回的问题，注入 prompt 指导针对性修正。
        `parent_code`：父节点（暖启动时首层父为基线）的流水线代码，作为变异起点注入 prompt。
        """
        node_dir_abs = self.agent_root / node_dir_name
        node_dir_abs.mkdir(parents=True, exist_ok=True)
        # 预调用推荐引擎，把结果直接注入 prompt 强制引导算子选择
        recommended = _recommend_categories(profile.task_description)
        prompt = build_pipeline_prompt(
            profile=profile,
            entry_path=self.entry_path,
            node_dir=node_dir_name,
            diagnostic=diagnostic,
            memory_summary=memory_summary,
            llm_block=self.llm_block,
            review_issues=review_issues,
            parent_code=parent_code,
            target_samples=self.target_samples,
            recommended_categories=recommended,
        )
        return self._run_and_assemble(prompt, node_dir_name)

    # ---- 自修正（执行崩溃后重试）----
    def self_correct(self, node_dir_name: str, traceback: str) -> PipelineConfig | None:
        prompt = build_selfcorrect_prompt(node_dir_name, traceback)
        return self._run_and_assemble(prompt, node_dir_name)

    # ---- 自修正（Phase-1 评审打回后重试；prompt 提示可查看 cache 下各算子中间产物定位）----
    def self_correct_review(self, node_dir_name: str, issues: list[str]) -> PipelineConfig | None:
        prompt = build_review_selfcorrect_prompt(node_dir_name, issues)
        return self._run_and_assemble(prompt, node_dir_name)

    # ---- 内部：跑 SDK + 读产物 ----
    def _run_and_assemble(self, prompt: str, node_dir_name: str) -> PipelineConfig | None:
        result = run_pipeline_agent(prompt, cwd=str(self.agent_root), agent_cfg=self.agent_cfg)
        if not result.success:
            self.logger.error("PipelineAgent SDK 运行失败：%s", result.error)
            return None

        node_dir_abs = self.agent_root / node_dir_name
        pipeline_path = node_dir_abs / self.pipeline_filename
        if not pipeline_path.exists():
            self.logger.error("未在 %s 找到生成的流水线文件，视为扩展失败。", pipeline_path)
            return None

        code = pipeline_path.read_text(encoding="utf-8")
        decision = _read_decision(node_dir_abs / "decision.json")
        operators, field_flow, rationale = _parse_decision(decision)
        self.logger.info(
            "PipelineAgent 产出流水线：%s（%d 个算子，flow=%s）",
            pipeline_path, len(operators), field_flow,
        )
        return PipelineConfig(
            operators=operators,
            field_flow=field_flow,
            code=code,
            rationale=rationale,
        )


def _read_decision(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _parse_decision(decision: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    """把 Stage-1 决策 JSON 归一化为 (operators, field_flow, rationale)。

    ops 可能是字符串列表或对象列表，统一成 [{"name", "custom", "purpose", "params"}]。
    """
    raw_ops = decision.get("ops", [])
    operators: list[dict[str, Any]] = []
    for op in raw_ops:
        if isinstance(op, str):
            operators.append({"name": op, "custom": True, "purpose": "", "params": {}})
        elif isinstance(op, dict):
            operators.append({
                "name": op.get("name", op.get("operator", "?")),
                "custom": op.get("custom", True),
                "purpose": op.get("purpose", ""),
                "params": op.get("params", {}),
            })
    field_flow = decision.get("field_flow", "")
    rationale = decision.get("reason", decision.get("rationale", ""))
    return operators, field_flow, rationale
