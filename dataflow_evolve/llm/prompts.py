"""各 Agent 的 prompt 模板集中管理。

- PipelineAgent 的模板面向 Claude Agent SDK（指示其使用本地技能写流水线）。
- Profiler / Review / Diagnostic 的模板面向 OpenAI-compatible API，强制 JSON 输出。
"""
from __future__ import annotations

from dataflow_evolve.core.schemas import DiagnosticReport, TaskProfile

# ---------------------------------------------------------------------------
# TaskProfilerAgent
# ---------------------------------------------------------------------------
PROFILER_PROMPT = """你是一名数据工程任务分析师。根据下面的高层任务描述（及可选样例），\
把它形式化为结构化的数据需求。

任务描述：
{task_description}

{sample_block}

请只输出一个 JSON 对象，字段如下：
{{
  "task_description": "对任务的精炼复述",
  "target_schema": {{"字段名": "类型(str/int/float/list/...)", ...}},
  "domain": "领域，如 math / code / general-QA",
  "quality_criteria": ["高质量数据应满足的要点", ...],
  "downstream_task": "下游训练任务标识，如 math-QA-SFT"
}}
不要输出 JSON 以外的任何内容。"""


def build_profiler_prompt(task_description: str, sample_text: str | None = None) -> str:
    sample_block = ""
    if sample_text:
        sample_block = f"样例数据（节选）：\n{sample_text}\n"
    return PROFILER_PROMPT.format(
        task_description=task_description, sample_block=sample_block
    )


# ---------------------------------------------------------------------------
# PipelineAgent（面向 Agent SDK）
# ---------------------------------------------------------------------------
PIPELINE_AGENT_PROMPT = """使用 `generating-dataflow-pipeline` 技能，为下述目标编写一条\
完整、可直接运行的 DataFlow 流水线。

Target: {target}
Sample file: {seed_path}
Expected outputs: {expected_outputs}

要求：
1. 先按技能流程输出 Stage-1 算子决策摘要（JSON：{{"ops": [...], "field_flow": "...", "reason": "..."}}），\
再生成完整流水线代码。
2. 把最终流水线代码写入文件：`{node_dir}/pipeline.py`，并把 Stage-1 决策 JSON 写入 `{node_dir}/decision.json`。\
直接用 Write 工具把这两个文件写到 `{node_dir}/` 下（这是相对 cwd 的路径，务必创建该目录）。
3. 流水线的 `first_entry_file_name` 必须指向上面的 Sample file 路径。
4. 目标 schema（下游期望字段及类型）：{target_schema}
5. 流水线里的 LLM serving 必须用 `APILLMServing_request`，并严格使用下列参数（不要改成 openai.com/gpt-4o）：
{llm_block}
6. FileStorage 的 `cache_path` 设为 `{node_dir}/cache`，使产物落在本节点目录内。\
确保流水线最终会在 `{node_dir}/` 下产出一个 jsonl 结果文件。
{size_block}{parent_block}{diagnostic_block}{review_block}{memory_block}
约束：这是无头自动化运行，禁止触发任何交互式询问（AskUserQuestion）或 git/gh 仓库同步子流程。\
若需要超出 6 个核心算子的能力，可查阅 `dataflow-dev` 的 knowledge_base.md。"""

PIPELINE_SELFCORRECT_PROMPT = """上一次生成的流水线 `{node_dir}/pipeline.py` 执行时崩溃了。\
请用 `dataflow-dev` 的 `known_issues.md` 报错速查表匹配下面的错误并修复，然后\
覆盖写回 `{node_dir}/pipeline.py`（保持同一文件名）。

错误栈：
```
{traceback}
```

约束：无头运行，禁止交互式询问与 git/gh 子流程。只修复问题、保持流水线目标不变。"""


def build_pipeline_prompt(
    profile: TaskProfile,
    seed_path: str,
    node_dir: str,
    diagnostic: DiagnosticReport | None = None,
    memory_summary: str | None = None,
    llm_block: str = "",
    review_issues: list[str] | None = None,
    parent_code: str | None = None,
    target_samples: int = 0,
) -> str:
    # 目标产出规模：注入后让 Agent 据此设置算子参数（合成式调 num_samples / 转换式控采样过滤）。
    # 默认 0 = 不指定（避免 CondorGenerator 默认仅 3 条这种意外的小数据集，见 review）。
    size_block = ""
    if target_samples and target_samples > 0:
        size_block = (
            f"\n【目标产出规模】最终数据集约需 {target_samples} 条。请据此设置算子参数：\n"
            f"   - 若用从零合成类算子（如 CondorGenerator），把其 `num_samples` 设到约 {target_samples}；\n"
            f"   - 若是转换/过滤类流水线，确保读入足够源数据并避免过度过滤，使最终产出接近该规模。\n"
        )
    # 父节点（暖启动时首层父为基线）pipeline.py 代码作为变异起点：
    # 让 Agent「在父节点基础上做有的放矢的变异」，而非每次从零冷启动（见 docs/design-decisions.md §2.4）。
    parent_block = ""
    if parent_code:
        parent_block = (
            f"\n【变异起点】下面是父节点（或强基线）已验证可运行的流水线代码。请在它的基础上做"
            f"**针对性变异**（增删/替换算子、调参、改 prompt 等），目标是产出比它更优的数据；"
            f"不要原样照抄，也不要无依据地大改：\n```python\n{parent_code}\n```\n"
        )
    diagnostic_block = ""
    if diagnostic and diagnostic.actionable_suggestions:
        suggestions = "\n".join(f"  - {s}" for s in diagnostic.actionable_suggestions)
        diagnostic_block = (
            f"\n7. 父节点诊断反馈（请据此有的放矢地调整算子选择/参数）：\n"
            f"   失败概述：{diagnostic.summary}\n"
            f"   可执行建议：\n{suggestions}\n"
        )
    # Phase-1 评审打回的反馈：与诊断互斥（未训练的节点没有诊断），同样注入以指导针对性修正
    review_block = ""
    if review_issues:
        issues_text = "\n".join(f"  - {s}" for s in review_issues)
        review_block = (
            f"\n7. 上一轮数据被 Phase-1 评审打回（数据质量不达标，未进入训练）。"
            f"请针对下列问题修正流水线（如字段缺失/类型不符/与任务不相关/产出为空），再重新生成：\n"
            f"{issues_text}\n"
        )
    memory_block = ""
    if memory_summary:
        memory_block = f"\n参考历史记忆：\n{memory_summary}\n"
    return PIPELINE_AGENT_PROMPT.format(
        target=profile.task_description,
        seed_path=seed_path,
        expected_outputs=list(profile.target_schema.keys()),
        node_dir=node_dir,
        target_schema=profile.target_schema,
        llm_block=llm_block or "   （未提供具体 API 参数，沿用技能默认。）",
        size_block=size_block,
        parent_block=parent_block,
        diagnostic_block=diagnostic_block,
        review_block=review_block,
        memory_block=memory_block,
    )


def build_selfcorrect_prompt(node_dir: str, traceback: str) -> str:
    return PIPELINE_SELFCORRECT_PROMPT.format(node_dir=node_dir, traceback=traceback)


PIPELINE_REVIEW_SELFCORRECT_PROMPT = """上一次生成的流水线 `{node_dir}/pipeline.py` 能跑通，\
但产出的数据被 Phase-1 评审判定【不达标】（未进入训练）。请修复数据质量问题，然后覆盖写回 \
`{node_dir}/pipeline.py`（保持同一文件名与流水线目标不变）。

评审反馈的问题：
{issues}

【关键定位线索】DataFlow 的 FileStorage 会在 `{node_dir}/cache/` 下为**每个算子**按执行顺序落一个\
中间 jsonl（step1 = 第 1 个算子产出，step2 = 第 2 个算子产出，依此类推；可对照 `pipeline.py` 里 \
`forward()` 的算子调用顺序）。请用 Glob / Read 工具**逐个查看这些中间文件里的实际数据**，\
逐 step 追踪数据质量是从哪一步（哪个算子）开始变坏的——既包括行数骤减/变空，也包括行数没变但\
**内容劣化**（某字段从有值变 null/空、schema 不符、跑题等）——据此判断是哪个算子的配置或用法有问题并修复它。

约束：无头运行，禁止交互式询问（AskUserQuestion）与 git/gh 子流程。只修复数据质量、保持目标不变。"""


def build_review_selfcorrect_prompt(node_dir: str, issues: list[str]) -> str:
    issues_text = "\n".join(f"  - {s}" for s in issues) if issues else "  - （评审未给出具体条目，请据中间产物自行判断）"
    return PIPELINE_REVIEW_SELFCORRECT_PROMPT.format(node_dir=node_dir, issues=issues_text)


# ---------------------------------------------------------------------------
# ReviewAgent（Phase-1 快反馈）
# ---------------------------------------------------------------------------
REVIEW_PROMPT = """你是数据质量评审员。下面是某条流水线生成数据集的抽样，以及目标任务画像。\
请评估数据是否满足要求。

目标任务：{task_description}
期望 schema（字段→类型）：{target_schema}
质量要点：{quality_criteria}

数据抽样（每行一条 JSON）：
{samples}

请只输出一个 JSON 对象：
{{
  "schema_score": 0.0-1.0,        // 字段齐全度/类型符合度
  "relevance_score": 0.0-1.0,     // 与目标任务的相关性
  "issues": ["发现的问题", ...]
}}
不要输出 JSON 以外的内容。"""


def build_review_prompt(profile: TaskProfile, samples_text: str) -> str:
    return REVIEW_PROMPT.format(
        task_description=profile.task_description,
        target_schema=profile.target_schema,
        quality_criteria=profile.quality_criteria,
        samples=samples_text,
    )


# ---------------------------------------------------------------------------
# DiagnosticAgent（Phase-2 慢反馈）
# ---------------------------------------------------------------------------
DIAGNOSTIC_PROMPT = """你是模型失败归因专家。下面是某数据集训练出的模型在下游评估中的失败样例。\
请聚类失败模式，并给出"如何改进数据生成流水线"的可执行建议。

下游任务：{downstream_task}
失败样例（input / prediction / ground_truth / error_type）：
{bad_cases}

请只输出一个 JSON 对象：
{{
  "summary": "一句话概括主要失败模式",
  "failure_patterns": ["失败模式1", ...],
  "actionable_suggestions": ["对数据流水线的具体改进建议（指向算子/参数/数据来源）", ...]
}}
不要输出 JSON 以外的内容。"""


def build_diagnostic_prompt(downstream_task: str, bad_cases_text: str) -> str:
    return DIAGNOSTIC_PROMPT.format(
        downstream_task=downstream_task, bad_cases=bad_cases_text
    )
