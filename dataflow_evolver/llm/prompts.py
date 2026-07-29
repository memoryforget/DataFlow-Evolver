"""各 Agent 的 prompt 模板集中管理。

- PipelineAgent 的模板面向 Claude Agent SDK（指示其通过 MCP + 源码阅读 + compile 校验写流水线）。
- Profiler / Review / Diagnostic 的模板面向 OpenAI-compatible API，强制 JSON 输出。
"""
from __future__ import annotations

from dataflow_evolver.core.schemas import DiagnosticReport, ReviewResult, TaskProfile

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
PIPELINE_AGENT_PROMPT = """为下述目标编写一条完整、可直接运行的 DataFlow 流水线（含自定义算子）。

Target: {target}
Expected outputs: {expected_outputs}
Entry file（空输入）: {entry_path}

要求：
1. 先输出算子决策摘要，再生成完整流水线代码。摘要将原样写入 `{node_dir}/decision.json`，\
**键名必须严格按下面这份 schema**（框架按键名读取，改名/换同义词会导致信息丢失）：
{{"ops": [{{"name": "算子类名", "purpose": "这一步做什么（一句话）", "params": {{}}}}, ...],
 "field_flow": "Empty → 字段A → 字段A + 字段B → ...",
 "reason": "整体设计理由"}}
其中 `ops` 必须是**对象数组**且按流水线执行顺序列全每一个算子（含自定义算子），\
每个都要给 `purpose`；不要写成 `operators`、`description`、`design_rationale` 等其它键名。
2. 在 `{node_dir}/operators/` 目录下为每个处理步骤编写独立的自定义算子文件（每个算子一个 .py）。\
算子必须继承 OperatorABC（`from dataflow.core import OperatorABC`），实现 run() 方法。\
然后把流水线主文件写入 `{node_dir}/pipeline.py`，决策 JSON 写入 `{node_dir}/decision.json`。
3. 这是【从零合成（from-scratch synthesis）】任务：流水线不依赖任何输入语料。\
`first_entry_file_name` 必须指向上面的 Entry file（一个空 jsonl，仅为满足 DataFlow 的 first_entry 形参）；\
请用自定义的**生成类算子**凭模型知识合成数据。
4. 目标 schema（下游期望字段及类型）：{target_schema}
5. 流水线里的 LLM serving 必须用 `APILLMServing_request`，并严格使用下列参数：
{llm_block}
6. FileStorage 的 `cache_path` 设为 `{node_dir}/cache`，使产物落在本节点目录内。\
确保流水线最终会在 `{node_dir}/` 下产出一个 jsonl 结果文件。
7. 写完代码后，**必须执行 compile 校验**：在 `if __name__` 块中按此顺序编写：\
`pipeline.compile()` → `if os.environ.get('DF_COMPILE_ONLY'): sys.exit(0)` → `pipeline.forward()`。\
然后运行 `DF_COMPILE_ONLY=1 python {node_dir}/pipeline.py`。如果 compile() 报 KeyError/ImportError/TypeError，\
修复后重跑，直到 compile 通过。
{recommendation_block}{size_block}{parent_block}{diagnostic_block}{review_block}{sibling_block}{memory_block}
约束：这是无头自动化运行，禁止触发任何交互式询问（AskUserQuestion）或 git/gh 子流程。\
写自定义算子时，可用 MCP get_operator_detail 查看 reasoning 类别的已有算子源码作为编码风格参考。"""

PIPELINE_SELFCORRECT_PROMPT = """上一次生成的流水线 `{node_dir}/pipeline.py` 执行时崩溃了。\
请用 `dataflow-dev` 的 `known_issues.md` 报错速查表匹配下面的错误并修复。

**诊断步骤**：
1. 先读取 `.claude/skills/dataflow-dev/diagnostics/known_issues.md` 匹配错误关键词
2. 如果命中 Issue #010（LLM API）或 #011（FileStorage），直接应用 Issue 中的修复示例
3. 如果不确定 API 签名，查阅 DataFlow 源码确认：
   - FileStorage API: `dataflow/utils/storage.py`
   - LLM Serving API: `dataflow/serving/api_llm_serving_request.py`
   - OperatorABC: `dataflow/core/operator.py`
4. 用 Read 工具查看 `{node_dir}/operators/` 下的自定义算子源码确认问题位置

**常见错误快速修复**：
- `FileStorage.write() takes 2 positional arguments but 3 were given`
  → 修复：`storage.write(df)` (删除第二个参数)

- `'APILLMServing_request' object has no attribute 'request'`
  → 修复：`self.llm_serving.generate_from_input(user_inputs=..., system_prompt=...)`

修复后覆盖写回对应文件（pipeline.py 或 operators/ 下的算子文件），然后运行 \
`DF_COMPILE_ONLY=1 python {node_dir}/pipeline.py` 验证 compile 通过。

{decision_sync_note}

错误栈：
```
{traceback}
```

约束：无头运行，禁止交互式询问与 git/gh 子流程。只修复问题、保持流水线目标不变。"""


def build_pipeline_prompt(
    profile: TaskProfile,
    entry_path: str,
    node_dir: str,
    diagnostic: DiagnosticReport | None = None,
    memory_summary: str | None = None,
    llm_block: str = "",
    parent_code: str | None = None,
    target_samples: int = 0,
    recommended_categories: dict | None = None,
    parent_rationale: str | None = None,
    parent_review: ReviewResult | None = None,
    siblings: list[str] | None = None,
) -> str:
    # 目标产出规模：注入后让 Agent 据此设置算子参数（合成式调 num_samples / 转换式控采样过滤）。
    # 关键：产出条数必须【从环境变量 DF_TARGET_SAMPLES 运行时读取】，不要写死——这样同一份
    # pipeline.py 可被复用两档规模：搜索期小规模探针（便宜、给 Tier-1 打分），终选期大规模重产
    # （用于真实 SFT+评测），无需重新调用 Agent SDK 生成代码（见 docs/posttrainbench_adaptation.md）。
    # 默认 0 = 不指定（避免 CondorGenerator 默认仅 3 条这种意外的小数据集，见 review）。
    size_block = ""
    if target_samples and target_samples > 0:
        size_block = (
            f"\n【目标产出规模 / 运行时可调，务必遵守】最终数据集条数**必须从环境变量 "
            f"DF_TARGET_SAMPLES 读取，不要写死具体数字**。在生成类算子里这样取值：\n"
            f"       import os\n"
            f"       num_samples = int(os.environ.get('DF_TARGET_SAMPLES', '{target_samples}'))\n"
            f"   然后用 num_samples 控制产出条数（或循环/批次次数），使最终产出接近该值。\n"
            f"   - 生成类算子：num_samples 直接作为生成条数；若算子内部会对 prompt 数做倍乘/除"
            f"（如 CondorGenerator 每 prompt 产多题），请换算使最终落盘条数≈num_samples；\n"
            f"   - 过滤/验证类算子：确保不会过度过滤导致最终产出远低于 num_samples；\n"
            f"   - 环境变量缺省时回退 {target_samples}，保证脱离编排器单独运行也能出数据。\n"
        )
    # 父节点（暖启动时首层父为基线）pipeline.py 代码作为变异起点：
    # 让 Agent「在父节点基础上做有的放矢的变异」，而非每次从零冷启动（见 docs/design-decisions.md §2.4）。
    parent_block = ""
    if parent_code:
        parent_block = (
            f"\n【变异起点】下面是父节点（或强基线）已验证可运行的流水线代码。请在它的基础上做"
            f"**针对性变异**（改进算子逻辑、调整 prompt、增删处理步骤、改参数等），目标是产出比它更优的数据；"
            f"不要原样照抄，也不要无依据地大改：\n```python\n{parent_code}\n```\n"
        )
    # 设计意图（decision.json 的 reason）独立于代码注入：代码只能表达"做了什么"，不能表达
    # "为什么这么做"。而 code 为空（续跑时 node 目录已被清理，_load_node_code 返回空串）恰恰
    # 是子节点信息最匮乏的时候——没有变异起点只能冷启动，此时设计意图是仅存的方向性信息，
    # 按最需要它的时候把它丢掉，取舍是反的。措辞随代码有无切换：有代码时"在此基础上"的"此"
    # 指那段代码，没代码时这句会悬空。
    if parent_rationale:
        if parent_code:
            parent_block += (
                f"\n【父节点设计意图】父节点这样设计的理由：{parent_rationale}\n"
                f"请把它当作已知约束：在此基础上改进，明确判断该取舍不成立再推翻它。\n"
            )
        else:
            parent_block += (
                f"\n【父节点设计意图】父节点这样设计的理由：{parent_rationale}\n"
                f"（父节点代码本次不可用，需从零编写流水线。）这是父节点的设计取舍，可作为方向参考："
                f"延续其中站得住的部分，或明确判断它不成立后另走一条路。\n"
            )
    diagnostic_block = ""
    if diagnostic and diagnostic.actionable_suggestions:
        suggestions = "\n".join(f"  - {s}" for s in diagnostic.actionable_suggestions)
        diagnostic_block = (
            f"\n【父节点诊断反馈】（请据此有的放矢地调整算子选择/参数）：\n"
            f"   失败概述：{diagnostic.summary}\n"
            f"   可执行建议：\n{suggestions}\n"
        )
    # parent_review 为 None 的情形是正常的：首层节点的父是基线/虚拟根，本就没被评审过。
    review_block = _parent_quality_block(parent_review) if parent_review is not None else ""
    # 兄弟节点已试方向：同一父节点下要长出多个子节点，若每次 prompt 完全相同，就只能指望
    # 采样随机性撞出不同方向，等于把多次昂贵扩展浪费成一次的信息量。
    sibling_block = ""
    if siblings:
        sibling_block = (
            "\n【兄弟节点已试过的方向（请勿重复，另选一个变异方向）】\n"
            + "\n".join(f"  - {s}" for s in siblings)
            + "\n"
        )
    memory_block = ""
    if memory_summary:
        memory_block = f"\n参考历史记忆：\n{memory_summary}\n"
    # 预注入 MCP 推荐结果：指引 Agent 可参考哪些类别的算子源码作为编码风格参考
    recommendation_block = ""
    if recommended_categories:
        rec_list = recommended_categories.get("recommended_categories", [])
        rec_text = "; ".join(
            f"{r['category']}(reason: {r.get('reason', '')})"
            for r in rec_list
        )
        recommendation_block = (
            f"\n【MCP 算子源码参考（强烈建议使用）】\n"
            f"   相关类别：{rec_text}\n"
            f"   **使用步骤**：\n"
            f"   1. 调用 MCP tool: `list_operators` 查看 {rec_list[0]['category']} 类别的算子列表\n"
            f"   2. 选择 1-2 个相关算子（如 ReasoningQuestionGenerator, ReasoningAnswerGenerator）\n"
            f"   3. 调用 MCP tool: `get_operator_detail` 获取算子的**完整源码**（在返回的 source_code 字段中）\n"
            f"   4. 学习源码中的：\n"
            f"      - Prompt 构造模式（如何拼接 system_prompt 和 user_input）\n"
            f"      - LLM 调用方式（generate_from_input 的正确用法）\n"
            f"      - 响应解析技巧（如何从 LLM 输出中提取结构化数据）\n"
            f"      - 错误处理模式（如何处理 None 返回值）\n"
            f"   5. 编写你自己的算子时，遵循相同的模式和 API 约定\n"
            f"   \n"
            f"   **注意**：MCP 提供的是**实现参考**，不要直接复用已有算子，而是学习其编码风格后写自定义算子。\n"
        )
    return PIPELINE_AGENT_PROMPT.format(
        target=profile.task_description,
        entry_path=entry_path,
        expected_outputs=list(profile.target_schema.keys()),
        node_dir=node_dir,
        target_schema=profile.target_schema,
        llm_block=llm_block or "   （未提供具体 API 参数，沿用技能默认。）",
        recommendation_block=recommendation_block,
        size_block=size_block,
        parent_block=parent_block,
        diagnostic_block=diagnostic_block,
        review_block=review_block,
        sibling_block=sibling_block,
        memory_block=memory_block,
    )


def build_selfcorrect_prompt(node_dir: str, traceback: str) -> str:
    return PIPELINE_SELFCORRECT_PROMPT.format(
        node_dir=node_dir,
        traceback=traceback,
        decision_sync_note=DECISION_SYNC_NOTE.format(node_dir=node_dir),
    )


PIPELINE_REVIEW_SELFCORRECT_PROMPT = """上一次生成的流水线 `{node_dir}/pipeline.py` 能跑通，\
但产出的数据被 Phase-1 评审判定【不达标】（未进入训练）。请修复数据质量问题，然后覆盖写回 \
`{node_dir}/pipeline.py` 或 `{node_dir}/operators/` 下的算子文件（保持流水线目标不变）。

评审反馈的问题：
{issues}
{signals_block}

【关键定位线索】DataFlow 的 FileStorage 会在 `{node_dir}/cache/` 下为**每个算子**按执行顺序落一个\
中间 jsonl（step1 = 第 1 个算子产出，step2 = 第 2 个算子产出，依此类推；可对照 `pipeline.py` 里 \
`forward()` 的算子调用顺序）。请用 Glob / Read 工具**逐个查看这些中间文件里的实际数据**，\
逐 step 追踪数据质量是从哪一步（哪个算子）开始变坏的——既包括行数骤减/变空，也包括行数没变但\
**内容劣化**（某字段从有值变 null/空、schema 不符、跑题等）——据此判断是哪个算子的逻辑或 prompt 有问题并修复它。

【DataFlow 源码参考】如果不确定 API 用法，可以查阅 DataFlow 源码确认：
- LLM Serving API 签名: `dataflow/serving/api_llm_serving_request.py`
- FileStorage API: `dataflow/utils/storage.py`
- Operator 基类: `dataflow/core/operator.py`
- 已有算子实现参考: `dataflow/operators/reasoning/` (数学推理类算子)

常见数据质量问题修复方向：
1. **字段为空/null**：检查对应算子的 LLM prompt 是否明确要求输出该字段
2. **LLM 返回 None**：检查是否正确处理 API 失败（`[r if r is not None else '' for r in responses]`）
3. **内容跑题**：检查 system_prompt 和 prompt 模板是否约束足够强
4. **格式不符**：检查响应解析逻辑（如 JSON 提取、正则匹配）

修复后运行 `DF_COMPILE_ONLY=1 python {node_dir}/pipeline.py` 验证 compile 通过。
{history_block}
{decision_sync_note}

约束：无头运行，禁止交互式询问（AskUserQuestion）与 git/gh 子流程。只修复数据质量、保持目标不变。"""

# 自修正只被要求覆盖写回 pipeline.py / operators/，于是 decision.json 会停留在初次生成那一版：
# 同一个 PipelineConfig 里 code 是新的、operators/field_flow/rationale 是旧的。这份错配会传染到
# global memory 的 ops/flow、子节点继承的 parent_rationale、以及 checkpoint。
DECISION_SYNC_NOTE = """【务必同步】若本次修复**增删了算子、改变了字段流转、或调整了设计取舍**，\
请同步覆盖写回 `{node_dir}/decision.json`（键名仍为 `ops` / `field_flow` / `reason`）。\
框架按这份文件记录本节点的算子构成与设计意图，并据此渲染历史记忆、传给子节点作为变异依据；\
不更新会让下游看到一份已不存在的设计。"""


def build_review_selfcorrect_prompt(
    node_dir: str,
    issues: list[str],
    history: list[str] | None = None,
    failure_signals: list[str] | None = None,
) -> str:
    issues_text = "\n".join(f"  - {s}" for s in issues) if issues else "  - （评审未给出具体条目，请据中间产物自行判断）"
    # 失败信号在渲染时按编号查表拼出。ReviewResult 里信号与 issues 是分开的两个字段，
    # 需要信号文本的消费点自己组装即可——FAILURE_SIGNALS 就在本模块，零成本。
    signals_block = ""
    if failure_signals:
        listed = "\n".join(
            f"  - {sid}：{FAILURE_SIGNALS[sid]}"
            for sid in failure_signals
            if sid in FAILURE_SIGNALS
        )
        if listed:
            signals_block = f"\n命中的失败信号（标准化标签，指示改进方向）：\n{listed}\n"
    # 历史修复轮次：只给最新一轮 issues 会让 LLM 不知道某段逻辑是上一轮为修 X 才加的，
    # 从而把它改掉、下一轮 X 复现，在节点内部空转掉有限的自修正次数。
    history_block = ""
    if history:
        history_block = (
            "\n【本节点已尝试过的修复（勿推翻其中已见效的部分，除非确有必要）】\n"
            + "\n".join(f"  - {h}" for h in history)
            + "\n"
        )
    return PIPELINE_REVIEW_SELFCORRECT_PROMPT.format(
        node_dir=node_dir,
        issues=issues_text,
        signals_block=signals_block,
        history_block=history_block,
        decision_sync_note=DECISION_SYNC_NOTE.format(node_dir=node_dir),
    )


# ---------------------------------------------------------------------------
# ReviewAgent（Phase-1 快反馈）
# ---------------------------------------------------------------------------
# ANDES F1-F7 失败信号分类。让 Tier-1 评审在打分之外标注命中的信号编号，得到标准化、可解释的
# 问题标签，存于 `ReviewResult.failure_signals`（与具体证据 `issues` 分开保存，各渲染点按编号
# 查本表拼文本）。仅作观测/反馈，不改变 tier1_score 的打分公式（见 docs/posttrainbench_adaptation.md）。
FAILURE_SIGNALS: dict[str, str] = {
    "F1": "描述过窄：生成围绕极少数结构展开，能力维度单一",
    "F2": "华而不实：表面花哨但逻辑塌陷，缺乏实质数据多样性",
    "F3": "模板锁定：大规模近重复 / 模板化雷同",
    "F4": "覆盖塌陷：未覆盖目标任务相关的关键能力维度",
    "F5": "约束漂移：偏离目标任务/领域的要求或答案格式",
    "F6": "有形无实：形式满足但训练价值低（过易/无信息量）",
    "F7": "描述空泛：生成结果语义松散、逐渐跑题",
}


# Tier-1 五维（与 ReviewAgent 的加权维度一致）：标量 tier1 对生成者几乎无指导价值，
# 拆成五维并点名最弱维，才能把"往哪改"讲清楚。
_DIM_LABELS: tuple[tuple[str, str], ...] = (
    ("correctness_score", "正确性"),
    ("relevance_score", "相关性"),
    ("difficulty_score", "难度"),
    ("diversity_score", "多样性"),
    ("schema_score", "schema 符合度"),
)


def _weakest_dims(review: ReviewResult, k: int = 2) -> list[tuple[str, float]]:
    dims = [(label, float(getattr(review, attr, 0.0) or 0.0)) for attr, label in _DIM_LABELS]
    return sorted(dims, key=lambda kv: kv[1])[:k]


def _parent_quality_block(review: ReviewResult) -> str:
    """把父节点的 Tier-1 归因渲染成可行动的变异指引。"""
    dims = " | ".join(
        f"{label} {float(getattr(review, attr, 0.0) or 0.0):.2f}" for attr, label in _DIM_LABELS
    )
    weakest = "、".join(f"{label}({score:.2f})" for label, score in _weakest_dims(review))
    lines = [
        "\n【父节点数据质量归因】",
        f"  Tier-1 复合分：{review.tier1_score:.3f}（评审{'通过' if review.passed else '未通过'}）",
        f"  五维得分：{dims}",
        f"  **最弱维度：{weakest} ← 本次变异应优先攻这些维度**",
        f"  硬指标：重复率 {review.dup_rate:.2f} | 空值率 {review.null_rate:.2f} | "
        f"可解析率 {review.parse_valid_rate:.2f}",
    ]
    if review.failure_signals:
        lines.append("  命中的失败信号（针对性改进方向）：")
        lines.extend(
            f"    - {sid}：{FAILURE_SIGNALS[sid]}"
            for sid in review.failure_signals
            if sid in FAILURE_SIGNALS
        )
    detail = review.issues
    if detail:
        lines.append("  评审原文问题（具体证据，逐条对照修正）：")
        lines.extend(f"    - {s}" for s in detail)
    return "\n".join(lines) + "\n"


def _failure_signals_block() -> str:
    return "\n".join(f"  - {sid}：{desc}" for sid, desc in FAILURE_SIGNALS.items())


REVIEW_PROMPT = """你是数据质量评审员。下面是某条流水线生成数据集的抽样，\
以及目标任务画像。请从五个维度评估这批**合成数据**用于微调目标模型的价值（各维 0.0-1.0）。

目标任务：{task_description}
期望 schema（字段→类型）：{target_schema}
质量要点：{quality_criteria}

数据抽样（每行一条 JSON）：
{samples}

评分维度说明：
- correctness_score：解答/标签是否正确、推理过程与最终答案是否自洽一致（错误答案、推理与答案矛盾都要扣分）。
- relevance_score：与目标任务/领域的相关性。
- difficulty_score：对目标模型是否**非平凡且可学**——中间难度带最高；过于简单（无学习价值）或超纲无法学都要扣分。
- diversity_score：抽样批内的多样性、是否存在大量近重复/模板化雷同。
- schema_score：字段齐全度与类型符合度。

常见失败信号（若明确命中，请在 failure_signals 中列出对应编号；没有则留空数组）：
{failure_signal_block}

请只输出一个 JSON 对象：
{{
  "correctness_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "difficulty_score": 0.0-1.0,
  "diversity_score": 0.0-1.0,
  "schema_score": 0.0-1.0,
  "failure_signals": ["F3", ...],
  "issues": ["发现的问题（尽量具体，可引用样本）", ...]
}}
不要输出 JSON 以外的内容。"""


def build_review_prompt(profile: TaskProfile, samples_text: str) -> str:
    return REVIEW_PROMPT.format(
        task_description=profile.task_description,
        target_schema=profile.target_schema,
        quality_criteria=profile.quality_criteria,
        samples=samples_text,
        failure_signal_block=_failure_signals_block(),
    )


# ---------------------------------------------------------------------------
# DiagnosticAgent（Phase-2 慢反馈）
# ---------------------------------------------------------------------------
DIAGNOSTIC_PROMPT = """你是模型失败归因专家。下面是某数据集训练出的模型在下游评估中的失败样例。\
请聚类失败模式，并给出"如何改进数据生成流水线"的可执行建议。

下游任务：{downstream_task}
失败样例（input / prediction=模型完整输出 / extracted_answer=解析出的最终答案 / ground_truth / error_type）：
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
