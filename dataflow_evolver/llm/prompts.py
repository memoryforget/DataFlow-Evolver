"""Prompts for the fixed-corpus PipelineAgent -> ReviewAgent feedback loop."""
from __future__ import annotations

from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.schemas import ReviewResult, TaskProfile


PIPELINE_PROMPT = """为下述目标编写一条完整、可直接运行的 DataFlow 流水线（含自定义算子）。

Target: {target}
Expected outputs: {expected_outputs}
Entry file: {entry_path}
{input_block}

核心要求：
1. 这是固定语料加工/增强任务。`first_entry_file_name` 必须指向上面的 Entry file；流水线必须
   在用户提供的 raw data 上做抽取、清洗、过滤、改写、增强、蒸馏或字段转换，不能绕开输入另起
   一套纯生成数据。
2. 不允许联网拉取外部数据，包括 `load_dataset`、`hf_hub_download`、`requests`/curl，以及
   FileStorage 的 `hf:` / `ms:` 远程入口。调用配置好的 LLM serving API 不受此限。
3. 目标 schema：{target_schema}。最终 JSONL 至少包含这些字段且非空；可以保留额外的中间字段
   和溯源信息。
4. 每个处理步骤放在 `{node_dir}/operators/` 下的独立自定义算子文件中。算子继承
   `OperatorABC`，实现 `run()`；主文件写到 `{node_dir}/pipeline.py`。
5. 同步写 `{node_dir}/decision.json`，schema 必须为：
   {{"ops": [{{"name": "算子类名", "purpose": "作用", "params": {{}}}}, ...],
     "field_flow": "raw fields -> ... -> target fields", "reason": "设计理由"}}
6. LLM 算子使用 `APILLMServing_request`，配置如下：
{llm_block}
7. FileStorage 的 `cache_path` 设为 `{node_dir}/cache`，最终在 `{node_dir}/` 下产出 JSONL。
8. 写完后执行 compile 校验：`pipeline.compile()` →
   `if os.environ.get('DF_COMPILE_ONLY'): sys.exit(0)` → `pipeline.forward()`；然后运行
   `DF_COMPILE_ONLY=1 python {node_dir}/pipeline.py`，修复到 compile 通过。

{recommendation_block}{size_block}{init_block}{parent_block}{review_block}
约束：无头自动化运行，禁止 AskUserQuestion 与 git/gh 子流程。MCP 仅用于查看 DataFlow 算子源码，
不能用于获取新数据。"""


DECISION_SYNC_NOTE = """如果修复改变了算子、字段流或设计取舍，请同步覆盖
`{node_dir}/decision.json`（键名仍为 `ops` / `field_flow` / `reason`）。"""


PIPELINE_SELFCORRECT_PROMPT = """流水线 `{node_dir}/pipeline.py` 执行失败。请先读取
`.claude/skills/dataflow-dev/diagnostics/known_issues.md`，再对照 DataFlow 源码定位并修复。

可查阅：
- `dataflow/utils/storage.py`
- `dataflow/serving/api_llm_serving_request.py`
- `dataflow/core/operator.py`
- `{node_dir}/operators/`

修复后运行 `DF_COMPILE_ONLY=1 python {node_dir}/pipeline.py`，直到 compile 通过。
{decision_sync_note}

错误栈：
```
{traceback}
```

只修复执行问题，不改变任务目标，不联网取数，不进行交互或 git/gh 操作。"""


FAILURE_SIGNALS: dict[str, str] = {
    "F1": "描述过窄：数据围绕极少数结构展开，能力维度单一",
    "F2": "华而不实：表面变化很多但逻辑结构高度一致",
    "F3": "模板锁定：大量近重复或模板化雷同",
    "F4": "覆盖塌陷：未覆盖目标任务的关键能力维度",
    "F5": "约束漂移：偏离目标任务、领域或答案格式",
    "F6": "有形无实：形式满足但训练价值低",
    "F7": "描述空泛：内容语义松散或逐渐跑题",
}

_DIM_LABELS: tuple[tuple[str, str], ...] = (
    ("correctness_score", "正确性"),
    ("relevance_score", "相关性"),
    ("difficulty_score", "难度"),
    ("diversity_score", "多样性"),
    ("schema_score", "schema 符合度"),
)


REVIEW_PROMPT = """你是数据质量评审员。请评估流水线从固定 raw data 处理得到的候选数据，
判断其用于微调目标模型的价值。每个维度取 0.0-1.0。

目标任务：{task_description}
期望 schema：{target_schema}
质量要点：{quality_criteria}

数据抽样（每行一条 JSON）：
{samples}

评分维度：
- correctness_score：标签/解答正确，推理与答案自洽。
- relevance_score：与目标任务和领域相关。
- difficulty_score：具有非平凡但可学习的难度。
- diversity_score：内容与结构多样，不是近重复或模板变体。
- schema_score：必需字段齐全、非空且类型正确；额外元数据不扣分。

失败信号：
{failure_signal_block}

只输出以下 JSON object：
{{
  "correctness_score": 0.0,
  "relevance_score": 0.0,
  "difficulty_score": 0.0,
  "diversity_score": 0.0,
  "schema_score": 0.0,
  "failure_signals": ["F3"],
  "issues": ["具体问题及样本证据"]
}}
没有失败信号时使用空数组。不要输出 JSON 以外的内容。"""


def build_pipeline_prompt(
    profile: TaskProfile,
    corpus: InputCorpus,
    node_dir: str,
    llm_block: str = "",
    parent_code: str | None = None,
    target_samples: int = 0,
    recommended_categories: dict | None = None,
    parent_rationale: str | None = None,
    parent_review: ReviewResult | None = None,
    is_init: bool = False,
    init_reference_code: str | None = None,
    init_instruction: str | None = None,
) -> str:
    if not corpus.grounded:
        raise ValueError("轻量 Pipeline prompt 要求固定 raw corpus")
    return PIPELINE_PROMPT.format(
        target=profile.task_description,
        expected_outputs=list(profile.target_schema),
        entry_path=corpus.entry_path,
        input_block=corpus.prompt_block(),
        target_schema=profile.target_schema,
        node_dir=node_dir,
        llm_block=llm_block or "   使用项目已有的默认 serving 配置。",
        recommendation_block=_recommendation_block(recommended_categories),
        size_block=_size_block(target_samples),
        init_block=(
            _init_block(init_reference_code, init_instruction) if is_init else ""
        ),
        parent_block=_parent_block(parent_code, parent_rationale),
        review_block=(
            _parent_quality_block(parent_review) if parent_review is not None else ""
        ),
    )


def build_selfcorrect_prompt(node_dir: str, traceback: str) -> str:
    return PIPELINE_SELFCORRECT_PROMPT.format(
        node_dir=node_dir,
        traceback=traceback,
        decision_sync_note=DECISION_SYNC_NOTE.format(node_dir=node_dir),
    )


def build_review_prompt(profile: TaskProfile, samples_text: str) -> str:
    signals = "\n".join(f"- {key}: {value}" for key, value in FAILURE_SIGNALS.items())
    return REVIEW_PROMPT.format(
        task_description=profile.task_description,
        target_schema=profile.target_schema,
        quality_criteria=profile.quality_criteria,
        samples=samples_text,
        failure_signal_block=signals,
    )


def _parent_block(code: str | None, rationale: str | None) -> str:
    if not code and not rationale:
        return ""
    lines = ["【上一轮流水线】"]
    if code:
        lines.append(
            "在下面这份已执行流水线基础上做针对性修改，不要原样复制，也不要无依据重写：\n"
            f"```python\n{code}\n```"
        )
    if rationale:
        lines.append(f"上一轮设计理由：{rationale}")
    return "\n".join(lines) + "\n"


def _parent_quality_block(review: ReviewResult) -> str:
    dims = [
        (label, float(getattr(review, attr, 0.0) or 0.0))
        for attr, label in _DIM_LABELS
    ]
    weakest = "、".join(f"{label}({score:.2f})" for label, score in sorted(dims, key=lambda x: x[1])[:2])
    lines = [
        "【上一轮 ReviewAgent 反馈——本轮必须直接回应】",
        f"Review：{review.review_score:.3f}；评审{'通过' if review.passed else '未通过'}",
        "五维：" + " | ".join(f"{label} {score:.2f}" for label, score in dims),
        f"优先改进：{weakest}",
        f"全量硬指标：重复率 {review.dup_rate:.2f} | 空值率 {review.null_rate:.2f} | "
        f"可解析率 {review.parse_valid_rate:.2f} | 污染率 {review.contamination_rate:.2f}",
    ]
    if review.failure_signals:
        lines.append("失败信号：")
        lines.extend(
            f"- {signal}: {FAILURE_SIGNALS[signal]}"
            for signal in review.failure_signals
            if signal in FAILURE_SIGNALS
        )
    if review.issues:
        lines.append("具体问题：")
        lines.extend(f"- {issue}" for issue in review.issues)
    lines.append("只修改能够解释上述反馈的步骤，并在 decision.json 的 reason 中说明本轮改动。")
    return "\n".join(lines) + "\n"


def _recommendation_block(recommended: dict | None) -> str:
    if not recommended:
        return ""
    items = recommended.get("recommended_categories", [])
    if not items:
        return ""
    text = "; ".join(
        f"{item.get('category')} ({item.get('reason', '')})" for item in items[:3]
    )
    return (
        "【DataFlow 源码参考】建议查看类别：" + text + "。可用 `list_operators` 和 "
        "`get_operator_detail` 学习 API 与实现模式，但仍应为当前 raw data 编写自定义算子。\n"
    )


def _size_block(target_samples: int) -> str:
    if target_samples <= 0:
        return ""
    return (
        f"【可调目标条数】从 `DF_TARGET_SAMPLES` 读取，缺省 {target_samples}；不要在代码中写死。"
        "过滤或增强后应尽量接近目标数量。\n"
    )


def _init_block(reference_code: str | None, instruction: str | None) -> str:
    if reference_code:
        return (
            "【初始流水线】将以下参考实现忠实改写成 DataFlow 契约，不改变处理逻辑：\n"
            f"```python\n{reference_code}\n```\n"
        )
    if instruction:
        return f"【初始流水线】额外构建要求：{instruction}\n"
    return "【初始流水线】从固定 raw data 出发，设计一条稳定、可解释的处理基线。\n"
