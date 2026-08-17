"""Prompts for the fixed-corpus PipelineAgent -> ReviewAgent feedback loop."""
from __future__ import annotations

from dataflow_evolver.core.downstream_eval import DownstreamEvaluationResult
from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.schemas import ReviewResult, TaskSpec
from dataflow_evolver.llm.skill_paths import skill_root_for_backend


_BAD_CASE_RESPONSE_HEAD_CHARS = 1200
_BAD_CASE_RESPONSE_TAIL_CHARS = 1800


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
4. 每个处理步骤放在 `{iteration_dir}/operators/` 下的独立自定义算子文件中。算子继承
   `OperatorABC`，实现 `run()`；主文件写到 `{iteration_dir}/pipeline.py`。
5. 同步写 `{iteration_dir}/decision.json`，schema 必须为：
   {{"ops": [{{"name": "算子类名", "purpose": "作用", "params": {{}}}}, ...],
     "field_flow": "raw fields -> ... -> target fields", "reason": "设计理由"}}
6. LLM 算子使用 `APILLMServing_request`，配置如下：
{llm_block}
7. FileStorage 的 `cache_path` 设为 `{iteration_dir}/cache`，最终在 `{iteration_dir}/` 下产出 JSONL。
8. 主入口必须严格按 `pipeline.compile()` →
   `if os.environ.get('DF_COMPILE_ONLY'): sys.exit(0)` → `pipeline.forward()` 排列。只写入所需产物，
   不要自行执行 `pipeline.py`（包括 compile-only 或 smoke test）；框架会在 Agent 返回后依次执行
   确定性的 artifact 校验、compile 预检和正式运行。
9. 从上一轮流水线演化时，优先复用可证明完全未变化的连续前缀中间 JSONL，避免重复执行昂贵
   的确定性算子或 LLM 算子。只有同时满足以下条件才允许复用：raw Entry file 及其内容未变；
   前缀中每个算子的源码、初始化参数、`run()` 参数、字段流，以及所用模型、prompt 和 serving
   参数均未变；父轮对应 step JSONL 存在且可解析。任一条件无法确认时必须从 raw input 重算。
   复用实现必须保留从 raw input 完整执行的 fallback，不能让最终 Pipeline 永久依赖父轮目录；
   也不能把父轮最终数据伪装成新的 raw input。请在 `decision.json.reason` 中记录复用截止 step、
   父轮缓存路径、验证条件与 fallback。缓存命中时，必须用父轮该 step JSONL 作为 suffix storage
   的 `first_entry_file_name`，并且只调用变化步骤及其下游算子，不能再次调用已复用的前缀算子；
   缓存未命中时则从 raw input 执行完整 Pipeline。两条分支都要打印一行
   `DF_PREFIX_CACHE_REUSE hit|miss cutoff=<step> source=<path>`，使框架日志可以审计是否真正跳过。
   只要某一步发生变化，该步及其所有下游步骤都必须重算。

{recommendation_block}{init_block}{parent_block}{review_block}{downstream_block}{history_block}
约束：无头自动化运行，禁止 AskUserQuestion 与 git/gh 子流程。MCP 仅用于查看 DataFlow 算子源码，
不能用于获取新数据。"""


DECISION_SYNC_NOTE = """如果修复改变了算子、字段流或设计取舍，请同步覆盖
`{iteration_dir}/decision.json`（键名仍为 `ops` / `field_flow` / `reason`）。"""


PIPELINE_SELFCORRECT_PROMPT = """流水线 `{iteration_dir}/pipeline.py` 未通过框架检查。
失败阶段：`{failure_stage}`。

请先读取
`{skill_root}/diagnostics/known_issues.md`，再对照 DataFlow 源码定位并修复。

可查阅：
- `dataflow/utils/storage.py`
- `dataflow/serving/api_llm_serving_request.py`
- `dataflow/core/operator.py`
- `{iteration_dir}/operators/`

只修改 `pipeline.py`、`operators/` 和必要的 `decision.json`，不要自行执行 `pipeline.py`、
compile-only 或 smoke test。Agent 返回后，框架会重新执行 artifact 校验、compile 预检与正式运行。
{decision_sync_note}

错误栈：
```
{traceback}
```

逐步产物诊断：
```
{diagnostics}
```

按失败阶段定位：`artifact_validation` 优先修正语法及 compile/guard/forward 主入口契约；
`compile` 优先检查 import、算子初始化、参数和字段流；`execution` / `output_validation` 则先找
第一个行数变为 0、字段丢失或 JSONL 异常的步骤，修复产生问题的上游算子，不要只在下游吞掉异常。

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
    ("schema_score", "schema 符合度"),
)


REVIEW_PROMPT = """你是数据质量评审员。下面只是候选数据集的随机抽样。请评估单条样本用于微调
目标模型的质量，每个维度取 0.0-1.0。不要从该抽样推断或评分整个数据集的多样性；
框架会用同一批 dataset embedding 单独计算多样性指标。

目标任务：{task_description}
期望 schema：{target_schema}
质量要点：{quality_criteria}

数据抽样（每行一条 JSON）：
{samples}

评分维度：
- correctness_score：标签/解答正确，推理与答案自洽。
- relevance_score：与目标任务和领域相关。
- difficulty_score：具有非平凡但可学习的难度。
- schema_score：必需字段齐全、非空且类型正确；额外元数据不扣分。

失败信号：
{failure_signal_block}

只输出以下 JSON object：
{{
  "correctness_score": 0.0,
  "relevance_score": 0.0,
  "difficulty_score": 0.0,
  "schema_score": 0.0,
  "failure_signals": ["F3"],
  "issues": ["具体问题及样本证据"]
}}
没有失败信号时使用空数组。不要输出 JSON 以外的内容。"""


def build_pipeline_prompt(
    task: TaskSpec,
    corpus: InputCorpus,
    iteration_dir: str,
    llm_block: str = "",
    parent_code: str | None = None,
    recommended_categories: dict | None = None,
    parent_rationale: str | None = None,
    parent_review: ReviewResult | None = None,
    downstream_feedback: DownstreamEvaluationResult | None = None,
    evolution_history: list[dict] | None = None,
    is_init: bool = False,
) -> str:
    if not corpus.grounded:
        raise ValueError("轻量 Pipeline prompt 要求固定 raw corpus")
    return PIPELINE_PROMPT.format(
        target=task.task_description,
        expected_outputs=list(task.target_schema),
        entry_path=corpus.entry_path,
        input_block=corpus.prompt_block(),
        target_schema=task.target_schema,
        iteration_dir=iteration_dir,
        llm_block=llm_block or "   使用项目已有的默认 serving 配置。",
        recommendation_block=_recommendation_block(recommended_categories),
        init_block=(
            "【初始流水线】从固定 raw data 出发，设计稳定、可解释的处理基线。\n"
            if is_init else ""
        ),
        parent_block=_parent_block(parent_code, parent_rationale),
        review_block=(
            _parent_quality_block(parent_review) if parent_review is not None else ""
        ),
        downstream_block=_downstream_feedback_block(downstream_feedback),
        history_block=_history_block(evolution_history),
    )

def build_selfcorrect_prompt(
    iteration_dir: str,
    traceback: str,
    diagnostics: str = "not available",
    failure_stage: str = "execution",
    skill_root: str | None = None,
) -> str:
    return PIPELINE_SELFCORRECT_PROMPT.format(
        iteration_dir=iteration_dir,
        failure_stage=failure_stage,
        traceback=traceback,
        diagnostics=diagnostics,
        skill_root=skill_root or skill_root_for_backend("claude"),
        decision_sync_note=DECISION_SYNC_NOTE.format(iteration_dir=iteration_dir),
    )


def build_review_prompt(task: TaskSpec, samples_text: str) -> str:
    signals = "\n".join(f"- {key}: {value}" for key, value in FAILURE_SIGNALS.items())
    return REVIEW_PROMPT.format(
        task_description=task.task_description,
        target_schema=task.target_schema,
        quality_criteria=task.quality_criteria,
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
    if code:
        lines.append(
            "上一轮代码中的绝对 cache_path 可用于定位该轮逐步 JSONL。先比较连续前缀，再决定"
            "是否启用上面的受约束缓存复用；不要仅因为类名相同就认定算子等价。"
        )
    return "\n".join(lines) + "\n"


def _parent_quality_block(review: ReviewResult) -> str:
    dims = [
        (label, float(getattr(review, attr, 0.0) or 0.0))
        for attr, label in _DIM_LABELS
    ]
    lines = [
        "【当前最佳 incumbent 的 ReviewAgent 反馈——本轮必须直接回应】",
        f"Review：{review.review_score:.3f}；评审{'通过' if review.passed else '未通过'}",
        "抽样四维：" + " | ".join(f"{label} {score:.2f}" for label, score in dims),
        f"全量硬指标：重复率 {review.dup_rate:.2f} | 空值率 {review.null_rate:.2f} | "
        f"可解析率 {review.parse_valid_rate:.2f} | 污染率 {review.contamination_rate:.2f}",
    ]
    dataset_quality = review.dataset_quality
    if dataset_quality.enabled:
        if dataset_quality.status == "ok" and dataset_quality.mmd is not None:
            delta_mmd = (
                f"{dataset_quality.delta_mmd_vs_parent:+.6f}"
                if dataset_quality.delta_mmd_vs_parent is not None
                else "baseline"
            )
            diversity = dataset_quality.embedding_diversity
            if diversity:
                delta_diversity = dataset_quality.delta_embedding_diversity_vs_parent
                delta_vendi = (
                    f"{delta_diversity['cosine_vendi_ratio']:+.6f}"
                    if "cosine_vendi_ratio" in delta_diversity
                    else "baseline"
                )
                delta_p95 = (
                    f"{delta_diversity['nearest_neighbor_cosine_p95']:+.6f}"
                    if "nearest_neighbor_cosine_p95" in delta_diversity
                    else "baseline"
                )
                lines.append(
                    f"Dataset embedding：ΔMMD {delta_mmd} | "
                    f"normalized Vendi {diversity.get('cosine_vendi_ratio', 0.0):.6f} "
                    f"(Δ {delta_vendi}) | "
                    f"NN cosine p95 {diversity.get('nearest_neighbor_cosine_p95', 0.0):.6f} "
                    f"(Δ {delta_p95})"
                )
                lines.append(
                    "ΔMMD 越负越好；normalized Vendi 越高越好；NN cosine p95 越低越好。"
                )
        else:
            lines.append(
                f"Dataset-level DAS：状态 {dataset_quality.status}；"
                f"原因：{dataset_quality.error or '未知'}。"
                "请优先保证输出至少含足够数量且字段可编码的有效记录。"
            )
    if review.issues:
        lines.append("具体问题：")
        lines.extend(f"- {issue}" for issue in review.issues)
    lines.append("只修改能够解释上述反馈的步骤，并在 decision.json 的 reason 中说明本轮改动。")
    return "\n".join(lines) + "\n"


def _history_block(history: list[dict] | None) -> str:
    if not history:
        return ""
    visible = history[-12:]
    lines = [
        "【简要进化历史——不要重复 rejected 尝试的同类策略】",
        "这里只记录 decision 摘要，不代表要继承 rejected pipeline。",
    ]
    if len(history) > len(visible):
        lines.append(f"- 更早的 {len(history) - len(visible)} 次尝试仅保存在运行记录中。")
    for item in visible:
        score = item.get("review_score")
        score_text = f"{float(score):.3f}" if score is not None else "n/a"
        operators = ">".join(str(name) for name in item.get("operators", [])) or "unknown"
        issues = "；".join(str(issue) for issue in item.get("issues", [])) or "none"
        lines.append(
            f"- iter {item.get('iteration')} [{item.get('outcome')}], score={score_text}, "
            f"ops={operators}, decision={item.get('decision') or 'none'}, issues={issues}"
        )
    return "\n".join(lines) + "\n"


def _downstream_feedback_block(
    feedback: DownstreamEvaluationResult | None,
) -> str:
    if feedback is None:
        return ""
    if not feedback.succeeded:
        return (
            "【下游 SFT/评测检查点】外部评测未成功，因此本轮没有可用的下游质量信号。\n"
            f"失败原因：{feedback.error or 'unknown error'}。不要根据基础设施失败臆测数据问题。\n"
        )

    if feedback.stage == "baseline":
        title = "【第 0 轮下游模型诊断——初始流水线必须分析并回应】"
        checkpoint = (
            "检查点：迭代开始前；未进行 SFT；"
            f"下游模型：{feedback.model}"
        )
    else:
        title = "【当前最佳 incumbent 的下游 SFT/评测反馈——本轮必须分析并回应】"
        checkpoint = (
            f"检查点：完成第 {feedback.checkpoint_iteration} 轮后；"
            f"被评测 incumbent：第 {feedback.incumbent_iteration} 轮；"
            f"下游模型：{feedback.model}"
        )

    lines = [
        title,
        checkpoint,
        "以下 bad cases 是诊断证据，不是可复制进训练数据的新数据源。禁止复述、硬编码或"
        "针对这些 benchmark 题目构造答案；应归纳能力缺口并改进 raw-data pipeline。",
        (
            "完整、未截断的下游证据保存在："
            f"{feedback.result_path or '[external result path not provided]'}。"
            "下面只提供有界首尾摘录；如需确认完整推理，使用文件工具读取该结果文件。"
        ),
    ]
    for benchmark_name, benchmark in feedback.benchmarks.items():
        delta = feedback.score_deltas.get(benchmark_name)
        delta_text = f", delta_vs_previous={delta:+.6f}" if delta is not None else ""
        baseline_delta = feedback.score_deltas_vs_baseline.get(benchmark_name)
        baseline_delta_text = (
            f", delta_vs_untrained_baseline={baseline_delta:+.6f}"
            if baseline_delta is not None
            else ""
        )
        lines.append(
            f"\n### {benchmark_name}: {benchmark.metric}={benchmark.score:.6f}"
            f"{delta_text}{baseline_delta_text}; "
            f"incorrect_count={benchmark.incorrect_count}"
        )
        for index, case in enumerate(benchmark.bad_cases, start=1):
            response_projection, omitted_chars = _project_bad_case_response(
                case.model_response
            )
            token_count = (
                str(case.response_token_count)
                if case.response_token_count is not None
                else "unknown"
            )
            finish_reason = case.finish_reason or "unknown"
            runaway = "true" if case.length_truncated else "false"
            lines.extend(
                [
                    f"Bad case {index}",
                    "Question:",
                    case.question,
                    (
                        "Response metadata: "
                        f"chars={len(case.model_response)}, tokens={token_count}, "
                        f"finish_reason={finish_reason}, "
                        f"length_truncated={str(case.length_truncated).lower()}, "
                        f"possible_runaway_generation={runaway}, "
                        f"prompt_omitted_chars={omitted_chars}"
                    ),
                    "Model response (bounded prompt projection):",
                    response_projection,
                    "Parsed answer:",
                    case.parsed_answer or "[empty; evaluator extracted no answer]",
                    "Correct answer:",
                    case.correct_answer,
                ]
            )
    if feedback.summary:
        lines.extend(["\n外部评测摘要：", feedback.summary])
    lines.append(
        "请把各 benchmark 的共同与特有失败模式映射到可执行的过滤、重写、增强、难度或覆盖策略，"
        "并在 decision.json 的 reason 中说明哪些改动由该反馈触发。"
    )
    return "\n".join(lines) + "\n"


def _project_bad_case_response(response: str) -> tuple[str, int]:
    """Render bounded evidence while keeping the full response in result.json."""
    limit = _BAD_CASE_RESPONSE_HEAD_CHARS + _BAD_CASE_RESPONSE_TAIL_CHARS
    if len(response) <= limit:
        return response, 0
    omitted = len(response) - limit
    marker = (
        f"\n\n...[{omitted} chars omitted from prompt; read the full result file]...\n\n"
    )
    return (
        response[:_BAD_CASE_RESPONSE_HEAD_CHARS]
        + marker
        + response[-_BAD_CASE_RESPONSE_TAIL_CHARS:],
        omitted,
    )


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
