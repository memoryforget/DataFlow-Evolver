"""Prompts for the fixed-corpus PipelineAgent -> ReviewAgent feedback loop."""
from __future__ import annotations

from dataflow_evolver.core.downstream_eval import DownstreamEvaluationResult
from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.schemas import ReviewResult, TaskSpec
_BAD_CASE_RESPONSE_HEAD_CHARS = 1200
_BAD_CASE_RESPONSE_TAIL_CHARS = 1800
_RUNAWAY_RESPONSE_HEAD_CHARS = 500
_RUNAWAY_RESPONSE_TAIL_CHARS = 500


_PIPELINE_RUNTIME_BOUNDARY = """
【最高优先级运行边界】
先加载并遵循 `$dataflow-evolver-pipeline`。DataFlow 运行时只使用已安装的
`open-dataflow` 包；不要读取、搜索或复制任何 DataFlow 源码。算子和 Pipeline
由本任务本地生成。通用 API、compile 和 Serving 响应兼容契约均以该 skill 为准；
本 Prompt 只负责本轮数据目标、反馈、父代复用和动态模型配置。检查范围限于当前
iteration 的任务产物，禁止自行执行 Pipeline 或做宽泛探索。
"""

PIPELINE_PROMPT = _PIPELINE_RUNTIME_BOUNDARY + "\n\n" + """为下述目标编写一条完整、可直接运行的 DataFlow 流水线（含自定义算子）。

Target: {target}
Expected outputs: {expected_outputs}
Entry file: {entry_path}
{input_block}

核心要求：
1. 这是固定语料加工/增强任务。将上面的 Entry file 作为 `FrameworkStepCache.from_env()` 的
   `raw_entry_path`；`first_entry_file_name` 必须使用框架返回的 `entry_path`。未命中复用时该路径
   就是固定 raw data。流水线不能绕开输入另起一套纯生成数据。
2. 不允许联网拉取外部数据，包括 `load_dataset`、`hf_hub_download`、`requests`/curl，以及
   FileStorage 的 `hf:` / `ms:` 远程入口。调用配置好的 LLM serving API 不受此限。
3. 目标 schema：{target_schema}。最终 JSONL 至少包含这些字段且非空；可以保留额外的中间字段
   和溯源信息。
4. 每个处理步骤放在 `{iteration_dir}/operators/` 下的独立自定义算子文件中，主文件写到
   `{iteration_dir}/pipeline.py`；算子和 Pipeline 的通用实现契约严格遵循专用 skill。
5. 同步写 `{iteration_dir}/decision.json`，schema 必须为：
   {{"ops": [{{"name": "算子类名", "purpose": "作用", "params": {{}}}}, ...],
     "field_flow": "raw fields -> ... -> target fields", "reason": "设计理由"}}
6. 以下动态配置只声明本轮可用的 LLM Serving 能力，不预设 Pipeline 是否调用 LLM、
   在哪个步骤调用或调用结果承担什么职责，你需要根据任务与反馈自行设计；若使用 LLM，
   必须在 Pipeline 层按专用 skill 构造并注入 Serving，且不得根据模型名称自行推断响应模式：
{llm_block}
7. FileStorage 的 `cache_path` 设为 `{iteration_dir}/cache`；正式运行产生的最后一个 step JSONL 是本轮最终数据集产物。
8. 主入口必须使用专用 skill 中唯一的 framework-owned compile/forward 契约。只写入所需产物，
   不要自行执行 `pipeline.py`（包括 compile-only 或 smoke test）；框架会在 Agent 返回后依次执行
   确定性的 artifact 校验、compile 预检和正式运行。
9. 以解决 Review、下游评测和 bad cases 暴露的问题为首要目标，不以最少代码、文件或算子为目标。
   根据职责边界和反馈自由增加、删除、拆分或合并算子；不要为了缓存复用而保留不合理的父流水线结构。
{init_block}{parent_block}{review_block}{downstream_block}{history_block}
约束：无头自动化运行，禁止 AskUserQuestion 与 git/gh 子流程，不能获取新数据；所有实现
依据均来自专用 skill、本轮 Prompt 和当前 iteration 产物。"""


DECISION_SYNC_NOTE = """如果修复改变了算子、字段流或设计取舍，请同步覆盖
`{iteration_dir}/decision.json`（键名仍为 `ops` / `field_flow` / `reason`）。"""


PIPELINE_SELFCORRECT_PROMPT = _PIPELINE_RUNTIME_BOUNDARY + "\n\n" + """流水线 `{iteration_dir}/pipeline.py` 未通过框架检查。
失败阶段：`{failure_stage}`。

只依据专用 skill、错误栈、逐步诊断以及当前 iteration 的 `pipeline.py` 和 `operators/`
定位问题。

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
    "F1": "抽样描述过窄：可见样本围绕极少数结构展开",
    "F2": "抽样华而不实：表面变化很多但可见逻辑结构高度一致",
    "F3": "抽样模板锁定：可见样本存在近重复或模板化雷同",
    "F4": "抽样覆盖不足：可见样本缺少目标任务的关键能力维度",
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


REVIEW_PROMPT = """你是数据质量评审员。下面是候选数据集的有界随机抽样。请评估该抽样中
可直接观察到的微调质量，每个维度取 0.0-1.0。F1-F4 只能在抽样本身提供重复、模板化或覆盖不足
的直接证据时报告，不能外推整个数据集；全量多样性由框架使用 dataset embedding 单独评估。

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
        init_block=(
            "【初始流水线】从固定 raw data 出发，设计稳定、可解释的处理基线。\n"
            if is_init else ""
        ),
        parent_block=_parent_block(parent_code, parent_rationale, iteration_dir),
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
) -> str:
    return PIPELINE_SELFCORRECT_PROMPT.format(
        iteration_dir=iteration_dir,
        failure_stage=failure_stage,
        traceback=traceback,
        diagnostics=diagnostics,
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


def _parent_block(code: str | None, rationale: str | None, iteration_dir: str) -> str:
    if not code and not rationale:
        return ""
    lines = ["【上一轮流水线】"]
    if code:
        lines.append(
            "把下面这份已执行流水线作为经过验证的参考实现。根据反馈自由保留、替换、拆分、合并、增加或删除算子；不要为了缓存复用而维持原结构，也不要无依据重写：\n"
            f"```python\n{code}\n```"
        )
    if rationale:
        lines.append(f"上一轮设计理由：{rationale}")
    if code:
        lines.append(
            "【Framework-owned 父轮复用】只判断哪些连续前缀算子在语义上保持不变，并复制这些算子"
            f"的本地源码到当前 `{iteration_dir}/operators/`。不要读取、选择、验证或硬编码父轮 cache "
            "JSONL；不要通过 `pipeline_step_stepN` 推断 producer，也不要自行实现 cache hit/miss。\n"
            "每条 Pipeline 都必须使用 skill 中的 `FrameworkStepCache.from_env()` 标准适配器。框架会在"
            "执行前比较 raw input、逻辑算子顺序、源码及本地依赖、初始化调用与参数 fingerprint，并从"
            "父轮 `step_manifest.json` 注入经过验证的 entry path 和 prefix count；无法证明时自动从 raw "
            "input 完整重算。Pipeline 只按 `should_run(logical_index)` 机械跳过框架确认的前缀。\n"
            "在 `decision.json.reason` 只记录哪些前缀算子未变化以及首个变化算子；缓存 artifact、step "
            "编号、lineage 与命中结论全部由框架记录，Agent 不得自行决定。源码不参与身份校验；保持相同的算子名、顺序、"
            "`decision.json.params` 与初始化调用即声明语义未变并授权框架复用。若算子实现或本地依赖的行为发生变化，"
            "必须更新算子类名、params 或初始化参数中的至少一项。"
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
        "完整、未截断的结果由框架单独保留用于人工审计；不要读取完整 result.json。"
        "下面提供的有界投影已经是本轮允许使用的全部下游诊断证据。",
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
            f"correct_count={benchmark.correct_count}; "
            f"total_count={benchmark.total_count}; "
            f"question_count={benchmark.question_count}; "
            f"n_sampling={benchmark.n_sampling}; "
            f"incorrect_count={benchmark.incorrect_count}; "
            f"wrong_answer_count={benchmark.wrong_answer_count}; "
            f"parse_failure_count={benchmark.parse_failure_count}; "
            f"runaway_count={benchmark.runaway_count}"
        )
        for index, case in enumerate(benchmark.bad_cases, start=1):
            is_runaway = case.length_truncated or case.finish_reason not in {
                "",
                "stop",
            }
            case_type = (
                "generation_runaway"
                if is_runaway
                else "parse_failure"
                if not case.parsed_answer.strip()
                else "wrong_answer"
            )
            response_projection, omitted_chars = _project_bad_case_response(
                case.model_response,
                length_truncated=is_runaway,
            )
            token_count = (
                str(case.response_token_count)
                if case.response_token_count is not None
                else "unknown"
            )
            finish_reason = case.finish_reason or "unknown"
            runaway = "true" if is_runaway else "false"
            lines.extend(
                [
                    f"Bad case {index}",
                    "Question:",
                    case.question,
                    (
                        "Response metadata: "
                        f"case_type={case_type}, "
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


def _project_bad_case_response(
    response: str,
    *,
    length_truncated: bool = False,
) -> tuple[str, int]:
    """Render bounded evidence while keeping the full response in result.json."""
    head_chars = (
        _RUNAWAY_RESPONSE_HEAD_CHARS
        if length_truncated
        else _BAD_CASE_RESPONSE_HEAD_CHARS
    )
    tail_chars = (
        _RUNAWAY_RESPONSE_TAIL_CHARS
        if length_truncated
        else _BAD_CASE_RESPONSE_TAIL_CHARS
    )
    limit = head_chars + tail_chars
    if len(response) <= limit:
        return response, 0
    omitted = len(response) - limit
    marker = (
        f"\n\n...[{omitted} chars omitted from prompt; full response retained for human audit only]...\n\n"
    )
    return (
        response[:head_chars]
        + marker
        + response[-tail_chars:],
        omitted,
    )
