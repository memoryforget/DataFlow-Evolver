"""PostTrainBench benchmark 可插拔映射表（本地自测适配）。

把每个 benchmark 映射为：
  - 数据合成的一句话 ``--task`` 文案（交 TaskProfilerAgent 自动形式化，不手写 TaskProfile）；
  - 官方 ``evaluate.py`` 的调用参数画像（inspect_ai 家族 vs 自起 vLLM server 家族，
    ``--max-tokens`` vs ``--max-new-tokens``，dev 采样上限，是否需要 GPT judge 等）；
  - 训练数据格式约束（最终答案格式 / SFT system prompt），用于把合成 SFT 数据对齐到
    官方固定 chat template 的评测口径（见 docs/posttrainbench_adaptation.md「格式对齐」）。

其余流程（TaskProfiler / PipelineAgent / ReviewAgent 五维）对各领域通用，无需逐领域改代码。
score_key 统一取 ``accuracy``（PostTrainBench 榜单口径）。

本模块**只依赖标准库**，可被桥接脚本（df-train / df-eval 环境）与单测直接 import。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# evaluate.py 的两类家族：
#   inspect：thin wrapper over inspect_ai（gsm8k/bfcl/humaneval/aime2025/gpqamain），用 --max-tokens。
#   server：脚本自起 vllm serve + GPT judge（healthbench/arenahardwriting），用 --max-new-tokens + OPENAI_API_KEY。
FAMILY_INSPECT = "inspect"
FAMILY_SERVER = "server"

# 借鉴 ANDES 的三档「内容级」输出结构约束（format_requirement）：把结构要求写进**答案文本本身**，
# 与训练数据 schema（我们仍用 ShareGPT）正交。unstructured=不加结构约束（答案格式仍由 answer_format /
# system prompt 管）；code / tool_call 要求答案用对应的 Markdown fence 包裹、前后无多余文字，
# 使合成 output 与评测端解析口径一致。见 docs/posttrainbench_adaptation.md。
FORMAT_UNSTRUCTURED = "unstructured"
FORMAT_CODE = "code"
FORMAT_TOOL_CALL = "tool_call"

FORMAT_REQUIREMENTS: dict[str, str] = {
    FORMAT_UNSTRUCTURED: "",
    FORMAT_CODE: (
        "输出必须是完整、可运行的目标语言代码（本任务为 Python），"
        "并包在 Markdown 代码块 ```python ... ``` 中；代码块前后不得有任何解释或多余文字。"
    ),
    FORMAT_TOOL_CALL: (
        "输出必须是一个合法的工具调用 JSON（含正确的函数名与参数），"
        "并包在 Markdown 代码块 ```json ... ``` 中；代码块前后不得有任何解释或多余文字。"
    ),
}

# limit=-1 表示官方全量（evaluate.py 不传 limit）。dev 阶段用小采样省算力；终评一律 -1。
BENCHMARKS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "task_dir": "gsm8k",
        "format_requirement": FORMAT_UNSTRUCTURED,
        "family": FAMILY_INSPECT,
        "score_key": "accuracy",
        "dev_limit": 150,
        "max_tokens": 4000,
        "needs_judge": False,
        "task_text": (
            "合成 GSM8K 风格的小学数学应用题 SFT 数据集（英文），每条包含题目与分步推理（CoT），"
            "并在回答的最后一行以 \"ANSWER: <数值>\" 给出最终数值答案，用于对给定 base model 做 "
            "post-training 以最大化 GSM8K 准确率。"
        ),
        "answer_format": "回答最后一行必须是 \"ANSWER: <整数或数值>\"（GSM8K 数值精确匹配口径）。",
        "sft_system_prompt": (
            "Solve the following grade-school math problem step by step. "
            "The last line of your response must be of the form \"ANSWER: $ANSWER\" "
            "where $ANSWER is the final numeric answer."
        ),
    },
    "aime2025": {
        "task_dir": "aime2025",
        "format_requirement": FORMAT_UNSTRUCTURED,
        "family": FAMILY_INSPECT,
        "score_key": "accuracy",
        "dev_limit": -1,   # AIME 仅 30 题，dev 直接跑全量即可（便宜）
        "max_tokens": 16000,
        "needs_judge": False,
        "task_text": (
            "合成 AIME（美国数学邀请赛）竞赛级数学题 SFT 数据集（英文），每条包含题目与严谨的分步推理，"
            "并将最终答案（0-999 的整数）放入 \\boxed{} 中，用于 post-training 提升目标模型的 AIME 2025 表现。"
        ),
        "answer_format": "最终答案为 0-999 的整数，必须置于 \\boxed{...} 中。",
        "sft_system_prompt": (
            "Solve the following competition math problem step by step and put the final "
            "integer answer (0-999) inside \\boxed{}."
        ),
    },
    "gpqamain": {
        "task_dir": "gpqamain",
        "format_requirement": FORMAT_UNSTRUCTURED,
        "family": FAMILY_INSPECT,
        "score_key": "accuracy",
        "dev_limit": 50,
        "max_tokens": 16000,
        "needs_judge": False,
        "task_text": (
            "合成研究生水平科学多选题（GPQA 风格，涵盖物理/化学/生物）SFT 数据集（英文），"
            "每条包含题目、四个选项、分步推理，并以 \"ANSWER: <选项字母>\" 结尾，"
            "用于 post-training 提升目标模型的 GPQA 表现。"
        ),
        "answer_format": "四选一多选题，回答以 \"ANSWER: <A/B/C/D>\" 结尾。",
        "sft_system_prompt": (
            "Answer the following multiple-choice science question. Reason step by step, "
            "then end with a line \"ANSWER: X\" where X is one of the option letters."
        ),
    },
    "bfcl": {
        "task_dir": "bfcl",
        "format_requirement": FORMAT_TOOL_CALL,
        "family": FAMILY_INSPECT,
        "score_key": "accuracy",
        "dev_limit": 150,
        "max_tokens": 16000,
        "needs_judge": False,
        "task_text": (
            "合成函数调用（function calling / tool use，BFCL 风格）SFT 数据集（英文）：每条包含一段用户请求、"
            "一组可用工具（函数）的 JSON schema，以及模型应当发出的**恰好一个**工具调用（函数名与参数需正确匹配用户意图）。"
            "用于 post-training 提升目标模型在 Berkeley Function Calling Leaderboard 上的表现。"
        ),
        "answer_format": (
            "助手输出应为**单个**工具调用，函数名与参数（键值）需与用户请求严格一致；"
            "参数不臆造、不缺失。"
        ),
        "sft_system_prompt": (
            "You are a function-calling assistant. Given the user request and the available tools, "
            "emit exactly one correct tool call with the right function name and arguments."
        ),
        "tool_calling": True,
    },
    "humaneval": {
        "task_dir": "humaneval",
        "format_requirement": FORMAT_CODE,
        "family": FAMILY_INSPECT,
        "score_key": "accuracy",
        "dev_limit": 150,
        "max_tokens": 4000,
        "needs_judge": False,
        "task_text": (
            "合成 Python 编程题 SFT 数据集（HumanEval 风格，英文）：每条含一个带 docstring 的函数签名与需求描述，"
            "以及一个通过隐藏单元测试的正确函数实现（完整可运行的 Python 代码）。"
            "用于 post-training 提升目标模型的 HumanEval pass@1。"
        ),
        "answer_format": "助手输出为完整、可运行、能通过单元测试的 Python 函数实现（含必要 import）。",
        "sft_system_prompt": (
            "You are an expert Python programmer. Complete the requested function with a correct, "
            "self-contained implementation that passes the unit tests."
        ),
    },
    "healthbench": {
        "task_dir": "healthbench",
        "format_requirement": FORMAT_UNSTRUCTURED,
        "family": FAMILY_SERVER,
        "score_key": "accuracy",
        "dev_limit": 32,
        "max_tokens": 16384,       # server 家族用 --max-new-tokens
        "needs_judge": True,       # 需要 OPENAI_API_KEY 起 GPT judge
        "judge_workers": 64,
        "task_text": (
            "合成医疗健康问答 SFT 数据集（HealthBench 风格，英文）：每条为真实场景的健康咨询问题与一段"
            "安全、准确、有依据、覆盖关键点的高质量回答，用于 post-training 提升目标模型在 HealthBench 上的评分。"
        ),
        "answer_format": "回答需安全、准确、覆盖评分要点（rubric），避免臆断与有害建议。",
        "sft_system_prompt": (
            "You are a careful medical assistant. Give safe, accurate, well-justified answers that "
            "cover the key clinical points and avoid harmful advice."
        ),
    },
    "arenahardwriting": {
        "task_dir": "arenahardwriting",
        "format_requirement": FORMAT_UNSTRUCTURED,
        "family": FAMILY_SERVER,
        "score_key": "accuracy",   # 榜单取 win rate 写入 accuracy 键
        "dev_limit": 32,
        "max_tokens": 16384,
        "needs_judge": True,
        "judge_workers": 64,
        "task_text": (
            "合成高质量开放式写作指令 SFT 数据集（Arena-Hard-v2.0 Writing 风格，英文）：每条为一个有挑战性的写作/表达类"
            "指令与一段结构清晰、内容充实、文采与逻辑兼备的高质量回答，用于 post-training 提升目标模型的写作偏好胜率。"
        ),
        "answer_format": "回答需结构清晰、切题充实、语言流畅，直接完成写作任务。",
        "sft_system_prompt": (
            "You are an expert writer. Produce a well-structured, substantive, and engaging response "
            "that fully addresses the writing instruction."
        ),
    },
}


def list_benchmarks() -> list[str]:
    """返回所有已支持的 benchmark 名。"""
    return sorted(BENCHMARKS.keys())


def get_benchmark(name: str) -> dict[str, Any]:
    """按名取 benchmark 配置；未知名抛 KeyError（附可选项）。"""
    key = (name or "").strip().lower()
    if key not in BENCHMARKS:
        raise KeyError(
            f"未知 benchmark：{name!r}；可选：{', '.join(list_benchmarks())}"
        )
    return BENCHMARKS[key]


def format_requirement_for(name: str) -> str:
    """返回该 benchmark 的 format_requirement 档位（unstructured/code/tool_call）。"""
    return get_benchmark(name).get("format_requirement", FORMAT_UNSTRUCTURED)


def format_instruction_for(name: str) -> str:
    """返回该档位对应的内容级结构约束文案；unstructured 返回空串。"""
    return FORMAT_REQUIREMENTS.get(format_requirement_for(name), "")


def composed_task_text(name: str) -> str:
    """给 TaskProfilerAgent 用的 ``--task`` 文案：基础任务描述 + 结构约束（若有）。

    unstructured 档不追加，保持原文案；code/tool_call 追加一行【输出格式要求】，
    使合成数据的 output 结构与评测端解析口径一致。
    """
    bench = get_benchmark(name)
    text = str(bench["task_text"])
    instr = format_instruction_for(name)
    if instr:
        text = f"{text}\n【输出格式要求】{instr}"
    return text


def resolve_benchmark_name(name: str | None, ptb_task_dir: str | None) -> str:
    """确定 benchmark 名：显式 name 优先；否则用 ptb_task_dir 的目录名反查。"""
    if name and name.strip():
        return name.strip().lower()
    if ptb_task_dir:
        leaf = Path(ptb_task_dir.rstrip("/")).name.strip().lower()
        if leaf in BENCHMARKS:
            return leaf
    raise ValueError(
        "无法确定 benchmark：请显式传 --benchmark，或让 --ptb-task-dir 的目录名与已知 benchmark 一致。"
    )


def templates_dir_for(ptb_task_dir: str) -> Path:
    """由任务目录（.../src/eval/tasks/<name>）推出共享模板目录（.../src/eval/templates）。"""
    return (Path(ptb_task_dir).resolve().parent.parent / "templates")


def build_eval_argv(
    name: str,
    *,
    model_path: str,
    templates_dir: str,
    limit: int,
    json_output_file: str,
    gpu_memory_utilization: float | None = None,
    judge_workers: int | None = None,
) -> list[str]:
    """构造官方 evaluate.py 的命令行参数（不含解释器与脚本路径本身）。

    - inspect 家族用 ``--max-tokens``；server 家族用 ``--max-new-tokens``（并可带 ``--judge-workers``）。
    - limit=-1（全量）时不传 ``--limit``（evaluate.py 语义：不传=全量）。
    """
    bench = get_benchmark(name)
    argv = [
        "--model-path", str(model_path),
        "--templates-dir", str(templates_dir),
        "--json-output-file", str(json_output_file),
    ]
    if limit is not None and int(limit) != -1:
        argv += ["--limit", str(int(limit))]

    if bench["family"] == FAMILY_SERVER:
        argv += ["--max-new-tokens", str(bench["max_tokens"])]
        jw = judge_workers if judge_workers is not None else bench.get("judge_workers")
        if jw is not None:
            argv += ["--judge-workers", str(int(jw))]
    else:
        argv += ["--max-tokens", str(bench["max_tokens"])]

    if gpu_memory_utilization is not None:
        argv += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    return argv
