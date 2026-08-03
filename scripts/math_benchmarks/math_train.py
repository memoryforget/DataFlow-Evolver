"""MATH 风格题集的加载与 prompt 构造（dev 评测用）。

典型数据源：MATH competition train split（7500 条，7 子学科 × 5 难度等级）。这些是公开训练
语料（非 benchmark test set），用作 MCTS dev 奖励信号不会污染终评。
答案格式：LaTeX \\boxed{} 内为最终答案；评测 prompt 复用 Qwen2.5-Math 的模板格式。

**本模块不做抽样，也没有默认题库路径。** 评哪些题是框架的决策：`core/dev_set.py` 在启动时
按 (source_sha, seed, size) 物化成一份文件并写进 manifest，经 `--dev-set` 下传。脚本这边
再留一套平行的抽样档位，只会让两个口径各说各话（曾经真的发生过，见 design-decisions.md）。
"""
from __future__ import annotations

from pathlib import Path

from .answer_extract import extract_boxed


def load_math_train(data_path: str, cache_dir: str | None = None) -> list[dict]:
    """读取题集文件，返回 [{"question", "gold_answer", "level", "subject"}, ...]。

    `data_path` 是必填的：调用方（框架）已经决定好评哪些题，本函数原样读完返回，
    不抽样、不截断、不回退到任何默认路径。
    """
    import json

    if not data_path:
        raise ValueError(
            "load_math_train 需要显式的题集路径：评哪些题由框架的 trainer.dev_set 决定，"
            "本模块不再提供默认题库或内置抽样。"
        )

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"dev 题集不存在：{path}\n"
            f"它由框架按 trainer.dev_set 物化（见 core/dev_set.py）。源文件不入库，"
            f"需自行准备：从 Qwen2.5-Math 仓库取 evaluation/data/math/train.jsonl 放到 "
            f"data/math_train.jsonl，或改 configs/*.yaml 的 trainer.dev_set.source "
            f"指向别处。详见 data/README.md。"
        )

    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = row.get("problem", "")
            gold = row.get("answer", "")
            if not gold:
                gold = extract_boxed(row.get("solution", "")) or ""
            examples.append({
                "question": question,
                "gold_answer": gold,
                "level": row.get("level", ""),
                "subject": row.get("subject", ""),
            })

    return examples


def build_prompt(question: str) -> str:
    """构造评测 prompt（复用 Qwen2.5-Math 的 qwen25-math-cot 模板格式）。

    注：这里只返回 user message 内容，chat template 由 vLLM tokenizer 自动处理。
    system prompt 需在推理时单独设置为：
    "Please reason step by step, and put your final answer within \\boxed{}"
    """
    return question
