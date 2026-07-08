"""MATH train 题库加载与判分（dev 评测用）。

数据源：MATH competition train split (7500 条，含 7 子学科 × 5 难度等级)。
这些是公开训练语料（非 benchmark test set），用作 MCTS dev 奖励信号不会污染终评。
答案格式：LaTeX \\boxed{} 内为最终答案。
评测 prompt 复用 Qwen2.5-Math 的模板格式。

默认只加载固定的 500 条子集（seed=42 随机采样），加速 MCTS 迭代。
设置环境变量 MATH_TRAIN_SAMPLE_SIZE=0 可加载全量 7500 条。
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from .answer_extract import extract_answer, extract_boxed, is_equiv

# 默认路径：项目内 data/math_train.jsonl
_DEFAULT_MATH_TRAIN_PATH = str(
    Path(__file__).resolve().parents[2] / "data" / "math_train.jsonl"
)

_SAMPLE_SIZE = int(os.environ.get("MATH_TRAIN_SAMPLE_SIZE", "500"))
_SAMPLE_SEED = 42


def load_math_train(
    data_path: str | None = None, cache_dir: str | None = None
) -> list[dict]:
    """加载 MATH train set（本地文件）。返回 [{"question", "gold_answer", "level", "subject"}, ...]

    默认返回固定的 500 条子集（seed=42），保证每次评测结果可复现。
    设置环境变量 MATH_TRAIN_SAMPLE_SIZE=0 返回全量。
    """
    import json

    path = Path(data_path) if data_path else Path(_DEFAULT_MATH_TRAIN_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"MATH train 文件不存在：{path}\n"
            f"请将 Qwen2.5-Math/evaluation/data/math/train.jsonl 复制到 data/math_train.jsonl，"
            f"或通过 MATH_TRAIN_PATH 环境变量指定路径。"
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

    if _SAMPLE_SIZE > 0 and len(examples) > _SAMPLE_SIZE:
        rng = random.Random(_SAMPLE_SEED)
        examples = rng.sample(examples, _SAMPLE_SIZE)

    return examples


def build_prompt(question: str) -> str:
    """构造评测 prompt（复用 Qwen2.5-Math 的 qwen25-math-cot 模板格式）。

    注：这里只返回 user message 内容，chat template 由 vLLM tokenizer 自动处理。
    system prompt 需在推理时单独设置为：
    "Please reason step by step, and put your final answer within \\boxed{}"
    """
    return question


# Qwen2.5-Math 评测用的 system prompt
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def score_predictions(
    predictions: list[str], examples: list[dict]
) -> tuple[float, list[dict]]:
    """判分。返回 (accuracy, bad_cases)。"""
    correct = 0
    bad_cases: list[dict] = []
    for pred_text, ex in zip(predictions, examples):
        pred_answer = extract_answer(pred_text, mode="boxed")
        if pred_answer is None:
            pred_answer = extract_answer(pred_text, mode="number")
        gold = ex["gold_answer"]
        if is_equiv(pred_answer or "", gold):
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred_answer or "(extraction failed)",
                "ground_truth": gold,
                "error_type": "wrong_answer",
            })
    accuracy = correct / len(examples) if examples else 0.0
    return accuracy, bad_cases
