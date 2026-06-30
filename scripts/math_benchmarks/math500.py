"""MATH-500 benchmark 加载与判分。

数据源：HuggingFace `HuggingFaceH4/MATH-500` (500 条竞赛级数学题)。
答案格式：LaTeX \\boxed{} 内为最终答案。
"""
from __future__ import annotations

from .answer_extract import extract_answer, extract_boxed, is_equiv


def load_math500(cache_dir: str | None = None) -> list[dict]:
    """加载 MATH-500 数据集。返回 [{"question", "gold_answer", "level", "type"}, ...]"""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir)
    examples = []
    for row in ds:
        # gold answer 在 solution 字段的 \boxed{} 中
        gold = extract_boxed(row.get("answer", "") or row.get("solution", ""))
        examples.append({
            "question": row["problem"],
            "gold_answer": gold or row.get("answer", ""),
            "level": row.get("level", ""),
            "type": row.get("type", ""),
        })
    return examples


def build_prompt(question: str) -> str:
    """构造 MATH-500 评测 prompt。"""
    return (
        f"Solve the following math problem. Show your work step by step. "
        f"Put your final answer in \\boxed{{}}.\n\n"
        f"Problem: {question}\n\n"
        f"Solution:"
    )


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
