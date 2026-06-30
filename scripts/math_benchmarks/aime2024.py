"""AIME 2024 benchmark 加载与判分。

数据源：HuggingFace `AI-MO/aimo-validation-aime` 或本地 JSON。
答案格式：整数 (0-999)。
"""
from __future__ import annotations

from .answer_extract import extract_answer, normalize_number, is_equiv


def load_aime2024(cache_dir: str | None = None) -> list[dict]:
    """加载 AIME 2024 数据集。返回 [{"question", "gold_answer"}, ...]"""
    from datasets import load_dataset

    try:
        ds = load_dataset(
            "AI-MO/aimo-validation-aime", split="train", cache_dir=cache_dir
        )
    except Exception:
        ds = load_dataset(
            "Maxwell-Jia/AIME_2024", split="train", cache_dir=cache_dir
        )
    examples = []
    for row in ds:
        question = row.get("problem", row.get("question", ""))
        gold = str(row.get("answer", ""))
        examples.append({"question": question, "gold_answer": gold})
    return examples


def build_prompt(question: str) -> str:
    """构造 AIME 评测 prompt。"""
    return (
        f"Solve the following AIME competition problem. "
        f"The answer is an integer between 0 and 999 inclusive. "
        f"Show your work step by step, then state your final answer as a single integer.\n\n"
        f"Problem: {question}\n\n"
        f"Solution:"
    )


def score_predictions(
    predictions: list[str], examples: list[dict]
) -> tuple[float, list[dict]]:
    """判分。AIME 答案为整数，精确匹配。返回 (accuracy, bad_cases)。"""
    correct = 0
    bad_cases: list[dict] = []
    for pred_text, ex in zip(predictions, examples):
        pred_answer = extract_answer(pred_text, mode="auto")
        gold = ex["gold_answer"]
        # AIME 答案是整数，做数值比较
        match = False
        if pred_answer is not None:
            pred_num = normalize_number(pred_answer)
            gold_num = normalize_number(gold)
            if pred_num is not None and gold_num is not None:
                match = abs(pred_num - gold_num) < 0.5  # 整数容差
            else:
                match = is_equiv(pred_answer, gold)
        if match:
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
