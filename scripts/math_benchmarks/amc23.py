"""AMC 2023 benchmark 加载与判分。

数据源：HuggingFace `AI-MO/aimo-validation-amc` 或本地 JSON。
答案格式：整数或字母。
"""
from __future__ import annotations

from .answer_extract import extract_answer, normalize_number, is_equiv


def load_amc23(cache_dir: str | None = None) -> list[dict]:
    """加载 AMC 2023 数据集。返回 [{"question", "gold_answer"}, ...]"""
    from datasets import load_dataset

    try:
        ds = load_dataset(
            "AI-MO/aimo-validation-amc", split="train", cache_dir=cache_dir
        )
    except Exception:
        # fallback: 返回空列表（需手动配置数据集）
        return []
    examples = []
    for row in ds:
        question = row.get("problem", row.get("question", ""))
        gold = str(row.get("answer", ""))
        examples.append({"question": question, "gold_answer": gold})
    return examples


def build_prompt(question: str) -> str:
    """构造 AMC 评测 prompt。"""
    return question


def score_predictions(
    predictions: list[str], examples: list[dict]
) -> tuple[float, list[dict]]:
    """判分。AMC 答案可能是整数或字母。返回 (accuracy, bad_cases)。"""
    correct = 0
    bad_cases: list[dict] = []
    for pred_text, ex in zip(predictions, examples):
        pred_answer = extract_answer(pred_text, mode="auto")
        gold = ex["gold_answer"]
        match = False
        if pred_answer is not None:
            # 尝试数值比较
            pred_num = normalize_number(pred_answer)
            gold_num = normalize_number(gold)
            if pred_num is not None and gold_num is not None:
                match = abs(pred_num - gold_num) < 0.5
            else:
                # 字符串等价比较
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
