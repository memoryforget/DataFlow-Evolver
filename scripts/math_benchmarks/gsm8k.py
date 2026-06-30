"""GSM8K benchmark 加载与判分。

数据源：HuggingFace `openai/gsm8k` test split (1319 条)。
答案格式：推理过程后跟 #### <number>。
"""
from __future__ import annotations

from pathlib import Path

from .answer_extract import extract_answer, is_equiv


def load_gsm8k(split: str = "test", cache_dir: str | None = None) -> list[dict]:
    """加载 GSM8K 指定 split。返回 [{"question", "gold_answer"}, ...]

    split: "train" (~7473 条) 或 "test" (1319 条)
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    examples = []
    for row in ds:
        answer_text = row["answer"]
        # 提取 #### 后的数值作为 gold
        gold = None
        if "####" in answer_text:
            gold = answer_text.split("####")[-1].strip()
        examples.append({"question": row["question"], "gold_answer": gold or ""})
    return examples


def build_prompt(question: str) -> str:
    """构造 GSM8K 评测 prompt（Qwen2.5-Math 格式：user message 仅含原题）。

    system prompt 需在推理时单独设置为：
    "Please reason step by step, and put your final answer within \\boxed{}"
    """
    return question


def score_predictions(
    predictions: list[str], examples: list[dict]
) -> tuple[float, list[dict]]:
    """判分。返回 (accuracy, bad_cases)。"""
    correct = 0
    bad_cases: list[dict] = []
    for pred_text, ex in zip(predictions, examples):
        pred_answer = extract_answer(pred_text, mode="auto")
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
