"""MATH train 题库加载与判分（dev 评测用）。

数据源：Qwen2.5-Math 提供的 MATH competition train split (7500 条)。
这些是公开训练语料（非 benchmark test set），用作 MCTS dev 奖励信号不会污染终评。
答案格式：LaTeX \\boxed{} 内为最终答案。
"""
from __future__ import annotations

from pathlib import Path

from .answer_extract import extract_answer, extract_boxed, is_equiv

# 默认路径：Qwen2.5-Math 仓库中的 MATH train 文件
_DEFAULT_MATH_TRAIN_PATH = str(
    Path(__file__).resolve().parents[2].parent
    / "Qwen2.5-Math" / "evaluation" / "data" / "math" / "train.jsonl"
)


def load_math_train(
    data_path: str | None = None, cache_dir: str | None = None
) -> list[dict]:
    """加载 MATH train set（本地文件）。返回 [{"question", "gold_answer", "level", "subject"}, ...]"""
    import json

    path = Path(data_path) if data_path else Path(_DEFAULT_MATH_TRAIN_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"MATH train 文件不存在：{path}\n"
            f"请确认 Qwen2.5-Math 仓库位于项目同级目录，或通过 MATH_TRAIN_PATH 环境变量指定路径。"
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
            # MATH train 格式：problem, solution, answer, level, subject, unique_id
            question = row.get("problem", "")
            gold = row.get("answer", "")
            if not gold:
                # 从 solution 中提取 \boxed{}
                gold = extract_boxed(row.get("solution", "")) or ""
            examples.append({
                "question": question,
                "gold_answer": gold,
                "level": row.get("level", ""),
                "subject": row.get("subject", ""),
            })
    return examples


def build_prompt(question: str) -> str:
    """构造 MATH train dev 评测 prompt。"""
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
