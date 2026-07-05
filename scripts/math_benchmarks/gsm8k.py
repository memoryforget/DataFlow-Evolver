"""GSM8K benchmark loader and scorer."""

from __future__ import annotations

from .additional_benchmarks import load_benchmark_rows
from .answer_extract import extract_answer, is_equiv


def load_gsm8k(split: str = "test", cache_dir: str | None = None) -> list[dict]:
    rows = load_benchmark_rows(
        bench_name="GSM8K",
        default_local_paths=[f"gsm8k/{split}.jsonl"],
    )
    examples = []
    for row in rows:
        if "gold_answer" in row:
            gold = str(row.get("gold_answer", "")).strip()
        else:
            answer_text = str(row.get("answer", ""))
            gold = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
        question = row.get("question", row.get("problem", ""))
        examples.append({"question": question, "gold_answer": gold or ""})
    return examples


def build_prompt(question: str) -> str:
    return question


def score_predictions(
    predictions: list[str], examples: list[dict]
) -> tuple[float, list[dict]]:
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
    return correct / len(examples) if examples else 0.0, bad_cases
