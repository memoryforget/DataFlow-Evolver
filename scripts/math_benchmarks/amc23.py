"""AMC 2023 benchmark loader and scorer."""

from __future__ import annotations

from .additional_benchmarks import load_benchmark_examples
from .answer_extract import extract_answer, is_equiv, normalize_number


def load_amc23(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="AMC23",
        default_local_paths=["amc/test.jsonl"],
    )


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
        match = False
        if pred_answer is not None:
            pred_num = normalize_number(pred_answer)
            gold_num = normalize_number(gold)
            if pred_num is not None and gold_num is not None:
                match = abs(pred_num - gold_num) < 0.5
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
    return correct / len(examples) if examples else 0.0, bad_cases
