"""MATH-500 benchmark loader and scorer."""

from __future__ import annotations

from .additional_benchmarks import load_benchmark_examples
from .answer_extract import extract_answer, is_equiv


def load_math500(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="MATH500",
        default_local_paths=["math/test.jsonl"],
    )


def build_prompt(question: str) -> str:
    return question


def score_predictions(
    predictions: list[str], examples: list[dict]
) -> tuple[float, list[dict]]:
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
    return correct / len(examples) if examples else 0.0, bad_cases
