"""GSM8K benchmark loader and scorer."""

from __future__ import annotations

from .additional_benchmarks import load_benchmark_rows


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
