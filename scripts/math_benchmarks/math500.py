"""MATH-500 benchmark loader and scorer."""

from __future__ import annotations

from .additional_benchmarks import load_benchmark_examples


def load_math500(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="MATH500",
        default_local_paths=["math/test.jsonl"],
    )


def build_prompt(question: str) -> str:
    return question
