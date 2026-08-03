"""AMC 2023 benchmark loader and scorer."""

from __future__ import annotations

from .additional_benchmarks import load_benchmark_examples


def load_amc23(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="AMC23",
        default_local_paths=["amc/test.jsonl"],
    )


def build_prompt(question: str) -> str:
    return question
