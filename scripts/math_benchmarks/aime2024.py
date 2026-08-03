"""AIME 2024 benchmark loader and scorer."""

from __future__ import annotations

from .additional_benchmarks import load_benchmark_examples


def load_aime2024(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="AIME2024",
        default_local_paths=["aime/test.jsonl"],
    )


def build_prompt(question: str) -> str:
    return question
