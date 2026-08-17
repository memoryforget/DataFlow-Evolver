from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize fixed 10% math diagnostic sets for downstream feedback."
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_parquet(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        raise FileNotFoundError("no parquet files matched")
    return pd.concat(
        [pd.read_parquet(path) for path in paths], ignore_index=True
    ).to_dict("records")


def stable_sample(
    records: list[dict[str, Any]], fraction: float, seed: int, name: str
) -> tuple[list[dict[str, Any]], list[int]]:
    count = max(1, math.floor(len(records) * fraction + 0.5))
    salt = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed ^ salt)
    indices = sorted(rng.sample(range(len(records)), count))
    return [records[index] for index in indices], indices


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_root = args.benchmark_root.resolve()
    qwen_data = benchmark_root / "qwen25_math_eval_data"

    gsm8k_source = benchmark_root / "gsm8k/main/test-00000-of-00001.parquet"
    math_sources = sorted((benchmark_root / "MATH").glob("*/test/*.parquet"))
    amc23_source = benchmark_root / "AMC23/default/test/0000.parquet"
    aime24_source = qwen_data / "aime24/test.jsonl"
    aime25_source = benchmark_root / "AIME-2025/data/aime-2025.parquet"
    minerva_source = qwen_data / "minerva_math/test.jsonl"
    gaokao_source = qwen_data / "gaokao2024_mix/test.jsonl"
    olympiad_source = qwen_data / "olympiadbench/test.jsonl"

    loaders: dict[
        str, tuple[list[Path], Callable[[], list[dict[str, Any]]], Path, str, str]
    ] = {
        "gsm8k": (
            [gsm8k_source],
            lambda: read_parquet([gsm8k_source]),
            output_dir / "shared_data/gsm8k/test.jsonl",
            "gsm8k",
            "benchmark_data parquet",
        ),
        "math": (
            math_sources,
            lambda: [
                {
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "level": row.get("level", ""),
                    "subject": row.get("type", ""),
                    "unique_id": f"diagnostic/math/{index}",
                }
                for index, row in enumerate(read_parquet(math_sources))
            ],
            output_dir / "shared_data/math/test.jsonl",
            "math",
            "benchmark_data category parquets",
        ),
        "amc23": (
            [amc23_source],
            lambda: [
                {**row, "problem": row["question"], "answer": str(row["answer"])}
                for row in read_parquet([amc23_source])
            ],
            output_dir / "shared_data/amc23/test.jsonl",
            "amc23",
            "benchmark_data parquet",
        ),
        "aime24": (
            [aime24_source],
            lambda: read_jsonl(aime24_source),
            output_dir / "shared_data/aime24/test.jsonl",
            "aime24",
            "Qwen2.5-Math official 30-problem evaluation file",
        ),
        "aime25": (
            [aime25_source],
            lambda: [
                {
                    "id": row["id"],
                    "problem": row["problem"],
                    "answer": str(row["answer"]),
                }
                for row in read_parquet([aime25_source])
            ],
            output_dir / "shared_data/aime25/test.jsonl",
            "aime25",
            "benchmark_data parquet; vendored evaluator uses the AIME answer parser",
        ),
        "minerva_math": (
            [minerva_source],
            lambda: read_jsonl(minerva_source),
            output_dir / "shared_data/minerva_math/test.jsonl",
            "minerva_math",
            "Qwen2.5-Math evaluator-ready file",
        ),
        "gaokao2024_mix": (
            [gaokao_source],
            lambda: read_jsonl(gaokao_source),
            output_dir / "shared_data/gaokao2024_mix/test.jsonl",
            "gaokao2024_mix",
            "Qwen2.5-Math evaluator-ready file",
        ),
        "olympiadbench": (
            [olympiad_source],
            lambda: read_jsonl(olympiad_source),
            output_dir / "shared_data/olympiadbench/test.jsonl",
            "olympiadbench",
            "Qwen2.5-Math evaluator-ready English math subset",
        ),
    }

    manifest: dict[str, Any] = {
        "version": 1,
        "seed": args.seed,
        "fraction": args.fraction,
        "policy": "one fixed sample per benchmark, materialized before evolution",
        "benchmarks": {},
        "notes": [
            "The benchmark_data/AIME-2024 parquet contains 960 training-style rows, not the official 30 AIME 2024 questions; the Qwen2.5-Math 30-question file is used instead.",
            "AIME 2025 is evaluated under its own benchmark name by the vendored evaluator, using the AIME answer parser.",
        ],
    }
    for name, (
        sources,
        loader,
        destination,
        parser_name,
        source_note,
    ) in loaders.items():
        records = loader()
        sampled, indices = stable_sample(records, args.fraction, args.seed, name)
        write_jsonl(destination, sampled)
        manifest["benchmarks"][name] = {
            "source_paths": [str(path.resolve()) for path in sources],
            "source_note": source_note,
            "source_rows": len(records),
            "diagnostic_rows": len(sampled),
            "sampled_indices": indices,
            "qwen_data_dir": str(destination.parent.parent.resolve()),
            "qwen_data_name": parser_name,
            "diagnostic_path": str(destination.resolve()),
            "diagnostic_sha256": sha256(destination),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
