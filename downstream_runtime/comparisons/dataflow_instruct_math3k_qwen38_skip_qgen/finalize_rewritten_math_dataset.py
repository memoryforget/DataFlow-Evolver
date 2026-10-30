from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-rows", type=int, default=1000)
    return parser.parse_args()


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def step_number(path: Path) -> int:
    match = re.search(r"step(\d+)\.jsonl$", path.name)
    return int(match.group(1)) if match else -1


def main() -> None:
    args = parse_args()
    cache_files = list((args.run_dir / "cache").glob("*.jsonl"))
    if not cache_files:
        raise RuntimeError("DataFlow pipeline produced no JSONL cache")
    source_path = max(cache_files, key=lambda path: (step_number(path), path.stat().st_mtime))

    rows: list[dict[str, Any]] = []
    seen_instructions: set[str] = set()
    source_counts: Counter[str] = Counter()
    with source_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = str(row.get("instruction") or "").strip()
            generated_cot = str(row.get("generated_cot") or "").strip()
            if not instruction or not generated_cot:
                continue
            if instruction in seen_instructions:
                raise RuntimeError("pipeline produced duplicate instructions")
            seen_instructions.add(instruction)
            normalized = dict(row)
            normalized["instruction"] = instruction
            normalized["output"] = generated_cot
            rows.append(normalized)
            source_counts[str(row.get("source_subset") or "unknown")] += 1

    if len(rows) < args.min_rows:
        raise RuntimeError(
            f"pipeline retained only {len(rows)} rows; minimum required is {args.min_rows}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    manifest = {
        "run_name": args.run_dir.name,
        "mode": "dataflow_instruct_math3k_qwen38_skip_qgen_control",
        "completed_iterations": 1,
        "max_iterations": 1,
        "selection_rule": "all unique final DataFlow pipeline survivors with non-empty generated_cot",
        "best": {
            "iteration": 0,
            "iteration_dir": str(args.run_dir.resolve()),
            "dataset_path": str(args.output.resolve()),
            "pipeline_path": str(args.pipeline.resolve()),
            "accepted_as_incumbent": True,
            "review": None,
        },
    }
    (args.run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "pipeline_completed",
        "official_pipeline_cache": str(source_path.resolve()),
        "cache_row_counts": {
            path.name: count_rows(path)
            for path in sorted(cache_files, key=step_number)
        },
        "final_dataset": str(args.output.resolve()),
        "final_rows": len(rows),
        "final_sha256": sha256(args.output),
        "source_counts": dict(sorted(source_counts.items())),
        "boxed_count": sum("\\boxed{" in str(row["output"]) for row in rows),
        "non_boxed_count": sum("\\boxed{" not in str(row["output"]) for row in rows),
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
