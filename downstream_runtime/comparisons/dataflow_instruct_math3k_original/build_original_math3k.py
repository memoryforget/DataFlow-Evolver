from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and snapshot the original DataFlow-Instruct Math-3K control."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=3_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    rows = read_jsonl(args.input)
    if len(rows) != args.expected_rows:
        raise RuntimeError(f"expected {args.expected_rows} rows, got {len(rows)}")

    source_counts: Counter[str] = Counter()
    identities: set[tuple[str, str]] = set()
    source_lines: list[int] = []
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        instruction = str(row.get("instruction") or "").strip()
        output = str(row.get("output") or "").strip()
        subset = str(row.get("source_subset") or "")
        source_id = str(row.get("source_id") or "")
        if not instruction or not output:
            raise RuntimeError(f"row {index} has an empty instruction or output")
        if subset not in {"math_train", "gsm8k_train"} or not source_id:
            raise RuntimeError(f"row {index} has invalid source provenance")
        identity = (subset, source_id)
        if identity in identities:
            raise RuntimeError(f"duplicate source identity: {identity}")
        identities.add(identity)
        try:
            source_line = int(row["source_line_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"row {index} has invalid source_line_number") from exc
        source_lines.append(source_line)
        source_counts[subset] += 1
        normalized = dict(row)
        normalized["instruction"] = instruction
        normalized["output"] = output
        normalized_rows.append(normalized)

    expected_counts = {"gsm8k_train": 1_500, "math_train": 1_500}
    if dict(sorted(source_counts.items())) != expected_counts:
        raise RuntimeError(f"unexpected source counts: {dict(source_counts)}")
    if len(source_lines) != len(set(source_lines)):
        raise RuntimeError("duplicate DataFlow-Instruct source line numbers")
    if source_lines != sorted(source_lines):
        raise RuntimeError("Math-3K rows do not preserve DataFlow-Instruct source order")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in normalized_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    output_record = {
        "path": str(args.output.resolve()),
        "rows": len(normalized_rows),
        "sha256": sha256(args.output),
        "source_counts": dict(sorted(source_counts.items())),
        "unique_source_ids": len(identities),
        "unique_source_line_numbers": len(set(source_lines)),
        "original_output_boxed_count": sum(
            "\\boxed{" in str(row["output"]) for row in normalized_rows
        ),
        "preserves_source_order": True,
    }
    builder_path = Path(__file__).resolve()
    manifest = {
        "run_name": args.output.parent.name,
        "mode": "dataflow_instruct_math3k_original_control",
        "completed_iterations": 1,
        "max_iterations": 1,
        "selection_rule": "all exact original MATH/GSM8K matches; no CoT rewrite",
        "provenance_kind": "direct_original_cot",
        "input": {
            "path": str(args.input.resolve()),
            "rows": len(rows),
            "sha256": sha256(args.input),
        },
        "best": {
            "iteration": 0,
            "iteration_dir": str(args.output.parent.resolve()),
            "dataset_path": str(args.output.resolve()),
            "pipeline_path": str(builder_path),
            "accepted_as_incumbent": True,
            "review": None,
        },
    }
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dataset_built",
        "mode": manifest["mode"],
        "input": manifest["input"],
        "output": output_record,
    }
    write_json(args.manifest, manifest)
    write_json(args.summary, summary)
    print(json.dumps(output_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
