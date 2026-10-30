from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the selected Math-3K portion of DataFlow-Instruct-10K."
    )
    parser.add_argument("--instruct", required=True, type=Path)
    parser.add_argument("--selected-math", required=True, type=Path)
    parser.add_argument("--rewritten-math", required=True, type=Path)
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--expected-instruct", type=int, default=10_000)
    parser.add_argument("--expected-selected", type=int, default=3_000)
    parser.add_argument("--expected-rewritten", type=int, default=2_572)
    parser.add_argument("--expected-untouched", type=int, default=7_000)
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


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


def conversation_value(row: dict[str, Any], role: str) -> str:
    values = [
        str(turn.get("value") or "").strip()
        for turn in row.get("conversations", [])
        if turn.get("from") == role
    ]
    if len(values) != 1 or not values[0]:
        raise ValueError(f"expected exactly one non-empty {role!r} turn")
    return values[0]


def index_by_source_line(
    rows: list[dict[str, Any]], label: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            line_number = int(row["source_line_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} row lacks a valid source_line_number") from exc
        if line_number < 1:
            raise ValueError(f"{label} has invalid source line {line_number}")
        if line_number in indexed:
            raise ValueError(f"{label} duplicates source line {line_number}")
        indexed[line_number] = row
    return indexed


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    for path in (args.instruct, args.selected_math, args.rewritten_math, args.pipeline):
        if not path.is_file():
            raise FileNotFoundError(path)

    instruct_rows = read_jsonl(args.instruct)
    selected_rows = read_jsonl(args.selected_math)
    rewritten_rows = read_jsonl(args.rewritten_math)
    if len(instruct_rows) != args.expected_instruct:
        raise RuntimeError(
            f"expected {args.expected_instruct} instruct rows, got {len(instruct_rows)}"
        )
    if len(selected_rows) != args.expected_selected:
        raise RuntimeError(
            f"expected {args.expected_selected} selected rows, got {len(selected_rows)}"
        )
    if len(rewritten_rows) != args.expected_rewritten:
        raise RuntimeError(
            f"expected {args.expected_rewritten} rewritten rows, got {len(rewritten_rows)}"
        )

    selected_by_line = index_by_source_line(selected_rows, "selected Math-3K")
    rewritten_by_line = index_by_source_line(rewritten_rows, "rewritten Math")
    unexpected_rewrites = set(rewritten_by_line) - set(selected_by_line)
    if unexpected_rewrites:
        raise RuntimeError(
            f"rewritten rows are outside Math-3K: {sorted(unexpected_rewrites)[:10]}"
        )

    for line_number, selected in selected_by_line.items():
        if line_number > len(instruct_rows):
            raise RuntimeError(f"selected source line is out of range: {line_number}")
        source_instruction = conversation_value(instruct_rows[line_number - 1], "human")
        if normalize(source_instruction) != normalize(str(selected.get("instruction") or "")):
            raise RuntimeError(f"selected instruction mismatch at source line {line_number}")
    for line_number, rewritten in rewritten_by_line.items():
        selected = selected_by_line[line_number]
        if normalize(str(rewritten.get("instruction") or "")) != normalize(
            str(selected.get("instruction") or "")
        ):
            raise RuntimeError(f"rewritten instruction mismatch at source line {line_number}")
        if not str(rewritten.get("output") or "").strip():
            raise RuntimeError(f"rewritten output is empty at source line {line_number}")

    output_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for line_number, source_row in enumerate(instruct_rows, start=1):
        if line_number in selected_by_line:
            rewritten = rewritten_by_line.get(line_number)
            if rewritten is None:
                counts["rejected_math"] += 1
                continue
            normalized_row = dict(rewritten)
            normalized_row["instruction"] = str(rewritten["instruction"]).strip()
            normalized_row["output"] = str(rewritten["output"]).strip()
            output_rows.append(normalized_row)
            counts["rewritten_math"] += 1
            continue

        output_rows.append(
            {
                "instruction": conversation_value(source_row, "human"),
                "output": conversation_value(source_row, "gpt"),
                "source": "dataflow_instruct_10k_non_math",
                "source_subset": "unmatched_non_math",
                "source_line_number": line_number,
            }
        )
        counts["untouched_non_math"] += 1

    expected_counts = {
        "rejected_math": args.expected_selected - args.expected_rewritten,
        "rewritten_math": args.expected_rewritten,
        "untouched_non_math": args.expected_untouched,
    }
    if dict(sorted(counts.items())) != expected_counts:
        raise RuntimeError(f"unexpected composition: {dict(counts)} != {expected_counts}")
    expected_total = args.expected_rewritten + args.expected_untouched
    if len(output_rows) != expected_total:
        raise RuntimeError(f"expected {expected_total} output rows, got {len(output_rows)}")
    output_line_numbers = [int(row["source_line_number"]) for row in output_rows]
    if output_line_numbers != sorted(output_line_numbers):
        raise RuntimeError("output does not preserve DataFlow-Instruct-10K source order")
    if len(output_line_numbers) != len(set(output_line_numbers)):
        raise RuntimeError("output contains duplicate source lines")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    input_records = {
        "dataflow_instruct_10k": {
            "path": str(args.instruct.resolve()),
            "rows": len(instruct_rows),
            "sha256": sha256(args.instruct),
        },
        "selected_math3k": {
            "path": str(args.selected_math.resolve()),
            "rows": len(selected_rows),
            "sha256": sha256(args.selected_math),
        },
        "rewritten_math2572": {
            "path": str(args.rewritten_math.resolve()),
            "rows": len(rewritten_rows),
            "sha256": sha256(args.rewritten_math),
        },
        "pipeline": {
            "path": str(args.pipeline.resolve()),
            "sha256": sha256(args.pipeline),
        },
    }
    output_record = {
        "path": str(args.output.resolve()),
        "rows": len(output_rows),
        "sha256": sha256(args.output),
        "composition": dict(sorted(counts.items())),
        "unique_source_line_numbers": len(set(output_line_numbers)),
        "preserves_source_order": True,
    }
    manifest = {
        "run_name": args.output.parent.name,
        "mode": "dataflow_instruct_9572_qwen38_math_rewrite",
        "completed_iterations": 1,
        "max_iterations": 1,
        "selection_rule": "replace selected Math-3K rows with all 2,572 rewrite survivors",
        "composition": output_record["composition"],
        "inputs": input_records,
        "best": {
            "iteration": 0,
            "iteration_dir": str(args.output.parent.resolve()),
            "dataset_path": str(args.output.resolve()),
            "pipeline_path": str(args.pipeline.resolve()),
            "accepted_as_incumbent": True,
            "review": None,
        },
    }
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dataset_built",
        "inputs": input_records,
        "output": output_record,
    }
    write_json(args.manifest, manifest)
    write_json(args.summary, summary)
    print(json.dumps(output_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
