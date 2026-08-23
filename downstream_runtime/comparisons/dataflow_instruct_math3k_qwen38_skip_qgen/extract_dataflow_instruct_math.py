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
        description="Extract the exact MATH+GSM8K 3k subset from DataFlow-Instruct-10K."
    )
    parser.add_argument("--instruct", required=True, type=Path)
    parser.add_argument("--math-train", required=True, type=Path)
    parser.add_argument("--gsm8k-train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--expected-math", type=int, default=1500)
    parser.add_argument("--expected-gsm8k", type=int, default=1500)
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


def unique_index(
    rows: list[dict[str, Any]], field: str, label: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(normalize(str(row[field])), []).append(row)
    duplicates = {key: values for key, values in grouped.items() if len(values) != 1}
    if duplicates:
        raise ValueError(f"{label} has {len(duplicates)} duplicated normalized questions")
    return {key: values[0] for key, values in grouped.items()}


def gsm8k_answer(row: dict[str, Any]) -> str:
    answer = str(row["answer"]).strip()
    if "####" not in answer:
        raise ValueError("GSM8K answer does not contain the expected #### delimiter")
    final = answer.rsplit("####", 1)[1].strip().replace(",", "")
    if not final:
        raise ValueError("empty GSM8K final answer")
    return final


def main() -> None:
    args = parse_args()
    instruct_rows = read_jsonl(args.instruct)
    math_rows = read_jsonl(args.math_train)
    gsm8k_rows = read_jsonl(args.gsm8k_train)

    math_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in math_rows:
        math_grouped.setdefault(normalize(str(row["problem"])), []).append(row)
    gsm8k_index = unique_index(gsm8k_rows, "question", "GSM8K train")

    output_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    source_ids: set[tuple[str, str]] = set()
    unmatched = 0
    for line_number, row in enumerate(instruct_rows, start=1):
        instruction = conversation_value(row, "human")
        original_output = conversation_value(row, "gpt")
        key = normalize(instruction)
        math_candidates = math_grouped.get(key, [])
        gsm8k_row = gsm8k_index.get(key)
        if math_candidates and gsm8k_row is not None:
            raise ValueError(f"row {line_number} matches both MATH and GSM8K")
        if math_candidates:
            if len(math_candidates) != 1:
                raise ValueError(
                    f"row {line_number} ambiguously matches {len(math_candidates)} MATH rows"
                )
            source = math_candidates[0]
            source_name = "math_train"
            source_id = str(source["unique_id"])
            golden_answer = str(source["answer"]).strip()
            extra = {
                "math_level": source.get("level"),
                "math_subject": source.get("subject"),
            }
        elif gsm8k_row is not None:
            source = gsm8k_row
            source_name = "gsm8k_train"
            source_id = str(source["idx"])
            golden_answer = gsm8k_answer(source)
            extra = {}
        else:
            unmatched += 1
            continue

        identity = (source_name, source_id)
        if identity in source_ids:
            raise ValueError(f"duplicate selected source identity: {identity}")
        source_ids.add(identity)
        source_counts[source_name] += 1
        output_rows.append(
            {
                "instruction": instruction,
                "output": original_output,
                "golden_answer": golden_answer,
                "source": "dataflow_instruct_10k_math",
                "source_subset": source_name,
                "source_id": source_id,
                "source_line_number": line_number,
                **extra,
            }
        )

    expected = {
        "gsm8k_train": args.expected_gsm8k,
        "math_train": args.expected_math,
    }
    if dict(sorted(source_counts.items())) != expected:
        raise RuntimeError(f"unexpected domain counts: {dict(source_counts)} != {expected}")
    if len(output_rows) != args.expected_math + args.expected_gsm8k:
        raise AssertionError("unexpected extracted row count")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "exact normalized question match against local MATH and GSM8K train splits",
        "normalization": "Unicode-preserving casefold plus whitespace collapse",
        "inputs": {
            "dataflow_instruct_10k": {
                "path": str(args.instruct.resolve()),
                "rows": len(instruct_rows),
                "sha256": sha256(args.instruct),
            },
            "math_train": {
                "path": str(args.math_train.resolve()),
                "rows": len(math_rows),
                "sha256": sha256(args.math_train),
            },
            "gsm8k_train": {
                "path": str(args.gsm8k_train.resolve()),
                "rows": len(gsm8k_rows),
                "sha256": sha256(args.gsm8k_train),
            },
        },
        "output": {
            "path": str(args.output.resolve()),
            "rows": len(output_rows),
            "sha256": sha256(args.output),
            "source_counts": dict(sorted(source_counts.items())),
            "unmatched_non_math_rows": unmatched,
            "unique_source_ids": len(source_ids),
            "original_output_boxed_count": sum(
                "\\boxed{" in row["output"] for row in output_rows
            ),
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
