from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PATTERN = re.compile(
    r"\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--serving-stats", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pipeline-summary", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--min-rows", type=int, default=1_000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    args = parse_args()
    serving_stats = load_json(args.serving_stats)
    think_lengths: list[int] = []
    answer_lengths: list[int] = []
    malformed: list[int] = []
    missing_boxed: list[int] = []
    rows = 0
    with args.dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            output = str(row.get("output") or "")
            match = PATTERN.fullmatch(output)
            if match is None:
                malformed.append(line_number)
                continue
            think = match.group("think").strip()
            answer = match.group("answer").strip()
            if not think or not answer:
                malformed.append(line_number)
                continue
            if "\\boxed{" not in answer:
                missing_boxed.append(line_number)
            think_lengths.append(len(think))
            answer_lengths.append(len(answer))

    if rows < args.min_rows:
        raise RuntimeError(f"reasoning dataset has only {rows} rows")
    if malformed:
        raise RuntimeError(
            f"{len(malformed)} rows violate the think/answer contract: {malformed[:20]}"
        )
    if missing_boxed:
        raise RuntimeError(
            f"{len(missing_boxed)} answer sections lack boxed answers: {missing_boxed[:20]}"
        )
    if int(serving_stats.get("reasoning_field_promoted", 0)) < rows:
        raise RuntimeError(
            "fewer reasoning-field responses were observed than final retained rows"
        )

    contract = {
        "rows": rows,
        "rows_with_nonempty_think_and_answer": len(think_lengths),
        "rows_with_boxed_answer_section": rows - len(missing_boxed),
        "think_length_chars": {
            "min": min(think_lengths),
            "p50": percentile(think_lengths, 0.50),
            "p95": percentile(think_lengths, 0.95),
            "max": max(think_lengths),
        },
        "answer_length_chars": {
            "min": min(answer_lengths),
            "p50": percentile(answer_lengths, 0.50),
            "p95": percentile(answer_lengths, 0.95),
            "max": max(answer_lengths),
        },
        "serving_stats": serving_stats,
    }
    report = {
        "schema_version": 1,
        "status": "ok",
        "format": "<think>{reasoning}</think>\\n<answer>{content}</answer>",
        "contract": contract,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)

    manifest = load_json(args.manifest)
    manifest["mode"] = "dataflow_instruct_math3k_qwen38_reasoning_control"
    manifest["reasoning_contract"] = contract
    write_json(args.manifest, manifest)

    pipeline_summary = load_json(args.pipeline_summary)
    pipeline_summary["mode"] = manifest["mode"]
    pipeline_summary["reasoning_contract"] = contract
    write_json(args.pipeline_summary, pipeline_summary)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
