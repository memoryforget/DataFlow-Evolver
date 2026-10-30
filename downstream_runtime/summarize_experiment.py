#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
    "billable_tokens_estimate",
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def token_block(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    raw = value.get("tokens")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for key in TOKEN_KEYS:
        try:
            result[key] = int(raw.get(key) or 0)
        except (TypeError, ValueError):
            result[key] = 0
    return result


def add_tokens(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    namespaces = ("agent", "pipeline_llm", "review_llm", "embedding")
    totals: dict[str, dict[str, int]] = {name: {} for name in namespaces}
    iterations: list[dict[str, Any]] = []

    for iteration_dir in sorted(run_dir.glob("iteration_*")):
        combined = read_json(iteration_dir / "token_usage.json") or {}
        row: dict[str, Any] = {
            "iteration": iteration_dir.name,
            "token_usage_path": str((iteration_dir / "token_usage.json").resolve()),
            "namespaces": {},
        }
        for namespace in namespaces:
            tokens = token_block(combined.get(namespace))
            row["namespaces"][namespace] = tokens
            add_tokens(totals[namespace], tokens)
        iterations.append(row)

    manifest = read_json(run_dir / "manifest.json")
    downstream = (manifest or {}).get("downstream_evaluation")
    summary = {
        "run_dir": str(run_dir),
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "iterations": iterations,
        "token_totals_by_namespace": totals,
        "framework_wall_time": {
            "data_preparation_wall_time_seconds": (manifest or {}).get(
                "data_preparation_wall_time_seconds"
            ),
            "downstream_evaluation_wall_time_seconds": (manifest or {}).get(
                "downstream_evaluation_wall_time_seconds"
            ),
            "total_wall_time_seconds": (manifest or {}).get(
                "total_wall_time_seconds"
            ),
        },
        "downstream_evaluation": downstream,
        "manifest_available": manifest is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
