#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path("/jizhicfs/xiaqifeng/DataFlow-Evolver")
sys.path.insert(0, str(REPO / "scripts/downstream"))

from math_runtime import evaluate_model, load_yaml, train_sft, write_json


EXPECTED_BENCHMARKS = [
    "gsm8k",
    "math",
    "amc23",
    "aime24",
    "aime25",
    "minerva_math",
    "gaokao2024_mix",
    "olympiadbench",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DataFlow-Evolver best dataset and evaluate all full math benchmarks."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--full-data-dir", required=True, type=Path)
    parser.add_argument("--bad-cases-per-benchmark", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def resolve_best(run_dir: Path) -> tuple[dict[str, Any], Path, Path, int]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evolver manifest does not exist: {manifest_path}")
    manifest = load_json_object(manifest_path)
    best = manifest.get("best")
    if not isinstance(best, dict):
        raise ValueError("manifest.best must be a JSON object")

    dataset_path = Path(str(best.get("dataset_path", ""))).expanduser().resolve()
    pipeline_path = Path(str(best.get("pipeline_path", ""))).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"best dataset does not exist: {dataset_path}")
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"best pipeline does not exist: {pipeline_path}")
    dataset_path.relative_to(run_dir)
    # Comparison runs keep their immutable pipeline implementation in the
    # tracked comparison directory instead of copying it into every run.
    # Existence is sufficient because the pipeline is provenance-only here.

    best_iteration = int(best.get("iteration", -1)) + 1
    if best_iteration < 1:
        raise ValueError("manifest.best.iteration must be a zero-based nonnegative integer")
    return manifest, dataset_path, pipeline_path, best_iteration


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    full_data_dir = args.full_data_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"runtime config does not exist: {config_path}")
    if not full_data_dir.is_dir():
        raise FileNotFoundError(f"full benchmark directory does not exist: {full_data_dir}")
    if args.bad_cases_per_benchmark < 1:
        raise ValueError("--bad-cases-per-benchmark must be at least 1")

    manifest, dataset_path, pipeline_path, best_iteration = resolve_best(run_dir)
    output_dir = run_dir / "full_benchmark_eval" / "final_best"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty final output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / "runtime"

    config = load_yaml(config_path)
    config["config_path"] = str(config_path)
    for group in config["evaluation"]["groups"]:
        group["data_dir"] = str(full_data_dir)

    request = {
        "protocol_version": 3,
        "kind": "final_best_full_benchmark",
        "checkpoint_iteration": int(manifest.get("completed_iterations", best_iteration)),
        "incumbent_iteration": best_iteration,
        "dataset_path": str(dataset_path),
        "pipeline_path": str(pipeline_path),
        "training_performed": True,
        "expected_benchmarks": EXPECTED_BENCHMARKS,
        "bad_cases_per_benchmark": args.bad_cases_per_benchmark,
    }
    plan = {
        "schema_version": 1,
        "kind": "final_best_full_benchmark",
        "source_config": str(config_path),
        "full_data_dir": str(full_data_dir),
        "request": request,
        "evaluation": config["evaluation"],
    }
    write_json(output_dir / "request.json", request)
    write_json(output_dir / "evaluation_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    started = time.perf_counter()
    try:
        write_json(output_dir / "status.json", {"status": "running", "stage": "sft"})
        model_path, training_artifacts = train_sft(request, config, runtime_dir)
        write_json(
            output_dir / "status.json",
            {"status": "running", "stage": "full_benchmark", "model": str(model_path)},
        )
        benchmarks, evaluation_artifacts = evaluate_model(
            model_path.resolve(),
            EXPECTED_BENCHMARKS,
            args.bad_cases_per_benchmark,
            config,
            runtime_dir,
        )
        result = {
            "status": "ok",
            "kind": "final_best_full_benchmark",
            "model": str(model_path.resolve()),
            "best_iteration": best_iteration,
            "dataset_path": str(dataset_path),
            "benchmarks": benchmarks,
            "summary": (
                f"trained manifest best iteration {best_iteration} and evaluated "
                f"{len(benchmarks)} full benchmarks; "
                f"wall_time_seconds={time.perf_counter() - started:.3f}"
            ),
            "artifacts": {**training_artifacts, **evaluation_artifacts},
        }
        write_json(output_dir / "result.json", result)
        write_json(
            output_dir / "status.json",
            {
                "status": "ok",
                "stage": "completed",
                "result_path": str((output_dir / "result.json").resolve()),
            },
        )
    except Exception as exc:
        failure = {
            "status": "failed",
            "kind": "final_best_full_benchmark",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "wall_time_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "failure.json", failure)
        write_json(output_dir / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
