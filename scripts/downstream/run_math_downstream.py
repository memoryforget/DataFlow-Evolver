from __future__ import annotations

import argparse
import time
from pathlib import Path

from math_runtime import evaluate_model, load_request, load_yaml, train_sft, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External Qwen2.5 SFT + math diagnostic runner for DataFlow-Evolver."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    request = load_request(args.request.resolve())
    config = load_yaml(args.config.resolve())
    config["config_path"] = str(args.config.resolve())
    runtime_dir = args.result.resolve().parent / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    if request["stage"] == "baseline":
        model_path = Path(str(config["base_model"])).resolve()
        artifacts["checkpoint"] = str(model_path)
    else:
        model_path, training_artifacts = train_sft(request, config, runtime_dir)
        artifacts.update(training_artifacts)

    benchmarks, evaluation_artifacts = evaluate_model(
        model_path,
        request["expected_benchmarks"],
        request["bad_cases_per_benchmark"],
        config,
        runtime_dir,
    )
    artifacts.update(evaluation_artifacts)
    result = {
        "model": str(model_path),
        "benchmarks": benchmarks,
        "summary": (
            f"stage={request['stage']}; evaluated {len(benchmarks)} fixed diagnostic benchmarks; "
            f"training_performed={request['training_performed']}; "
            f"wall_time_seconds={time.perf_counter() - started:.3f}"
        ),
        "artifacts": artifacts,
    }
    write_json(args.result.resolve(), result)


if __name__ == "__main__":
    main()
