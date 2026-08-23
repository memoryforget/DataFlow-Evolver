#!/usr/bin/env bash
set -euo pipefail

REPO=/jizhicfs/xiaqifeng/DataFlow-Evolver
RUN_NAME=${DF_RUN_NAME:-math_mix_v3_periodic_seed42}
FINAL_ENABLED=${DF_FINAL_FULL_BENCHMARK_ENABLED:-true}

"$REPO/downstream_runtime/run_math_periodic.sh"

if [[ "$FINAL_ENABLED" == "true" ]]; then
  PYTHON=${DF_AGENT_PYTHON:-/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python}
  "$PYTHON" "$REPO/downstream_runtime/run_final_best_full_benchmark.py" \
    --run-dir "$REPO/workspace/runs/$RUN_NAME" \
    --config "$REPO/configs/downstream_math_runtime.local.yaml" \
    --full-data-dir "${DF_FINAL_FULL_BENCHMARK_DATA_DIR:-$REPO/data/benchmarks/qwen25_math_eval_data}" \
    --bad-cases-per-benchmark "${DF_FINAL_BAD_CASES_PER_BENCHMARK:-2}"
fi
