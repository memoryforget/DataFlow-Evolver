#!/usr/bin/env bash
set -euo pipefail

REPO=/jizhicfs/xiaqifeng/DataFlow-Evolver
RUN_NAME=${DF_RUN_NAME:-math_mix_v3_periodic_seed42}
RUN_DIR="$REPO/workspace/runs/$RUN_NAME"
RECORD_DIR="$REPO/artifacts/experiment_runs/$RUN_NAME"
mkdir -p "$RECORD_DIR"

if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 2
fi

START_EPOCH=$(date +%s)
START_ISO=$(date --iso-8601=seconds)
export DF_RUN_NAME="$RUN_NAME"

{
  echo "run_name=$RUN_NAME"
  echo "start_iso=$START_ISO"
  echo "start_epoch=$START_EPOCH"
  echo "hostname=$(hostname)"
  echo "repo=$REPO"
  echo "git_commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "git_branch=$(git -C "$REPO" branch --show-current)"
  echo "python=/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python"
  echo "max_iterations=${DF_MAX_ITERATIONS:-5}"
  echo "downstream_interval=${DF_DOWNSTREAM_EVAL_INTERVAL:-2}"
  echo "downstream_baseline=${DF_DOWNSTREAM_BASELINE_ENABLED:-true}"
  echo "pipeline_workers=${DF_PIPELINE_MAX_WORKERS:-100}"
  echo "pipeline_execution_timeout_sec=${DF_EXECUTION_TIMEOUT_SEC:-43200}"
  echo "agent_run_timeout_sec=${DF_AGENT_RUN_TIMEOUT_SEC:-3600}"
  echo "codex_model=${CODEX_MODEL:-gpt-5.4}"
  echo "codex_effort=${CODEX_EFFORT:-xhigh}"
  echo "review_model=${DF_REVIEW_MODEL:-gpt-5.4}"
  echo "pipeline_model=${DF_PIPELINE_MODEL:-Qwen3.8-27B}"
  echo "final_full_benchmark=${DF_FINAL_FULL_BENCHMARK_ENABLED:-true}"
  echo "final_full_benchmark_data=${DF_FINAL_FULL_BENCHMARK_DATA_DIR:-/jizhicfs/xiaqifeng/DataFlow-Evolver/data/benchmarks/qwen25_math_eval_data}"
  echo "baseline_result_path=${DF_DOWNSTREAM_BASELINE_RESULT_PATH:-}"
} > "$RECORD_DIR/launch_metadata.txt"

git -C "$REPO" status --short --branch > "$RECORD_DIR/git_status.txt"
git -C "$REPO" diff --binary > "$RECORD_DIR/source_changes.patch"
cp "$REPO/downstream_runtime/dataflow_math_periodic.local.yaml" \
  "$RECORD_DIR/dataflow_math_periodic.local.yaml"
cp "$REPO/configs/downstream_math_runtime.local.yaml" \
  "$RECORD_DIR/downstream_math_runtime.local.yaml"
cp "$REPO/downstream_runtime/task_math_mix_v3.txt" \
  "$RECORD_DIR/task_math_mix_v3.txt"
cp "$REPO/downstream_runtime/run_math_periodic_with_final.sh" \
  "$RECORD_DIR/run_math_periodic_with_final.sh"
cp "$REPO/downstream_runtime/run_final_best_full_benchmark.py" \
  "$RECORD_DIR/run_final_best_full_benchmark.py"
nvidia-smi > "$RECORD_DIR/nvidia_smi_start.txt"

set +e
/usr/bin/time -v -o "$RECORD_DIR/resource_usage.txt" \
  "$REPO/downstream_runtime/run_math_periodic_with_final.sh" \
  > >(tee -a "$RECORD_DIR/main.log") \
  2> >(tee -a "$RECORD_DIR/main.log" >&2)
EXIT_CODE=$?
set -e

END_EPOCH=$(date +%s)
END_ISO=$(date --iso-8601=seconds)
WALL_SECONDS=$((END_EPOCH - START_EPOCH))

cat > "$RECORD_DIR/wall_clock.json" <<EOF
{
  "start_iso": "$START_ISO",
  "end_iso": "$END_ISO",
  "start_epoch": $START_EPOCH,
  "end_epoch": $END_EPOCH,
  "wall_time_seconds": $WALL_SECONDS,
  "exit_code": $EXIT_CODE
}
EOF

nvidia-smi > "$RECORD_DIR/nvidia_smi_end.txt" || true
/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python \
  "$REPO/downstream_runtime/summarize_experiment.py" \
  --run-dir "$RUN_DIR" \
  --output "$RECORD_DIR/token_and_framework_summary.json" \
  > "$RECORD_DIR/token_summary_stdout.log" 2>&1 || true

exit "$EXIT_CODE"
