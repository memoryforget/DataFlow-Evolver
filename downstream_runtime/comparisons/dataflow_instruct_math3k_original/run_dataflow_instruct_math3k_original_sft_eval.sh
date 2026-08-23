#!/usr/bin/env bash
set -euo pipefail

REPO=/jizhicfs/xiaqifeng/DataFlow-Evolver
COMPARISON_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_NAME=${DF_RUN_NAME:-dataflow_instruct_math3k_original_seed42}
RUN_DIR="$REPO/workspace/runs/$RUN_NAME"
RECORD_DIR="$REPO/artifacts/experiment_runs/$RUN_NAME"
SOURCE_RUN="$REPO/workspace/runs/dataflow_instruct_math3k_qwen38_skip_qgen_seed42"
PYTHON=/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python
INPUT="$SOURCE_RUN/input/dataflow_instruct_math3k.jsonl"
OUTPUT_DATASET="$RUN_DIR/dataflow_instruct_math3k_original.jsonl"
FULL_BENCHMARK_DATA="$REPO/data/benchmarks/qwen25_math_eval_data"
EXPECTED_INPUT_SHA=0c40ee2ec56cc2071f2e6dcc7d2503f77a790d40b68fc44ecd7163f1086ae814

if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 2
fi
if [[ -e "$RECORD_DIR" ]] && [[ -n "$(find "$RECORD_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty record directory: $RECORD_DIR" >&2
  exit 2
fi

actual_input_sha=$(sha256sum "$INPUT" | awk '{print $1}')
if [[ "$actual_input_sha" != "$EXPECTED_INPUT_SHA" ]]; then
  echo "SHA256 mismatch for $INPUT: $actual_input_sha" >&2
  exit 3
fi

cd "$REPO"
mkdir -p "$RUN_DIR" "$RECORD_DIR"
START_EPOCH=$(date +%s)
START_ISO=$(date --iso-8601=seconds)

finish_record() {
  local exit_code=$?
  local end_epoch
  end_epoch=$(date +%s)
  printf '{"start_iso":"%s","end_iso":"%s","start_epoch":%s,"end_epoch":%s,"wall_time_seconds":%s,"exit_code":%s}\n' \
    "$START_ISO" "$(date --iso-8601=seconds)" "$START_EPOCH" "$end_epoch" \
    "$((end_epoch - START_EPOCH))" "$exit_code" > "$RECORD_DIR/wall_clock.json"
  nvidia-smi > "$RECORD_DIR/nvidia_smi_end.txt" 2>&1 || true
}
trap finish_record EXIT

printf 'run_name=%s\nstart_iso=%s\nhostname=%s\nrepo=%s\ngit_commit=%s\ngit_branch=%s\ninput_dataset=%s\nfull_benchmark_data=%s\n' \
  "$RUN_NAME" "$START_ISO" "$(hostname)" "$REPO" "$(git rev-parse HEAD)" \
  "$(git branch --show-current)" "$INPUT" "$(readlink -f "$FULL_BENCHMARK_DATA")" \
  > "$RECORD_DIR/launch_metadata.txt"
git status --short --branch > "$RECORD_DIR/git_status.txt"
nvidia-smi > "$RECORD_DIR/nvidia_smi_start.txt"

"$PYTHON" "$COMPARISON_DIR/build_original_math3k.py" \
  --input "$INPUT" \
  --output "$OUTPUT_DATASET" \
  --manifest "$RUN_DIR/manifest.json" \
  --summary "$RUN_DIR/dataset_summary.json"

/usr/bin/time -v -o "$RECORD_DIR/downstream_resource_usage.txt" \
  "$PYTHON" "$REPO/downstream_runtime/run_final_best_full_benchmark.py" \
  --run-dir "$RUN_DIR" \
  --config "$REPO/configs/downstream_math_runtime.local.yaml" \
  --full-data-dir "$FULL_BENCHMARK_DATA" \
  --bad-cases-per-benchmark 2

cp "$RUN_DIR/dataset_summary.json" "$RECORD_DIR/dataset_summary.json"
cp "$RUN_DIR/full_benchmark_eval/final_best/result.json" "$RECORD_DIR/final_result.json"
