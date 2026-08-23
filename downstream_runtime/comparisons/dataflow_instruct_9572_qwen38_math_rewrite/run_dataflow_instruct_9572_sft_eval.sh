#!/usr/bin/env bash
set -euo pipefail

REPO=/jizhicfs/xiaqifeng/DataFlow-Evolver
COMPARISON_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_NAME=${DF_RUN_NAME:-dataflow_instruct_9572_qwen38_math_rewrite_seed42}
RUN_DIR="$REPO/workspace/runs/$RUN_NAME"
RECORD_DIR="$REPO/artifacts/experiment_runs/$RUN_NAME"
SOURCE_RUN="$REPO/workspace/runs/dataflow_instruct_math3k_qwen38_skip_qgen_seed42"
PYTHON=/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python
INSTRUCT="$REPO/data/dataflow-instruct-10k/train.jsonl"
SELECTED_MATH="$SOURCE_RUN/input/dataflow_instruct_math3k.jsonl"
REWRITTEN_MATH="$SOURCE_RUN/final_sft_math3k_qwen38.jsonl"
PIPELINE="$REPO/downstream_runtime/comparisons/dataflow_instruct_math3k_qwen38_skip_qgen/reasoning_math_pipeline_skip_qgen.py"
FULL_BENCHMARK_DATA="$REPO/data/benchmarks/qwen25_math_eval_data"
OUTPUT_DATASET="$RUN_DIR/dataflow_instruct_9572.jsonl"
EXPECTED_INSTRUCT_SHA=76ea3f877b52b8af900adaf51e468ec5dfc473b7f34d53dd4d595cc5f1baa616
EXPECTED_SELECTED_SHA=0c40ee2ec56cc2071f2e6dcc7d2503f77a790d40b68fc44ecd7163f1086ae814
EXPECTED_REWRITTEN_SHA=89efa9195f168284c9d2274341ba98c5feafd66b0117f14601b529c72b37820d
EXPECTED_PIPELINE_SHA=f0fcf3467a928770731082f9d90da8cc8b4e9e45748dcba87fe13bbd6e626673

if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 2
fi
if [[ -e "$RECORD_DIR" ]] && [[ -n "$(find "$RECORD_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty record directory: $RECORD_DIR" >&2
  exit 2
fi

check_sha() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch for $path: $actual != $expected" >&2
    exit 3
  fi
}

check_sha "$INSTRUCT" "$EXPECTED_INSTRUCT_SHA"
check_sha "$SELECTED_MATH" "$EXPECTED_SELECTED_SHA"
check_sha "$REWRITTEN_MATH" "$EXPECTED_REWRITTEN_SHA"
check_sha "$PIPELINE" "$EXPECTED_PIPELINE_SHA"

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

cat > "$RECORD_DIR/launch_metadata.txt" <<EOF
run_name=$RUN_NAME
start_iso=$START_ISO
hostname=$(hostname)
repo=$REPO
git_commit=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
input_dataset=$INSTRUCT
selected_math=$SELECTED_MATH
rewritten_math=$REWRITTEN_MATH
pipeline=$PIPELINE
full_benchmark_data=$(readlink -f "$FULL_BENCHMARK_DATA")
EOF
git status --short --branch > "$RECORD_DIR/git_status.txt"
nvidia-smi > "$RECORD_DIR/nvidia_smi_start.txt"

"$PYTHON" "$COMPARISON_DIR/build_dataflow_instruct_9572.py" \
  --instruct "$INSTRUCT" \
  --selected-math "$SELECTED_MATH" \
  --rewritten-math "$REWRITTEN_MATH" \
  --pipeline "$PIPELINE" \
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
