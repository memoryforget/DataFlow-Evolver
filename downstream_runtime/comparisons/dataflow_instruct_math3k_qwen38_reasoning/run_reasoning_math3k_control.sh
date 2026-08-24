#!/usr/bin/env bash
set -euo pipefail

REPO=/jizhicfs/xiaqifeng/DataFlow-Evolver
COMPARISON_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_COMPARISON="$REPO/downstream_runtime/comparisons/dataflow_instruct_math3k_qwen38_skip_qgen"
RUN_NAME=${DF_RUN_NAME:-dataflow_instruct_math3k_qwen38_reasoning_seed42}
RUN_DIR="$REPO/workspace/runs/$RUN_NAME"
RECORD_DIR="$REPO/artifacts/experiment_runs/$RUN_NAME"
PYTHON=/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python
PIPELINE="$SOURCE_COMPARISON/reasoning_math_pipeline_skip_qgen.py"
EXPECTED_PIPELINE_SHA=f0fcf3467a928770731082f9d90da8cc8b4e9e45748dcba87fe13bbd6e626673
INSTRUCT="$REPO/data/dataflow-instruct-10k/train.jsonl"
MATH_TRAIN=/jizhicfs/xiaqifeng/Qwen2.5-Math/evaluation/data/math/train.jsonl
GSM8K_TRAIN=/jizhicfs/xiaqifeng/Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl
FULL_BENCHMARK_DATA="$REPO/data/benchmarks/qwen25_math_eval_data"
OUTPUT_DATASET="$RUN_DIR/final_sft_math3k_qwen38_reasoning.jsonl"

if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 2
fi
if [[ -e "$RECORD_DIR" ]] && [[ -n "$(find "$RECORD_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty record directory: $RECORD_DIR" >&2
  exit 2
fi

actual_pipeline_sha=$(sha256sum "$PIPELINE" | awk '{print $1}')
if [[ "$actual_pipeline_sha" != "$EXPECTED_PIPELINE_SHA" ]]; then
  echo "Pipeline SHA256 mismatch: $actual_pipeline_sha" >&2
  exit 3
fi

cd "$REPO"
mkdir -p "$RUN_DIR/input" "$RUN_DIR/cache" "$RECORD_DIR"
START_EPOCH=$(date +%s)
START_ISO=$(date --iso-8601=seconds)

write_stage() {
  printf '{"stage":"%s","updated_at":"%s"}\n' "$1" "$(date --iso-8601=seconds)" > "$RUN_DIR/status.json"
}

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

printf 'run_name=%s\nstart_iso=%s\nhostname=%s\nrepo=%s\ngit_commit=%s\ngit_branch=%s\npipeline=%s\npipeline_sha256=%s\npipeline_model=%s\nreasoning_adapter=%s\n' \
  "$RUN_NAME" "$START_ISO" "$(hostname)" "$REPO" "$(git rev-parse HEAD)" \
  "$(git branch --show-current)" "$PIPELINE" "$actual_pipeline_sha" \
  "${DF_PIPELINE_MODEL:-Qwen3.8-27B}" "$COMPARISON_DIR/run_pipeline_with_reasoning.py" \
  > "$RECORD_DIR/launch_metadata.txt"
git status --short --branch > "$RECORD_DIR/git_status.txt"
nvidia-smi > "$RECORD_DIR/nvidia_smi_start.txt"

write_stage extract_math3k
"$PYTHON" "$SOURCE_COMPARISON/extract_dataflow_instruct_math.py" \
  --instruct "$INSTRUCT" \
  --math-train "$MATH_TRAIN" \
  --gsm8k-train "$GSM8K_TRAIN" \
  --output "$RUN_DIR/input/dataflow_instruct_math3k.jsonl" \
  --metadata "$RUN_DIR/input/extraction_manifest.json"

export no_proxy="${no_proxy:+${no_proxy},}28.59.7.41,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
export DF_MATH_BASELINE_INPUT="$RUN_DIR/input/dataflow_instruct_math3k.jsonl"
export DF_MATH_BASELINE_CACHE_DIR="$RUN_DIR/cache"
export DF_PIPELINE_API_URL=${DF_PIPELINE_API_URL:-http://28.59.7.41:8000/v1/chat/completions}
export DF_PIPELINE_MODEL=${DF_PIPELINE_MODEL:-Qwen3.8-27B}
export DF_PIPELINE_API_KEY=${DF_PIPELINE_API_KEY:-EMPTY}
export DF_API_KEY=${DF_API_KEY:-$DF_PIPELINE_API_KEY}
export DF_PIPELINE_MAX_WORKERS=${DF_PIPELINE_MAX_WORKERS:-100}
export DF_PIPELINE_MAX_RETRIES=${DF_PIPELINE_MAX_RETRIES:-5}
curl -fsS --connect-timeout 10 --max-time 60 \
  http://28.59.7.41:8000/v1/models > "$RUN_DIR/teacher_models.json"
"$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); ids={str(x.get("id")) for x in d.get("data",[])}; assert sys.argv[2] in ids' \
  "$RUN_DIR/teacher_models.json" "$DF_PIPELINE_MODEL"

write_stage dataflow_qwen38_reasoning_rewrite
/usr/bin/time -v -o "$RECORD_DIR/pipeline_resource_usage.txt" \
  "$PYTHON" "$COMPARISON_DIR/run_pipeline_with_reasoning.py" \
  "$PIPELINE" "$RUN_DIR/reasoning_serving_stats.json" \
  > "$RUN_DIR/pipeline.log" 2>&1

write_stage finalize_and_validate_reasoning
"$PYTHON" "$SOURCE_COMPARISON/finalize_rewritten_math_dataset.py" \
  --run-dir "$RUN_DIR" \
  --pipeline "$PIPELINE" \
  --output "$OUTPUT_DATASET" \
  --min-rows "${DF_MIN_FINAL_ROWS:-1000}"
"$PYTHON" "$COMPARISON_DIR/validate_reasoning_dataset.py" \
  --dataset "$OUTPUT_DATASET" \
  --serving-stats "$RUN_DIR/reasoning_serving_stats.json" \
  --manifest "$RUN_DIR/manifest.json" \
  --pipeline-summary "$RUN_DIR/summary.json" \
  --report "$RUN_DIR/reasoning_contract_report.json" \
  --min-rows "${DF_MIN_FINAL_ROWS:-1000}"

write_stage sft_and_full_benchmark
/usr/bin/time -v -o "$RECORD_DIR/downstream_resource_usage.txt" \
  "$PYTHON" "$REPO/downstream_runtime/run_final_best_full_benchmark.py" \
  --run-dir "$RUN_DIR" \
  --config "$REPO/configs/downstream_math_runtime.local.yaml" \
  --full-data-dir "$FULL_BENCHMARK_DATA" \
  --bad-cases-per-benchmark 2

write_stage completed
cp "$RUN_DIR/summary.json" "$RECORD_DIR/pipeline_summary.json"
cp "$RUN_DIR/reasoning_contract_report.json" "$RECORD_DIR/reasoning_contract_report.json"
cp "$RUN_DIR/full_benchmark_eval/final_best/result.json" "$RECORD_DIR/final_result.json"
