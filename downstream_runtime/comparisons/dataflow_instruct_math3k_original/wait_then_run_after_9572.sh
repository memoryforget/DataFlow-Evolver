#!/usr/bin/env bash
set -euo pipefail

REPO=/jizhicfs/xiaqifeng/DataFlow-Evolver
PYTHON=/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python
UPSTREAM_SESSION=dataflow_instruct_9572_sft_eval
UPSTREAM_STATUS="$REPO/workspace/runs/dataflow_instruct_9572_qwen38_math_rewrite_seed42/full_benchmark_eval/final_best/status.json"
RUN_SCRIPT="$REPO/downstream_runtime/comparisons/dataflow_instruct_math3k_original/run_dataflow_instruct_math3k_original_sft_eval.sh"
POLL_SECONDS=${DF_DEPENDENCY_POLL_SECONDS:-60}

if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DF_DEPENDENCY_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

read_state() {
  if [[ ! -f "$UPSTREAM_STATUS" ]]; then
    printf 'missing|missing\n'
    return
  fi
  "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])); print("{}|{}".format(value.get("status", "missing"), value.get("stage", "missing")))' \
    "$UPSTREAM_STATUS" 2>/dev/null || printf 'invalid|invalid\n'
}

while true; do
  state=$(read_state)
  case "$state" in
    ok\|completed)
      break
      ;;
    failed\|*|error\|*)
      echo "Upstream 9,572-row experiment failed: $state" >&2
      exit 4
      ;;
  esac
  if ! tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; then
    echo "Upstream tmux exited without a successful terminal status: $state" >&2
    exit 5
  fi
  printf '%s waiting for upstream; state=%s\n' "$(date --iso-8601=seconds)" "$state"
  sleep "$POLL_SECONDS"
done

while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do
  sleep 5
done

printf '%s upstream completed; starting original Math-3K control\n' "$(date --iso-8601=seconds)"
exec bash "$RUN_SCRIPT"
