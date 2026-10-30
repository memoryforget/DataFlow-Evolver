#!/usr/bin/env bash
set -euo pipefail
cd /jizhicfs/xiaqifeng/DataFlow-Evolver
set -a
source /jizhicfs/xiaqifeng/vpn.sh
set +a
# Relay traffic uses the corporate proxy; the local teacher endpoint must stay direct.
export no_proxy="${no_proxy:+${no_proxy},}28.59.7.41"
export NO_PROXY="$no_proxy"
export HOME=${HOME:-/jizhicfs/xiaqifeng}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/jizhicfs/xiaqifeng/.cache}
export HF_HOME=${HF_HOME:-/jizhicfs/xiaqifeng/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/jizhicfs/xiaqifeng/.cache/huggingface/datasets}
: "${TREE_API_KEY:?export TREE_API_KEY first}"
export DF_API_URL=${DF_API_URL:-https://relay01.01tree.ai/codex/v1/chat/completions}
export DF_API_KEY=${DF_API_KEY:-$TREE_API_KEY}
export DF_AGENT_BASE_URL=${DF_AGENT_BASE_URL:-https://relay01.01tree.ai/codex/v1}
export DF_AGENT_API_KEY=${DF_AGENT_API_KEY:-$TREE_API_KEY}
export AGENT_BACKEND=${AGENT_BACKEND:-codex}
export CODEX_BIN=${CODEX_BIN:-/jizhicfs/xiaqifeng/bin/codex}
export CODEX_MODEL=${CODEX_MODEL:-gpt-5.4}
export CODEX_EFFORT=${CODEX_EFFORT:-xhigh}
export DF_AGENT_RUN_TIMEOUT_SEC=${DF_AGENT_RUN_TIMEOUT_SEC:-3600}
export CODEX_MODEL_PROVIDER=${CODEX_MODEL_PROVIDER:-tree-relay}
export CODEX_BASE_URL=${CODEX_BASE_URL:-https://relay01.01tree.ai/codex/v1}
export CODEX_API_KEY_ENV=${CODEX_API_KEY_ENV:-TREE_API_KEY}
export DF_REVIEW_MODEL=${DF_REVIEW_MODEL:-gpt-5.4}
export DF_PIPELINE_API_URL=${DF_PIPELINE_API_URL:-http://28.59.7.41:8000/v1/chat/completions}
export DF_PIPELINE_MODEL=${DF_PIPELINE_MODEL:-Qwen3.8-27B}
export DF_PIPELINE_API_KEY=${DF_PIPELINE_API_KEY:-EMPTY}
export DF_PIPELINE_MAX_WORKERS=${DF_PIPELINE_MAX_WORKERS:-100}
export DF_DOWNSTREAM_BASELINE_ENABLED=${DF_DOWNSTREAM_BASELINE_ENABLED:-true}
export DF_AGENT_PYTHON=/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python
TASK=$(cat downstream_runtime/task_math_mix_v3.txt)
exec "$DF_AGENT_PYTHON" -m dataflow_evolver.main \
  --config downstream_runtime/dataflow_math_periodic.local.yaml \
  --task "$TASK" \
  --log-level INFO
