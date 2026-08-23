#!/usr/bin/env bash
set -euo pipefail
export HOME=${HOME:-/jizhicfs/xiaqifeng}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export HF_HOME=${HF_HOME:-/jizhicfs/xiaqifeng/.cache/huggingface}
CACHE_ROOT=${DF_EMBEDDING_CACHE_ROOT:-/tmp/dataflow-evolver-embedding}
mkdir -p "$CACHE_ROOT"/{xdg,vllm,triton,torchinductor,cuda}
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
exec /jizhicfs/xiaqifeng/miniconda3/envs/dfe-math-eval/bin/vllm serve \
  /jizhicfs/xiaqifeng/DataFlow-Evolver/model/Qwen3-Embedding-8B \
  --task embed \
  --served-model-name /jizhicfs/xiaqifeng/DataFlow-Evolver/model/Qwen3-Embedding-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90
