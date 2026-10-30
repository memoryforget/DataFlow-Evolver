#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <conda-root> <llamafactory-root> <source-vllm-env>" >&2
  exit 2
fi

CONDA_ROOT=$1
LLAMAFACTORY_ROOT=$2
SOURCE_VLLM_ENV=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DFE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SFT_ENV="$CONDA_ROOT/envs/dfe-sft"
EVAL_ENV="$CONDA_ROOT/envs/dfe-math-eval"

test -x "$CONDA_ROOT/bin/conda"
test -f "$LLAMAFACTORY_ROOT/pyproject.toml"
test -f "$DFE_ROOT/scripts/downstream/vendor/qwen25_math/latex2sympy/setup.py"
test -x "$SOURCE_VLLM_ENV/bin/python"

if [[ ! -x "$SFT_ENV/bin/python" ]]; then
  "$CONDA_ROOT/bin/conda" create --yes --clone "$SOURCE_VLLM_ENV" --prefix "$SFT_ENV"
fi
"$SFT_ENV/bin/python" -m pip install --upgrade pip
"$SFT_ENV/bin/python" -m pip install "$LLAMAFACTORY_ROOT"
"$SFT_ENV/bin/python" -m pip install deepspeed

if [[ ! -x "$EVAL_ENV/bin/python" ]]; then
  "$CONDA_ROOT/bin/conda" create --yes --clone "$SOURCE_VLLM_ENV" --prefix "$EVAL_ENV"
fi
"$EVAL_ENV/bin/python" -m pip install --upgrade pip
"$EVAL_ENV/bin/python" -m pip install \
  antlr4-python3-runtime==4.11.1 \
  Pebble \
  python_dateutil \
  timeout-decorator \
  word2number
"$EVAL_ENV/bin/python" -m pip install --force-reinstall --no-deps \
  "$DFE_ROOT/scripts/downstream/vendor/qwen25_math/latex2sympy"

"$SFT_ENV/bin/python" -c "import deepspeed, llamafactory, torch, transformers; print('sft environment ok')"
"$EVAL_ENV/bin/python" -c "import pebble, vllm, torch, transformers; print('eval environment ok')"
