#!/usr/bin/env bash
# 本地 PostTrainBench 一键运行器（取代官方 apptainer/HTCondor/solve.sh）。
#
# 用法：
#   scripts/run_posttrainbench_local.sh <benchmark> <base_model_path> [ptb_task_dir]
# 例：
#   scripts/run_posttrainbench_local.sh gsm8k /workspace/xiaqifeng/models/Qwen3-1.7B
#
# 行为：
#   1. 由 <benchmark> 查 scripts/ptb_benchmarks.py 得到派生 --task 文案与 dev-limit；
#   2. export PTB_*（base model / 任务目录 / benchmark / dev-limit / 时间预算）；
#   3. 用 df-agent 环境启动 python -m dataflow_evolver.main（configs/posttrainbench_gsm8k.yaml，
#      env 参数化，可复用所有 benchmark）；
#   4. main.py 内部完成 Tier-1 搜索 → Tier-2 top-K 选模 → final_model/ → 官方 evaluate.py --limit -1 终评，
#      并在末尾打印 dev/终评分数（本脚本无需再单独跑终评）。
#
# 可用环境变量覆盖（均有默认值）：
#   PTB_ROOT            PostTrainBench 根目录（默认 /workspace/xiaqifeng/PostTrainBench）
#   PTB_TIME_BUDGET_SEC 总墙钟预算秒（默认 28800≈8h；快速试可设小，如 600）
#   DF_AGENT_PYTHON     df-agent 环境 python
#   PTB_TRAIN_PYTHON / PTB_EVAL_PYTHON  df-train / df-eval 环境 python
#   DF_API_KEY / DF_API_URL             辅助 Agent / 算子 LLM 的密钥与端点
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法：$0 <benchmark> <base_model_path> [ptb_task_dir]" >&2
  echo "benchmark 可选：aime2025 arenahardwriting bfcl gpqamain gsm8k healthbench humaneval" >&2
  exit 1
fi

BENCHMARK="$1"
BASE_MODEL="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PTB_ROOT="${PTB_ROOT:-/workspace/xiaqifeng/PostTrainBench}"
PTB_TASK_DIR="${3:-$PTB_ROOT/src/eval/tasks/$BENCHMARK}"
DF_AGENT_PYTHON="${DF_AGENT_PYTHON:-/workspace/xiaqifeng/miniconda3/envs/df-agent/bin/python}"
export PTB_TRAIN_PYTHON="${PTB_TRAIN_PYTHON:-/workspace/xiaqifeng/miniconda3/envs/df-train/bin/python}"
export PTB_EVAL_PYTHON="${PTB_EVAL_PYTHON:-/workspace/xiaqifeng/miniconda3/envs/df-eval/bin/python}"
export PTB_TIME_BUDGET_SEC="${PTB_TIME_BUDGET_SEC:-28800}"

# 基本校验
[[ -x "$DF_AGENT_PYTHON" ]] || { echo "df-agent python 不可执行：$DF_AGENT_PYTHON" >&2; exit 1; }
[[ -e "$BASE_MODEL" ]]      || { echo "base model 路径不存在：$BASE_MODEL" >&2; exit 1; }
[[ -d "$PTB_TASK_DIR" ]]    || { echo "PostTrainBench 任务目录不存在：$PTB_TASK_DIR" >&2; exit 1; }
[[ -f "$PTB_TASK_DIR/evaluate.py" ]] || { echo "任务目录缺少 evaluate.py：$PTB_TASK_DIR" >&2; exit 1; }

# 从映射表取派生 --task 文案（含 format_requirement 结构约束）与 dev-limit（分次单独取，避免空格被切碎）
TASK_TEXT="$(cd "$PROJECT_ROOT" && "$DF_AGENT_PYTHON" -c "import sys; sys.path.insert(0,'scripts'); import ptb_benchmarks as p; print(p.composed_task_text('$BENCHMARK'))")" \
  || { echo "查表失败（未知 benchmark？）：$BENCHMARK" >&2; exit 1; }
DEV_LIMIT="$(cd "$PROJECT_ROOT" && "$DF_AGENT_PYTHON" -c "import sys; sys.path.insert(0,'scripts'); import ptb_benchmarks as p; print(p.get_benchmark('$BENCHMARK')['dev_limit'])")"
FORMAT_REQ="$(cd "$PROJECT_ROOT" && "$DF_AGENT_PYTHON" -c "import sys; sys.path.insert(0,'scripts'); import ptb_benchmarks as p; print(p.format_requirement_for('$BENCHMARK'))")"

export PTB_BENCHMARK="$BENCHMARK"
export PTB_BASE_MODEL="$BASE_MODEL"
export PTB_TASK_DIR="$PTB_TASK_DIR"
export PTB_DEV_LIMIT="$DEV_LIMIT"

cd "$PROJECT_ROOT"
echo "=================================================================="
echo "PostTrainBench 本地运行"
echo "  benchmark      : $PTB_BENCHMARK"
echo "  base model     : $PTB_BASE_MODEL"
echo "  task dir       : $PTB_TASK_DIR"
echo "  dev limit      : $PTB_DEV_LIMIT"
echo "  format req     : $FORMAT_REQ"
echo "  time budget(s) : $PTB_TIME_BUDGET_SEC"
echo "  --task         : $TASK_TEXT"
echo "=================================================================="

exec "$DF_AGENT_PYTHON" -m dataflow_evolver.main \
  --config configs/posttrainbench_gsm8k.yaml \
  --task "$TASK_TEXT"
