# PTB GSM8K Full Experiment Launch Runbook

This runbook records the validated launch flow for the DataFlow-Evolver GSM8K
experiment on host `30100`. Do not commit real API keys.

## Current status (2026-08-04)

- Project compilation succeeds.
- Full `df-agent` test suite succeeds: `318 passed`.
- `df-train` sees the A100 with PyTorch `2.5.1+cu121`.
- `df-eval` sees the A100 with PyTorch `2.10.0+cu128`.
- Direct access to `hf-mirror.com` times out.
- Access through Mihomo at `127.0.0.1:7890` succeeds for the mirror API,
  a small model file, and `huggingface_hub` in `df-agent`.
- Launch is currently blocked because the supplied DataFlow API credential
  returns HTTP 403 `insufficient_user_quota`. Revalidate both API paths before
  starting the full run.

## 1. Start or verify Mihomo

Use a dedicated tmux window:

```bash
cd /vepfs-mlp2/c20250602/500050/lh/xqf/tools/mihomo
./mihomo -d . -f ./config.yaml
```

From another shell, verify it:

```bash
ss -ltnp | grep ':7890 '
curl -L --connect-timeout 10 --max-time 30 \
  -o /dev/null -w 'HTTP %{http_code}\n' \
  -x http://127.0.0.1:7890 \
  'https://hf-mirror.com/api/models?limit=1'
```

Expected result: port `7890` is listening and the request returns HTTP 200.

## 2. Enter the project and clear stale overrides

```bash
cd /vepfs-mlp2/c20250602/500050/lh/xqf/DataFlow-Evolver

unset PTB_INPUT_DIR PTB_INPUT_ENTRY PTB_TARGET_SAMPLES PTB_DECONTAM_REF
unset SFT_TEMPLATE SFT_LEARNING_RATE SFT_WARMUP_STEPS SFT_DEEPSPEED
```

Do not export `PTB_TARGET_SAMPLES`; `configs/default.yaml` currently defaults to
10,000 samples.

## 3. Configure API and Claude SDK access

Fill the keys only in the experiment shell or through a secret manager. Do not
write their values into this file or commit them.

```bash
export PATH=/vepfs-mlp2/c20250602/500050/lh/xqf/.npm-global/bin:/vepfs-mlp2/c20250602/500050/lh/xqf/.nvm/versions/node/v20.20.2/bin:$PATH

export ANTHROPIC_API_KEY='<required>'
export DF_API_KEY='<required>'
export DF_API_URL=http://123.129.219.111:3000/v1
export ANTHROPIC_BASE_URL=http://123.129.219.111:3000/
export IS_SANDBOX=1
```

## 4. Configure proxy and Hugging Face access

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export all_proxy="$ALL_PROXY"

# Keep the local proxy and internal API gateway off the proxy path.
export NO_PROXY=127.0.0.1,localhost,123.129.219.111
export no_proxy="$NO_PROXY"

export HF_ENDPOINT=https://hf-mirror.com
export PTB_HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/vepfs-mlp2/c20250602/500050/lh/xqf/DataFlow-Evolver/hf_cache
```

`HF_TOKEN` is optional for public datasets.

## 5. Configure PTB, GPU, SFT, and evaluation

```bash
export PTB_BASE_MODEL=/vepfs-mlp2/c20250602/500050/models/Qwen3-1.7B-Base
export PTB_TASK_DIR=/vepfs-mlp2/c20250602/500050/lh/xqf/PostTrainBench/src/eval/tasks/gsm8k
export PTB_BENCHMARK=gsm8k
export PTB_DEV_LIMIT=150
export PTB_MAX_ITERATIONS=4

export CUDA_VISIBLE_DEVICES=0
export SFT_CUDA_VISIBLE_DEVICES=0
export EVAL_CUDA_VISIBLE_DEVICES=0
export SFT_NUM_GPUS=1
export SFT_BATCH_SIZE=1
export SFT_GRAD_ACCUM=8
export SFT_FLASH_ATTN=sdpa
export SFT_NUM_EPOCHS=1
export SFT_WARMUP_RATIO=0.03
export SFT_TRAIN_TIMEOUT_SEC=16200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 6. Mandatory preflight

```bash
# Proxy and mirror
curl -L --connect-timeout 10 --max-time 30 \
  -o /dev/null -w 'HF mirror: HTTP %{http_code}\n' \
  'https://hf-mirror.com/api/models?limit=1'

# Paths
test -d "$PTB_BASE_MODEL"
test -d "$PTB_TASK_DIR"
test -x /vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python
test -x /vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-train/bin/python
test -x /vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-eval/bin/python

# GPU
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

# Code
/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python \
  -m compileall -q dataflow_evolver scripts tests
/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python \
  -m pytest -q
```

Also make one minimal request through the DataFlow API and one through Claude
CLI. Do not start the full experiment if either request fails, times out, or
reports insufficient quota.

## 7. Workspace policy

The current workspace contains `_agent_root`, `data_links`, and a previous start
script, but no checkpoint was present during validation. Do not run recursive
cleanup automatically. If a clean run is explicitly desired, first inspect and
archive any artifacts that need to be retained, then remove only the confirmed
targets.

The configured checkpoint is `workspace/checkpoint.json`. If it exists at
launch, the orchestrator may resume it instead of starting a new search.

## 8. Start the experiment in tmux

Create a new session name if `ptb_full_0804` already exists:

```bash
tmux new-session -s ptb_full_0804_retry
```

Inside tmux, export the variables above, then run:

```bash
TASK='为 PostTrainBench 官方 GSM8K 评测生成高质量数学推理 SFT 数据，用于 post-train `Qwen3-1.7B-Base`，目标是在官方 `inspect_evals/gsm8k` 评测中取得更高 accuracy。
每条训练样本必须符合本框架下游训练契约：字段为 `instruction` 和 `output`。`instruction` 是一条 grade-school math word problem；`output` 必须包含清晰、逐步的数学推理过程，并在最后单独一行严格写成 `ANSWER: <数值>`，不要使用 `\boxed{}`。该格式需要对齐 PTB 官方 GSM8K prompt：模型会被要求 step by step 解题，最后一行用 `ANSWER: $ANSWER` 给出数值答案，评分使用 numeric match。
允许使用公开外部数据作为补充，以提升题型覆盖、难度分布、推理质量和答案格式一致性；但严禁使用 GSM8K test split、其镜像、衍生测试集或任何包含 benchmark test questions/answers 的数据，避免 data contamination。'

/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python \
  -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "$TASK" \
  --log-level INFO \
  2>&1 | tee workspace/run_ptb_full_0804_official_task.log
```

## 9. Monitor

```bash
tmux attach -t ptb_full_0804_retry
tail -f workspace/run_ptb_full_0804_official_task.log
nvidia-smi
```

Watch for API quota/authentication failures, repeated infrastructure retries,
Hugging Face/ModelScope timeouts, missing dataset fields, CUDA OOM, and bridge
timeouts. The orchestrator is configured to checkpoint after each completed
round.
