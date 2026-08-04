# DataFlow-Evolver

DataFlow-Evolver is an MCTS-based data generation framework for post-training
experiments. The framework is backend-agnostic at the trainer-wrapper boundary;
for the current evaluation profile, `configs/default.yaml` is configured to use
the official PostTrainBench evaluator.

## Architecture

```text
TaskProfiler
  -> Init node
  -> MCTS loop
       -> PipelineAgent via Claude Agent SDK writes pipeline.py
       -> SubprocessExecutionWrapper runs the pipeline
       -> ReviewAgent checks schema, quality, and decontamination
       -> CappedTrainer fixes sample count
       -> posttrainbench_sft_eval.py runs SFT + official PTB evaluate.py --limit N
       -> TrainScore backpropagates through MCTS
```

Final evaluation uses the same PTB bridge in `--mode test`, which calls official
`evaluate.py --limit -1`.

## Installation

The project intentionally separates the orchestration environment from training
and evaluation. Do not merge these environments unless you are debugging a very
small local case.

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment 1: `df-agent`

Runs the orchestrator, API-backed helper agents, the Claude Agent SDK, and the
generated DataFlow pipelines.

```bash
conda create -n df-agent python=3.11 -y
conda activate df-agent

# Orchestrator + Claude Agent SDK.
uv pip install -e ".[agent]"

# df-agent does not train. Prefer CPU torch to avoid pulling a full CUDA stack.
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Runtime dependencies for generated DataFlow pipelines.
uv pip install -e ".[agent-runtime]"

# Optional but recommended for local development tests.
uv pip install -e ".[dev]"

# Required when agent_sdk.external_data: true; installs dataset-search and
# data-discovery dependencies for HuggingFace, ModelScope, Serper/GitHub/arXiv.
uv pip install -e ".[external-data]"
```

`external-data` intentionally does not use `modelscope[datasets]`: that extra
pins `datasets` too low and can break the HF/DataFlow side. The extra lists the
minimal ModelScope runtime dependencies directly (`addict`, `simplejson`,
`sortedcontainers`).

### Environment 2: `df-train`

Runs LlamaFactory SFT through `scripts/posttrainbench_sft_eval.py`.

```bash
conda create -n df-train python=3.11 -y
conda activate df-train
uv pip install -e ".[df-train]" \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### Environment 3: `df-eval`

Runs vLLM and official PTB/inspect_ai evaluation.

```bash
conda create -n df-eval python=3.11 -y
conda activate df-eval
uv pip install -e ".[df-eval]" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

The PTB evaluator also needs `inspect_ai` / `inspect_evals` in `df-eval`. Use the
versions required by your PostTrainBench checkout.

## Cross-Environment Bridge

`configs/default.yaml` points the trainer wrapper at the train and eval Python
executables:

```yaml
trainer:
  train_python: /path/to/miniconda3/envs/df-train/bin/python
  eval_python: /path/to/miniconda3/envs/df-eval/bin/python
```

These are YAML keys, not direct environment variables. If you want environment
overrides, write them in YAML as `${ENV:default}`. Internally, `eval_python` is
forwarded to the bridge as `SFT_EVAL_PYTHON`.

The bridge also fixes the child process environment for common conda issues:

- prepends the target env `bin` to `PATH` so `inspect_ai` can launch `vllm`
- prepends the target env `lib` to `LD_LIBRARY_PATH`
- sets CUDA path when the train env contains `nvcc`

## Claude Agent SDK Runtime

Pipeline generation uses Claude Agent SDK rather than a simple chat completion.
Make sure `claude` and `node` are discoverable in the orchestrator environment.

Required runtime variables:

```bash
export ANTHROPIC_API_KEY=...
export DF_API_KEY=...
export ANTHROPIC_BASE_URL=http://your-anthropic-compatible-endpoint/   # optional
export DF_API_URL=http://your-openai-compatible-endpoint/v1             # optional
export IS_SANDBOX=1                                                     # when required by the local Claude CLI setup
```

The Claude model is configured in `configs/default.yaml` under
`agent_sdk.model`; it is not selected by an environment variable.

## PTB Runtime Configuration

Use `configs/default.yaml`.

Important variables:

```bash
export PTB_BASE_MODEL=/path/to/base_model
export PTB_TASK_DIR=/path/to/PostTrainBench/src/eval/tasks/gsm8k
export PTB_BENCHMARK=gsm8k
export PTB_DEV_LIMIT=150
export PTB_TARGET_SAMPLES=4000
export PTB_MAX_ITERATIONS=5
```

Evaluation data access:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

If `HF_ENDPOINT` is unset, `scripts/posttrainbench_sft_eval.py` sets it to
`https://hf-mirror.com` for the official evaluator. A proxy is optional; it is
only needed when neither local HF cache nor the mirror is reachable.

Useful SFT/eval overrides:

```bash
export SFT_NUM_EPOCHS=1
export SFT_LEARNING_RATE=2.0e-5
export SFT_BATCH_SIZE=1
export SFT_GRAD_ACCUM=8
export SFT_NUM_GPUS=1
export SFT_CUTOFF_LEN=4096
export SFT_FLASH_ATTN=sdpa
export EVAL_CUDA_VISIBLE_DEVICES=0
export EVAL_MAX_CONNECTIONS=2
```

Keep training hyperparameters fixed within one search run; otherwise score
differences no longer isolate data quality.

## No Input Data By Default

PTB tasks do not provide training input data to the agent. `configs/default.yaml`
therefore leaves `project.input_dir` empty, so generation starts from scratch.

Only set the following for explicit ablations:

```bash
export PTB_INPUT_DIR=/path/to/input
export PTB_INPUT_ENTRY=some_file.jsonl
```

Never use benchmark test questions or answers as training data.

## External Data Tools

The framework can mount two in-process MCP tool groups when
`agent_sdk.external_data: true`:

- `dataset-search`: search/inspect/preview/download datasets from Hugging Face
  and ModelScope into `project.data_links_dir`.
- `data-discovery`: search the web, GitHub, and arXiv for dataset leads.

These tools run in the **df-agent** process, so their dependencies must be
installed in `df-agent` via `.[external-data]`. They are lazily imported: missing
optional packages make the corresponding tool return an explanatory error rather
than crashing the whole MCTS loop.

Useful configuration:

```yaml
agent_sdk:
  external_data: true
  discovery: true
project:
  data_links_dir: ""      # empty = <workspace>/data_links
dataset_search:
  endpoint: ${HF_ENDPOINT:https://hf-mirror.com}
  token: ${HF_TOKEN:}
  cache_dir: ${HF_HOME:}
  max_download_mb: 2048
  modelscope_endpoint: ""
  modelscope_token: ${MODELSCOPE_API_TOKEN:}
discovery:
  serper_api_key: ${SERPER_API_KEY:}
  github_token: ${GITHUB_TOKEN:}
```

For PTB runs, configure decontamination before enabling external data. Many Hub
datasets are benchmark mirrors or derivatives.

## Running

```bash
/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python \
  -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "合成 GSM8K 风格的小学数学应用题 SFT 数据（含 CoT、末行 ANSWER: <数值>）以 post-train 该模型" \
  --log-level INFO 2>&1 | tee workspace/run_$(date +%m%d_%H%M).log
```

See [`docs/run-flows.md`](./docs/run-flows.md) for the step-by-step PTB official
default run flow.

## Result Contract

The PTB bridge writes `<output>/results.json`:

```json
{
  "score": 0.1234,
  "metric_name": "gsm8k",
  "bad_cases": [],
  "metrics": {
    "accuracy": 0.1234,
    "stderr": 0.0091
  }
}
```

`score` is consumed by `DownstreamTrainerWrapper`. `bad_cases` remains empty
because the bridge intentionally does not parse per-sample benchmark logs back
into the search loop.

`run_manifest.json` records the command, dataset path, selected environment
variables, and script digest for reproducibility.

## Testing

```bash
/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python -m pytest tests/ -q
```

Smoke-check the PTB evaluator with a tiny `PTB_DEV_LIMIT` before a long run if
you changed environments or HF cache settings.

## Repository Layout

```text
configs/default.yaml                 # current PTB official default config
data/                                # local references and decontamination mirrors
dataflow_evolver/                    # orchestrator, agents, wrappers, MCTS
scripts/posttrainbench_sft_eval.py   # SFT + official PTB evaluation bridge
scripts/ptb_benchmarks.py            # benchmark metadata and eval argv mapping
scripts/run_posttrainbench_local.sh  # convenience runner
docs/                               # design and run-flow docs
tests/                              # unit tests
workspace/                          # run artifacts, _agent_root, data_links
```
