# DataFlow Evolver

[English](README.md) | [简体中文](README_zh.md)

DataFlow Evolver improves a data-processing pipeline over a fixed, user-provided raw
dataset. Every iteration follows one transparent feedback loop:

```text
incumbent pipeline -> challenger dataset -> Review Agent -> accept/reject
        ^                                                 |
        +------ compact feedback + decision history ------+
```

Each challenger starts from the best-so-far incumbent. A rejected pipeline remains in a
compact evolution history so later iterations can avoid repeating the same strategy, but
its code is never inherited. After the configured iterations, a human can train and
evaluate the selected incumbent dataset.

## Repository layout

```text
dataflow_evolver/                   framework source
configs/default.yaml
data/                               local datasets; ignored by Git
model/                              local models; ignored by Git
workspace/
  dataflow/                         tracked vendored DataFlow source
  .claude/skills/dataflow-dev/      Claude-native DataFlow coding skill
  .agents/skills/dataflow-dev/      Codex-native copy of the DataFlow skill
  .opencode/skills/dataflow-dev/    OpenCode-native copy of the DataFlow skill
  cache/das/                        shared embedding cache
  runs/<run_name>/
    manifest.json
    iterations.jsonl
    iteration_001/
      pipeline.py
      operators/
      cache/
      review.jsonl
      sessions/claude|opencode|codex|pipeline/
```

The framework starts Claude, OpenCode, or Codex with `workspace/` as the working
directory. Each backend uses a full copy under its native project skill location. The
three directories intentionally carry the same files, with only their internal
discovery-root paths adapted. Native backend
state and per-turn token usage are stored under each iteration's `sessions/<backend>/`.

## Install

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "setuptools<81"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[agent,agent-runtime,dataset-quality,dev]"
```

Install the `claude`, `opencode`, or `codex` CLI separately when selecting those
backends. Codex is invoked through `codex exec --json`; `CODEX_BIN` selects the binary.

## Configure

The main configuration is `configs/default.yaml`. At minimum, provide the raw dataset,
review API, unique run name, and task description:

```bash
export RAW_DATA_PATH=/path/to/raw-data/train.jsonl
export DF_API_URL=https://api.example/v1/chat/completions
export DF_API_KEY=...
export DF_API_MAX_WORKERS=10
export DF_RUN_NAME=experiment_001
export DF_MAX_ITERATIONS=4
export DF_EXECUTION_REPAIRS=1
export DF_COMPILE_TIMEOUT_SEC=120
export DF_AGENT_PYTHON="$(command -v python)"
```

`DF_MAX_ITERATIONS` controls review-guided pipeline iterations.
`DF_EXECUTION_REPAIRS` controls additional Agent repair calls after a generated pipeline
fails artifact validation, compile preflight, execution, or output validation. A repair
receives that stage, the traceback, and available per-step JSONL diagnostics. The Agent
only edits artifacts; the framework always runs the deterministic compile-only preflight
before allowing a full execution. Review feedback advances to the next normal iteration.

The framework does not impose a target sample count. Pipelines start from the complete raw
dataset and may clean, filter, transform, or augment it according to the task.

Use provider-neutral credentials for the selected PipelineAgent backend:

```bash
export DF_AGENT_API_KEY=...
export DF_AGENT_BASE_URL=https://provider-compatible.example/
```

The endpoint must implement the selected backend's protocol: an Anthropic-compatible
endpoint for Claude, the configured OpenCode provider protocol for OpenCode, and an
OpenAI Codex/Responses-compatible endpoint for Codex. These credentials are independent
of `DF_API_*`, which configures ReviewAgent and generated-pipeline LLM calls.

For Claude:

```bash
export AGENT_BACKEND=claude
export CLAUDE_AGENT_MODEL=claude-opus-4-8
```

For OpenCode:

```bash
export AGENT_BACKEND=opencode
export OPENCODE_BIN=opencode
export OPENCODE_MODEL=anthropic/<model-name>
export OPENCODE_PROVIDER=anthropic
```

For Codex:

```bash
export AGENT_BACKEND=codex
export CODEX_BIN=/path/to/codex
export CODEX_MODEL=gpt-5.4
export CODEX_EFFORT=high
```

Backend-specific `CLAUDE_AGENT_API_KEY_ENV`/`CLAUDE_AGENT_BASE_URL`,
`OPENCODE_API_KEY_ENV`/`OPENCODE_BASE_URL`, and
`CODEX_API_KEY_ENV`/`CODEX_BASE_URL` can override the generic values. Legacy `ANTHROPIC_API_KEY`,
`ANTHROPIC_BASE_URL`, `CODEX_API_KEY`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL` remain
accepted as fallbacks.

## Optional embedding-based dataset review

DAS compares a configurable candidate sample with a fixed high-quality proxy dataset.
The same candidate embeddings also produce threshold-free semantic-diversity evidence:

```text
DAS(candidate) = -MMD(candidate_embeddings, proxy_embeddings)
diversity(candidate) = cosine Vendi Score + nearest-neighbour cosine profile
```

Prepare the proxy data and embedding model locally, for example:

```text
data/ODA-Math-460k/
model/Qwen3-Embedding-8B/
```

Start an OpenAI-compatible embedding endpoint, then configure:

```bash
export DF_EMBEDDING_REVIEW_ENABLED=true
export DF_DAS_SAMPLE_SIZE=5000
export DF_DAS_PROXY_SOURCE=local
export DF_DAS_PROXY_PATH=$PWD/data/ODA-Math-460k
export DF_DAS_EMBEDDING_MODEL=$PWD/model/Qwen3-Embedding-8B
export DF_DAS_EMBEDDING_URL=http://127.0.0.1:8000/v1
```

The default candidate fields are `instruction` and `output`; the default proxy fields are
`question` and `response`. Change both mappings in the YAML for another schema. Candidate
embeddings are computed once per sampled dataset and reused by DAS, cosine Vendi Score,
and nearest-neighbour cosine statistics. These dataset-level signals require neither a
domain codebook nor a fixed semantic-duplicate threshold, and are not numerically mixed
directly. When dataset review is enabled, they first become a calibrated 0–1 embedding
quality score and then contribute the configured fraction of `review_score`:

```text
das_score = clip(1 - MMD / sqrt(2))
nn_score  = clip((1 - nearest_neighbor_cosine_p95) / 2)
embedding_quality = 0.40 * das_score + 0.30 * normalized_vendi + 0.30 * nn_score
review_score = (0.75 * llm_composite + 0.25 * embedding_quality) * hard_metric_factor
```

Set `DF_DATASET_QUALITY_SCORE_WEIGHT` to change the default `0.25` contribution.

## Run

```bash
python -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "Transform the supplied raw corpus into high-quality SFT records." \
  --log-level INFO
```

A non-empty existing `workspace/runs/<run_name>/` is rejected. Use a new run name rather
than mixing artifacts. The final `manifest.json` records the selected pipeline, dataset,
review, and all iterations. Token usage is stored per iteration in `token_usage.json`
under four separate namespaces: `agent`, `pipeline_llm`, `review_llm`, and `embedding`.
The Review API and embedding endpoint use provider-reported usage; missing usage is
marked explicitly instead of estimated. A fully cached embedding evaluation reports
zero requests and zero tokens for that iteration.

## Test

```bash
python -m pytest -q
```

See `docs/workflow.md` for review and repair semantics.
