# DataFlow Evolver

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
  .claude/skills/dataflow-dev/      canonical Claude/OpenCode coding skill
  cache/das/                        shared embedding cache
  runs/<run_name>/
    manifest.json
    iterations.jsonl
    iteration_001/
      pipeline.py
      operators/
      cache/
      review.jsonl
      sessions/claude|opencode|pipeline/
```

The framework starts both Claude and OpenCode with `workspace/` as their working directory.
Claude loads the project skill from `workspace/.claude/skills/`; OpenCode discovers the
same Claude-compatible skill, so no second copy is maintained. Native Claude/OpenCode
state is stored under each iteration's `sessions/` directory and is separate from the
project-level skill.

## Install

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "setuptools<81"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[agent,agent-runtime,dataset-quality,dev]"
```

Install either `claude` or `opencode` separately and make the selected CLI available on
`PATH`.

## Configure

The main configuration is `configs/default.yaml`. At minimum, provide the raw dataset,
review API, unique run name, and task description:

```bash
export RAW_DATA_DIR=/path/to/raw-data
export RAW_DATA_ENTRY=train.jsonl       # optional if discovery is unambiguous
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

For Claude:

```bash
export AGENT_BACKEND=claude
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=https://anthropic-compatible.example/
```

For OpenCode:

```bash
export AGENT_BACKEND=opencode
export OPENCODE_BIN=opencode
export OPENCODE_MODEL=anthropic/<model-name>
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=https://anthropic-compatible.example/
```

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
review, and all iterations. Token usage is stored per iteration in `token_usage.json`,
with Agent and pipeline-runtime calls kept as separate namespaces.

## Test

```bash
python -m pytest -q
```

See `docs/workflow.md` for review and repair semantics.
