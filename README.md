# DataFlow Evolver

DataFlow Evolver iteratively improves a DataFlow pipeline over a user-provided raw dataset.
Each iteration generates a pipeline, executes it, reviews the resulting dataset, and feeds
structured review evidence into the next iteration.

```text
raw data
  -> PipelineAgent
  -> DataFlow pipeline
  -> candidate dataset
  -> ReviewAgent
  -> next PipelineAgent prompt
```

Review combines:

- sampled LLM assessment of correctness, relevance, difficulty, diversity, and schema;
- full-dataset metrics such as parse, null, duplicate, and contamination rates;
- optional dataset-level DAS/MMD against a local proxy dataset.

PipelineAgent supports both Claude Agent SDK and OpenCode. The generated artifact contract
is the same for both backends.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install "setuptools<81"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[agent,agent-runtime,dataset-quality]"
```

Install the CLI required by the selected PipelineAgent backend separately and make it
available on `PATH`:

- Claude Agent SDK backend: `claude`
- OpenCode backend: `opencode`

For development and tests, also install the `dev` extra:

```bash
python -m pip install -e ".[dev,agent,agent-runtime,dataset-quality]"
```

## Local data and model assets

Place the raw input dataset in any local directory and pass that directory through
`RAW_DATA_DIR`.

Dataset-level DAS is optional. To use the default math-domain setup, download the proxy
dataset and embedding model yourself and place their complete snapshots at:

```text
data/ODA-Math-460k/
model/Qwen3-Embedding-8B/
```

Both directories are ignored by Git. The framework does not download these assets at
runtime.

## Configuration

The default configuration is [`configs/default.yaml`](configs/default.yaml). At minimum,
set the raw-data path, API endpoint, API key, and a unique run name:

```bash
export RAW_DATA_DIR=/path/to/raw-data
export RAW_DATA_ENTRY=train.jsonl      # optional when discovery is unambiguous

export DF_API_URL=https://your-api-endpoint/v1
export DF_API_KEY=<your-api-key>

export DF_AGENT_PYTHON="$(command -v python)"
export DF_RUN_NAME=experiment_001
export DF_MAX_ITERATIONS=4
```

The default candidate schema is:

```json
{"instruction": "str", "output": "str"}
```

Change `task.target_schema` and the DAS candidate field mapping together when using a
different schema.

### Claude backend

```bash
export AGENT_BACKEND=claude
export ANTHROPIC_API_KEY=<your-api-key>
export ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint/
```

### OpenCode backend

```bash
export AGENT_BACKEND=opencode
export OPENCODE_BIN=opencode
export OPENCODE_MODEL=anthropic/<model-name>
export OPENCODE_API_KEY_ENV=ANTHROPIC_API_KEY

export ANTHROPIC_API_KEY=<your-api-key>
export ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint/
```

### Dataset-level DAS/MMD

Start an OpenAI-compatible embedding service from the local model snapshot, then enable
DAS:

```bash
export DF_DAS_ENABLED=true
export DF_DAS_SAMPLE_SIZE=5000
export DF_DAS_SAMPLE_SEED=42

export DF_DAS_PROXY_SOURCE=local
export DF_DAS_PROXY_PATH=./data/ODA-Math-460k
export DF_DAS_EMBEDDING_MODEL=./model/Qwen3-Embedding-8B
export DF_DAS_EMBEDDING_URL=http://127.0.0.1:8000/v1
export DF_DAS_EMBEDDING_API_KEY=EMPTY
```

The embedding endpoint must accept `POST /v1/embeddings` with chat `messages`. By default,
candidate `instruction`/`output` and proxy `question`/`response` fields are converted into
the same user/assistant conversation format before embedding.

DAS is reported separately from the LLM `review_score`. Its MMD value, parent delta,
kernel terms, sample counts, and status are returned to the next PipelineAgent without
exposing proxy examples.

## Agent sessions and token usage

Claude and OpenCode session state is isolated per pipeline node and stored centrally in
the project workspace:

```text
workspace/agent_sessions/<run_name>/node_N/claude/.claude/projects/**/*.jsonl
workspace/agent_sessions/<run_name>/node_N/opencode/.local/share/opencode/opencode.db
workspace/agent_sessions/<run_name>/node_N/pipeline/pipeline_llm_calls.jsonl
```

Each completed Agent invocation refreshes:

```text
<node_dir>/agent_sessions.jsonl
<node_dir>/agent_token_usage.json
<node_dir>/pipeline_llm_token_usage.json
<node_dir>/token_usage.json
```

Agent usage is read from native Claude/OpenCode sessions. Pipeline usage is captured from
the raw API `usage` object before DataFlow converts the response to text. Successful retry
attempts are all counted; responses without usage are reported as `missing_usage` and are
not silently estimated. `token_usage.json` namespaces both accounts without combining
incompatible cache semantics into a fabricated grand total.

To use another central root, set:

```bash
export DF_AGENT_SESSION_ROOT=./agent_sessions
```

The framework retains the run/node/backend hierarchy below this root, so concurrent
experiments do not share Claude project logs or an OpenCode database. Session directories
may contain full prompts and responses and should be treated as private experiment data.
Pipeline call events contain request metadata and token counters, but no prompt or response
content; URL query strings are removed before writing.

## Run

Run from the repository root:

```bash
python -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "Transform the supplied raw corpus into high-quality SFT records." \
  --log-level INFO
```

Use a new `DF_RUN_NAME` for every run. A non-empty existing run directory is rejected to
prevent stale node artifacts from entering a new experiment.

## Outputs

The default workspace is `workspace_review_only/`:

- `<run_name>_iterations.jsonl`: all candidates and structured reviews;
- `<run_name>_best.json`: the selected pipeline and candidate dataset;
- node `pipeline.py` and operator files: generated DataFlow implementation;
- node `decision.json`: pipeline structure and rationale;
- node `review.jsonl`: sampled evidence, hard metrics, and optional DAS evidence;
- node `token_usage.json`: Agent and pipeline-runtime LLM usage.

## Tests

```bash
python -m pytest -q
```

See [`docs/workflow.md`](docs/workflow.md) for detailed feedback semantics and configuration.
