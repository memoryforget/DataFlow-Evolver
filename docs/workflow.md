# Pipeline Review Workflow

The framework uses the following pipeline evolution workflow:

```text
fixed raw data
    -> PipelineAgent writes pipeline.py
    -> execute pipeline
    -> ReviewAgent scores output data
         - sample-level: sampled LLM judge
         - full-file: schema/duplicate/null/contamination metrics
         - dataset-level (optional): proxy-set DAS/MMD
    -> inject the complete structured ReviewResult into the next PipelineAgent prompt
    -> repeat until max_iterations
```

It does **not** run RecipeAgent, external-data/data-discovery MCP, tree search, downstream
training, or benchmark evaluation. Training and evaluation are
manual follow-up steps after the review loop finishes.

## Inputs and outputs

- `project.input_dir` is required and must contain raw `jsonl/json/csv/parquet` data.
- `task.target_schema` is required and is never inferred at runtime.
- `loop.max_iterations` is the total number of candidates, including `node_0`.
- Every run needs a fresh `project.run_name`; a non-empty existing run directory is
  rejected to prevent stale files from contaminating the result.
- `<workspace>/<run_name>_iterations.jsonl` records every iteration.
- `<workspace>/<run_name>_best.json` points to the selected pipeline and dataset.

Selection prefers candidates that pass ReviewAgent's thresholds, then chooses the
highest `review_score`. Review failures do not trigger an opaque inner optimization
loop: their structured metrics, failure signals, and issues become the next
PipelineAgent prompt directly. Runtime crashes may receive the small number of
code-repair retries configured by `loop.execution_retries`.

## Dataset-level DAS/MMD review

`review.dataset_quality` enables a deterministic distribution-level evaluator modeled on
Data-Preparation-Bench. It maps both candidate and proxy rows to the same two-turn
`user`/`assistant` conversation, embeds a fixed-size sample, and computes biased RBF-MMD:

```text
DAS(candidate) = -MMD(candidate_embeddings, proxy_embeddings)
```

The default math proxy contract is ODA-Math-460k (`question` / `response`) and the
candidate contract is `instruction` / `output`. The sample size is configurable and
defaults to 5000 with seed 42. `require_full_sample: true` deliberately reports
`insufficient_samples` instead of comparing nodes at different sample sizes.

Embeddings are L2-normalized before MMD, and embedding batches are cached by their full
conversation content plus model/runtime identity. In particular, the fixed proxy is not
re-embedded on every iteration. Proxy records themselves are never included in the
PipelineAgent prompt: the next iteration receives only MMD/DAS, delta versus its parent,
kernel terms, counts, and error status.

Raw DAS is not added to `review_score`, because DAS is non-positive and its scale depends
on the embedding/kernel setup whereas the sampled LLM score is in `[0, 1]`. It is a
separate feedback signal; `review_score` and its existing selection semantics remain
backward compatible.

Start a Qwen3-Embedding-8B OpenAI-compatible chat embedding endpoint, then enable DAS:

```bash
export DF_DAS_ENABLED=true
export DF_DAS_SAMPLE_SIZE=5000
export DF_DAS_PROXY_SOURCE=local
export DF_DAS_PROXY_PATH=$PWD/data/ODA-Math-460k
export DF_DAS_PROXY_NAME=
export DF_DAS_EMBEDDING_MODEL=$PWD/model/Qwen3-Embedding-8B
export DF_DAS_EMBEDDING_URL=http://127.0.0.1:8000/v1
export DF_DAS_EMBEDDING_CONCURRENCY=128
```

For a pre-downloaded JSONL proxy, set `DF_DAS_PROXY_SOURCE=local` and point
`DF_DAS_PROXY_PATH` to the file. The configured proxy field names still apply.

## PipelineAgent backends

`agent.backend` selects the autonomous coding runtime:

- `claude` uses `claude-agent-sdk` and remains the default.
- `opencode` invokes `opencode run --format json` and parses its event stream.

The switch only changes the agent harness. Both implementations receive the same
pipeline prompt and must write `pipeline.py` plus `decision.json`; execution and review
remain unchanged. Both expose the same `dataflow-operators` MCP. OpenCode starts it as a
local stdio MCP subprocess, while Claude loads it in-process.

Agent state is isolated per node and stored centrally below
`workspace/agent_sessions/<run_name>/node_N/`. Claude writes JSONL sessions in the
`claude/.claude/projects/` subtree, while OpenCode writes its SQLite database in the
`opencode/.local/share/opencode/` subtree. The framework also writes
`agent_sessions.jsonl` and cumulative `agent_token_usage.json` in the node artifact
directory. Set `DF_AGENT_SESSION_ROOT` to select another central root; the relative
run/node path and backend name are retained below it.

Generated-pipeline LLM calls are accounted separately. The execution subprocess installs
a process-local `requests` response hook and appends metadata plus exact API usage to
`pipeline/pipeline_llm_calls.jsonl` under the same central node directory. It never records
prompt or response content. After every execution—including failures and self-correction
retries—the wrapper refreshes `pipeline_llm_token_usage.json` and the namespaced
`token_usage.json` in the node artifact directory. Calls whose endpoint omits usage remain
visible as `missing_usage` but are not added as estimated exact tokens.

OpenCode discovers the existing `.claude/skills/dataflow-dev/` directory through its
Claude-compatible skill discovery, so a duplicated `.opencode/skills/` copy is unnecessary.
The inline OpenCode permissions allow local code inspection, editing and compile tests but
deny web search, web fetch, questions and subagents.

## Launch

The configured `agent_root` must already contain the vendored `dataflow/` source and
`.claude/skills/dataflow-dev/`, as in the project's prepared `workspace/_agent_root`.

```bash
cd /path/to/DataFlow-Evolver

export RAW_DATA_DIR=/absolute/path/to/raw_data
export RAW_DATA_ENTRY=train.jsonl        # optional if the directory has one clear entry
export DF_RUN_NAME=review_only_001
export DF_MAX_ITERATIONS=4
export DF_API_URL=https://your-api-endpoint/v1
export DF_API_KEY=...                     # ReviewAgent and pipeline operators
export ANTHROPIC_API_KEY=...              # Claude/OpenCode Anthropic provider
export ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint/

python \
  -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "Clean and transform the supplied raw data into high-quality SFT records." \
  --log-level INFO
```

For OpenCode, add:

```bash
export AGENT_BACKEND=opencode
export OPENCODE_BIN=opencode
export OPENCODE_MODEL=anthropic/<model-name>
# Optional provider-specific reasoning variant:
export OPENCODE_VARIANT=max
```

`OPENCODE_API_KEY_ENV` defaults to `ANTHROPIC_API_KEY`, and `OPENCODE_BASE_URL`
defaults to `ANTHROPIC_BASE_URL` for the Anthropic provider. Credentials are referenced
by environment-variable name and are not embedded in inline configuration or command-line
arguments.

After the loop, inspect `<workspace>/<run_name>_best.json` and manually pass
`best.dataset_path` to the desired training/evaluation command. Those results are
outside the automated feedback loop.
