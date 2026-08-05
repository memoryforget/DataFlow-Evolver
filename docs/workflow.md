# Pipeline Review Workflow

The framework uses the following pipeline evolution workflow:

```text
fixed raw data
    -> PipelineAgent writes pipeline.py
    -> execute pipeline
    -> ReviewAgent scores output data
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

## PipelineAgent backends

`agent.backend` selects the autonomous coding runtime:

- `claude` uses `claude-agent-sdk` and remains the default.
- `opencode` invokes `opencode run --format json` and parses its event stream.

The switch only changes the agent harness. Both implementations receive the same
pipeline prompt and must write `pipeline.py` plus `decision.json`; execution and review
remain unchanged. Both expose the same `dataflow-operators` MCP. OpenCode starts it as a
local stdio MCP subprocess, while Claude loads it in-process.

OpenCode discovers the existing `.claude/skills/dataflow-dev/` directory through its
Claude-compatible skill discovery, so a duplicated `.opencode/skills/` copy is unnecessary.
The inline OpenCode permissions allow local code inspection, editing and compile tests but
deny web search, web fetch, questions and subagents.

## Launch

The configured `agent_root` must already contain the vendored `dataflow/` source and
`.claude/skills/dataflow-dev/`, as in the project's prepared `workspace/_agent_root`.

```bash
cd /vepfs-mlp2/c20250602/500050/lh/xqf/DataFlow-Evolver

export RAW_DATA_DIR=/absolute/path/to/raw_data
export RAW_DATA_ENTRY=train.jsonl        # optional if the directory has one clear entry
export DF_RUN_NAME=review_only_001
export DF_MAX_ITERATIONS=4
export DF_API_KEY=...                     # ReviewAgent and pipeline operators
export ANTHROPIC_API_KEY=...              # Claude/OpenCode Anthropic provider
export ANTHROPIC_BASE_URL=http://123.129.219.111:3000/

/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python \
  -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "Clean and transform the supplied raw data into high-quality SFT records." \
  --log-level INFO
```

For OpenCode, add:

```bash
export AGENT_BACKEND=opencode
export OPENCODE_BIN=/vepfs-mlp2/c20250602/500050/lh/xqf/.npm-global/bin/opencode
export OPENCODE_MODEL=anthropic/claude-opus-4-8
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
