# Lightweight Review-Only Loop

This branch provides a deliberately small experimental baseline:

```text
fixed raw data
    -> PipelineAgent writes pipeline.py
    -> execute pipeline
    -> ReviewAgent scores output data
    -> inject the complete structured ReviewResult into the next PipelineAgent prompt
    -> repeat until max_iterations
```

It does **not** run RecipeAgent, external-data/data-discovery MCP, MCTS, downstream
training, Tier-2 selection, or benchmark evaluation. Training and evaluation are
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
export ANTHROPIC_API_KEY=...              # PipelineAgent SDK
export ANTHROPIC_BASE_URL=http://123.129.219.111:3000/

/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python \
  -m dataflow_evolver.lightweight_main \
  --config configs/lightweight_review.yaml \
  --task "Clean and transform the supplied raw data into high-quality SFT records." \
  --log-level INFO
```

After the loop, inspect `<workspace>/<run_name>_best.json` and manually pass
`best.dataset_path` to the desired training/evaluation command. Those results are
intentionally outside this baseline's feedback loop.
