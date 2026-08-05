# DataFlow Evolver

The framework evolves data-processing pipelines through a direct review feedback loop:

```text
fixed raw data -> PipelineAgent -> DataFlow pipeline -> candidate JSONL
                      ^                              |
                      +------ ReviewResult ---------+
```

`PipelineAgent` can run through either Claude Agent SDK (default) or OpenCode. Both
backends receive the same prompt, operator MCP, skill, and artifact contract.

There is no RecipeAgent, web/external-data MCP, tree search, downstream training, or
benchmark evaluation in the automated loop. Each iteration starts from the previous pipeline
and receives the previous ReviewAgent result directly in its prompt. After the configured
number of iterations, the framework selects a review-passing candidate with the highest
Review score. Training and evaluation are manual follow-up steps.

## Run

The repository includes a prepared `workspace/_agent_root` containing DataFlow source and
the `dataflow-dev` skill.

```bash
export RAW_DATA_DIR=/absolute/path/to/raw_data
export RAW_DATA_ENTRY=train.jsonl          # optional when auto-selection is unambiguous
export DF_RUN_NAME=review_only_001          # must be new for every run
export DF_MAX_ITERATIONS=4
export DF_API_KEY=...
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=http://123.129.219.111:3000/

python -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "Transform the supplied raw corpus into high-quality SFT records."
```

To run PipelineAgent with OpenCode instead:

```bash
export AGENT_BACKEND=opencode
export OPENCODE_BIN=/absolute/path/to/opencode
export OPENCODE_MODEL=anthropic/claude-opus-4-8
```

The run writes:

- `workspace_review_only/<run_name>_iterations.jsonl`: every candidate and review;
- `workspace_review_only/<run_name>_best.json`: the selected pipeline and dataset paths;
- `workspace/_agent_root/<run_name>/node_N/review.jsonl`: exact ReviewAgent evidence.

See [docs/workflow.md](docs/workflow.md) for the complete
configuration and feedback semantics.

## Test

```bash
python -m pytest -q
```
