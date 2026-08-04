# Recipe-guided pipeline evolution (MVP)

This MVP separates data-distribution planning from pipeline implementation while keeping the
existing PipelineAgent, External Data MCP tools, arbitrary custom operators, MCTS, and downstream
evaluation intact.

## Ownership and execution order

1. **PipelineAgent / planning stage** proposes `node_N/recipe.json`. It may inspect the task,
   available corpus, parent recipe, and (when enabled) external-data options. It may not write
   pipeline code during this stage. This stage has its own system-prompt append; it does not inherit
   the implementation-stage instruction to write operators and compile a pipeline. Transient-error
   resume messages are phase-aware and keep the same recipe-only boundary.
2. **Framework `RecipeValidator`** parses and validates the proposal deterministically. It checks
   mode availability, sample budget, coverage buckets, proposal strategies, verification, and
   selection policy. In synthetic mode it rejects fixed-prompt/index variants and requires at least
   two genuine proposal strategies. Invalid recipes go back to the PipelineAgent with exact errors.
3. The framework canonicalizes the accepted recipe, writes it back to disk, computes its SHA-256,
   and freezes that contract for the node.
4. **The same PipelineAgent / implementation stage** receives the validated recipe and generates
   `pipeline.py`, `operators/*.py`, and `decision.json`. It is the implementer, not the approver.
5. Existing compile/execution checks validate that the pipeline runs. ReviewAgent validates the
   resulting dataset. Training/evaluation remains the final downstream evidence.

In short: the Agent proposes and implements; framework code approves the plan; execution, review,
and downstream evaluation approve the outcome.

## Modes

- `external`: primarily selects, cleans, maps, filters, and rewrites real input/external data.
- `hybrid`: uses real data as seeds and adds controlled synthesis.
- `synthetic`: builds candidates from multiple semantic strategies and coverage buckets, then
  verifies, normalizes/deduplicates, and selects them.

Mode availability is bounded by actual capabilities: without a corpus or External Data MCP only
`synthetic` is valid; a grounded corpus cannot be discarded in favor of pure synthesis.

## Dataset feedback added by the MVP

For hybrid/synthetic recipes every output row carries `_bucket`, `_generation_strategy`, and
`_operation_graph`. ReviewAgent scans the complete JSONL and records:

- normalized duplicate rate (surface formatting and numeric substitutions normalized),
- structural duplicate rate from operation graphs/archetypes,
- recipe coverage score against the planned bucket distribution,
- largest bucket fraction.

Normalized duplication, structural concentration, and coverage are connected to Tier-1 reward when
a hybrid/synthetic recipe is active. Exact and normalized duplication use the worse of the two rates
instead of double-penalizing the same duplicate. Missing audit metadata fails review rather than
silently awarding a diversity score.

## Deferred beyond the MVP

- semantic embedding deduplication,
- a reusable persona/program/archetype library,
- learned novelty or quality predictors,
- cross-node global coverage optimization,
- automatic source licensing policy beyond the existing External Data MCP audit trail.
