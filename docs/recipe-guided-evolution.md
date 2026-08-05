# Recipe-guided pipeline evolution (MVP)

This MVP separates data-distribution planning from pipeline implementation while keeping the
existing PipelineAgent, External Data MCP tools, arbitrary custom operators, MCTS, and downstream
evaluation intact.

## Ownership and execution order

1. **RecipeAgent** proposes `node_N/recipe.json`. It may inspect the task,
   available corpus, parent recipe, and (when enabled) external-data options. It may not write
   pipeline code during this stage. This stage has its own system-prompt append; it does not inherit
   the implementation-stage instruction to write operators and compile a pipeline. Transient-error
   resume messages are phase-aware and keep the same recipe-only boundary.
2. **Framework `RecipeValidator`** parses and validates the proposal deterministically. It checks
   mode availability, sample budget, coverage buckets, proposal strategies, verification, and
   selection policy. In synthetic mode it rejects fixed-prompt/index variants and requires at least
   two genuine proposal strategies. External artifacts must be non-empty JSONL files under the
   managed `data_links/` directory with a valid `.meta.json` sidecar. Invalid recipes go back to
   RecipeAgent with exact errors.
3. The framework canonicalizes the accepted recipe, writes it back to disk, computes its SHA-256,
   and freezes that contract for the node.
4. **PipelineAgent** receives the validated recipe and generates
   `pipeline.py`, `operators/*.py`, and `decision.json`. It is the implementer, not the approver.
5. Existing compile/execution checks validate that the pipeline runs. ReviewAgent validates the
   resulting dataset. Training/evaluation remains the final downstream evidence.

In short: RecipeAgent proposes, PipelineAgent implements, and framework code approves the plan;
execution, review, and downstream evaluation approve the outcome.

## Modes

- `external`: declares at least one licensed `input_corpus` or `external_dataset` source and emits
  only external rows. It primarily selects, cleans, maps, filters, and rewrites real data.
- `hybrid`: declares real sources plus a `mixing_policy`, preserves external seed rows, and adds
  controlled synthetic rows whose `_seed_id` resolves to a retained external `_record_id`.
- `synthetic`: declares `sources: []`, may not reference external data, builds candidates from
  multiple semantic strategies and coverage buckets, then verifies, deduplicates, and selects them.

Mode availability is bounded by actual capabilities: without a corpus or External Data MCP only
`synthetic` is valid; a grounded corpus cannot be discarded in favor of pure synthesis.

Every final row carries a unique `_record_id`, `_source_type`, and `_source_id`. The framework
cross-checks source declarations in three places: recipe validation, a pre-execution static scan of
pipeline data references, and a post-execution full-dataset audit. Hybrid additionally enforces the
minimum external/synthetic fractions in `mixing_policy` and requires 100% synthetic-to-seed linkage.
PipelineAgent may read one or many recipe sources directly, or materialize a normalized derived
file before the final pipeline runs. Because static source scanning cannot prove a transitive
`derived file -> recipe sources` relationship, preflight does not require literal path-set equality.
It still rejects unmanaged direct downloads and invalid managed files; any external reference fails
synthetic. Strict transitive source lineage requires a future derived-artifact manifest contract.

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
