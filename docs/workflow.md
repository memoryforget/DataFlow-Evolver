# Pipeline review workflow

The framework performs an observable best-so-far optimization loop:

```text
fixed raw data
  -> Pipeline Agent challenges the current incumbent
  -> deterministic artifact validation
  -> framework-owned compile preflight
  -> formal execution
  -> Review Agent evaluates the candidate dataset
  -> accept only if passed/review_score rank improves
  -> every N rounds, optionally SFT/evaluate the current incumbent in an external process
  -> compact decision outcome enters history; accepted pipeline becomes incumbent
  -> repeat
```

## Iterations and selection

- `loop.max_iterations` is the total number of generated pipeline iterations.
- Each iteration lives at `workspace/runs/<run_name>/iteration_XXX/`.
- Every challenger is generated from the best-so-far incumbent, never from a rejected
  pipeline. Rejected decision summaries and issues remain visible so the Agent avoids
  repeating failed strategies without receiving old pipeline code.
- Acceptance first prefers candidates that pass review thresholds, then compares
  `review_score`. The final dataset is the last accepted incumbent, not necessarily the
  final iteration's dataset.
- `manifest.json` and `iterations.jsonl` live beside the iteration directories.
- Optional downstream checkpoints run only after static selection, so they always receive
  the current incumbent rather than the latest challenger. See
  [`downstream_evaluation.md`](downstream_evaluation.md) for the process protocol.

## Execution repair

Every generated or repaired pipeline is checked in a fixed order:

1. Parse `pipeline.py` and require `compile -> DF_COMPILE_ONLY guard -> forward`.
2. Run `DF_COMPILE_ONLY=1 pipeline.py` with `loop.compile_timeout_sec`.
3. Only after compile succeeds, run the full pipeline with `loop.execution_timeout_sec`.
4. Validate that the final DataFlow JSONL exists and is non-empty.

The Agent writes or repairs artifacts but never runs the generated pipeline itself. A failed
check is classified as `artifact_validation`, `compile`, `execution`, or `output_validation`.
One generated pipeline attempt is followed by at most
`loop.execution_repairs` additional repair calls. Each repair prompt receives:

- the failure stage;
- the execution traceback;
- every fresh JSONL artifact's size and non-empty row count;
- the first valid row's field names;
- JSON parse errors.

The framework reruns the complete fixed check sequence after each repair. If the highest
DataFlow step is empty, execution fails; the framework never substitutes a non-empty
intermediate step as the final dataset.

## Review signals

Review combines three layers:

1. A sampled LLM assessment of per-sample correctness, relevance, difficulty, and schema.
2. Full-file deterministic metrics for parsing, required fields, nulls, duplicates, and
   optional decontamination.
3. Optional dataset-level embedding evidence: DAS/MMD against a fixed local proxy,
   cosine Vendi Score, and a nearest-neighbour cosine profile.

DAS embeds deterministic samples from the candidate and proxy with the same two-turn
conversation format, then computes biased RBF-MMD. The candidate embeddings are reused
for threshold-free, codebook-free semantic diversity: cosine Vendi Score estimates the
effective number of distinct samples, while nearest-neighbour cosine quantiles expose
local concentration without selecting a SemDeDup cutoff. The sample size and seed are
configurable; `require_full_sample: true` reports insufficient data instead of comparing
different sample sizes. Embeddings are cached in `workspace/cache/das/`.

For selection, the bounded RBF-MMD is mapped to `clip(1-MMD/sqrt(2))`, normalized Vendi is
already in `[0,1]`, and NN-cosine p95 is mapped to `clip((1-p95)/2)`. Their configurable
weighted mean contributes 25% of `review_score` by default; the four sampled LLM dimensions
contribute 75%, after which full-file penalties are applied. Raw metrics remain in review
evidence for audit.

The next Pipeline Agent receives only the four sampled dimensions, duplicate/null/parse/
contamination rates, delta MMD, normalized Vendi and its delta, NN-cosine p95 and its delta,
and concrete issues. It does not receive MMD kernel terms or the remaining cosine telemetry.

When periodic downstream evaluation is enabled, the latest validated benchmark scores and
three complete bad cases per benchmark are added as a separate prompt block. The block is
reused until the next checkpoint. It is diagnostic feedback and is not mixed into static
`review_score`.

## Agent backends and state

Both Claude Agent SDK and OpenCode receive the same prompt and must produce the same
artifact contract. `workspace/` is their single project root: tracked `.claude/` and
`dataflow/` directories provide static resources, while runtime state lives under the
current run and iteration:

```text
iteration_XXX/sessions/claude/
iteration_XXX/sessions/opencode/
iteration_XXX/sessions/pipeline/
```

Claude uses the iteration-local directory directly as `CLAUDE_CONFIG_DIR`. OpenCode uses
iteration-local XDG config, cache, and data directories. Native session records are used
for Agent token accounting. Generated-pipeline API usage is captured independently and
stored without prompt/response bodies.
